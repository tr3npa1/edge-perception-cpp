#!/usr/bin/env python3
"""Matched FP16-only measurements; never reuse historical or partial runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
from typing import Any

from benchmark_release import (
    build_snapshot, capture, execute, failure_excerpt, file_record,
    hardware_snapshot, read_benchmark, save_json, sha256, source_images, utc_now,
)

BACKENDS = ("native-trt", "fused-trt")


def cells(full: bool) -> list[tuple[str, str, int]]:
    # Sequential latency is a separate measurement, never streaming residence time.
    result = [(backend, "latency", 1) for backend in BACKENDS]
    result += [(backend, "throughput", depth) for backend in BACKENDS
               for depth in (1, 3) if full or (backend, depth) in
               (("native-trt", 3), ("fused-trt", 1))]
    return result


def source_snapshot(root: Path) -> dict[str, Any]:
    paths = {path for directory in ("src", "include", "cmake")
             for path in (root / directory).rglob("*") if path.is_file()}
    paths.update(root / name for name in (
        "CMakeLists.txt", "CMakePresets.json", "tools/benchmark_fp16_matrix.py",
        "tools/benchmark_release.py", "tools/run_fp16_matrix.ps1"))
    records = [{"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
               for path in sorted(paths) if path.is_file()]
    return {"files": records, "sha256": hashlib.sha256(
        json.dumps(records, sort_keys=True).encode()).hexdigest()}


def variability(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {}
    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else None
    return {"median": statistics.median(values), "minimum": min(values),
            "maximum": max(values), "mean": mean, "sample_stddev": deviation,
            "cv_percent": 100 * deviation / mean if deviation is not None and mean else None}


def validate_report(report: dict[str, Any], backend: str, mode: str, depth: int,
                    source_count: int, iterations: int) -> list[str]:
    errors = []
    if report.get("schema_version") != 1 or report.get("duration_unit") != "milliseconds":
        errors.append("Unsupported report schema or duration unit")
    config = report.get("config", {})
    if config.get("retain_raw_samples") is not True or config.get("failure_policy") != "stop_on_first_failure":
        errors.append("Raw samples and stop-on-first-failure are required")
    policy = "synchronous" if mode == "latency" else (
        "native_per_slot_workers" if backend == "native-trt" else "fused_submit_collect")
    if report.get("execution_policy") != policy:
        errors.append(f"Unexpected execution policy: expected {policy}")
    if report.get("effective_pipeline_depth") != depth:
        errors.append("Requested depth was not executed (silent serialization is not accepted)")
    samples = report.get("samples", [])
    if (len(samples) != iterations or
            sorted(sample.get("iteration_index", -1) for sample in samples) != list(range(iterations))):
        errors.append("Missing or duplicate measured iteration indices")
    expected_frames = {index % source_count for index in range(iterations)}
    if ({sample.get("source_frame_index") for sample in samples} != expected_frames
            or any(sample.get("source_frame_index") != sample.get("iteration_index", -1) % source_count for sample in samples)):
        errors.append("Source frame indices do not match the frozen input sequence")
    if mode == "throughput":
        slots = {sample.get("pipeline_slot_index") for sample in samples}
        if slots != set(range(depth)):
            errors.append(f"Not every requested slot was exercised: {slots}")
        if backend == "native-trt":
            if report.get("execution_policy") != "native_per_slot_workers":
                errors.append("Native streaming did not use independent per-slot workers")
            if depth > 1 and (report.get("maximum_active_workers") or 0) < 2:
                errors.append("Native streaming did not demonstrate concurrent active workers")
    for field in ("mean_ms", "median_ms", "p95_ms"):
        value = report.get("primary_latency", {}).get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            errors.append(f"Invalid primary latency: {field}")
    fps = report.get("throughput", {}).get("frames_per_second")
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        errors.append("Invalid measured wall-clock FPS")
    return errors


def semantics(report: dict[str, Any], backend: str) -> dict[str, Any]:
    target = report["target"]
    runtime = target["runtime_info"]
    if backend == "native-trt":
        config = target["engine_config"]
        if (target["type"] != "inference_engine" or runtime["backend"] != "native_tensorrt"
                or runtime["artifact_precision"] != "fp16"):
            raise ValueError("Unexpected native backend or precision")
        return {"engine": str(Path(config["artifact_path"]).resolve()),
                "input_shape": runtime["model"]["input_shape"], "input_type": runtime["model"]["input_element_type"],
                "preprocess": target["preprocess_config"], "postprocess": target["postprocess_config"],
                "device": config["gpu"]["device_id"], "cuda_graph_requested": config["gpu"]["enable_cuda_graph"]}
    config = target["pipeline_config"]
    if (target["type"] != "native_tensorrt_pipeline" or config["enable_gpu_postprocess"] is not True
            or runtime["gpu_postprocess_enabled"] is not True or config["require_cuda_graph"] is not True):
        raise ValueError("Fused GPU postprocessing and fail-on-graph-capture-error are required")
    return {"engine": str(Path(config["engine_path"]).resolve()),
            "input_shape": runtime["input_shape"], "input_type": runtime["input_element_type"],
            "preprocess": config["preprocess"], "postprocess": config["postprocess"],
            "device": config["device_id"], "cuda_graph_requested": config["enable_cuda_graph"]}


def verify_evidence(manifest: dict[str, Any], root: Path) -> None:
    for key in ("executable", "engine"):
        if sha256(Path(manifest[key]["path"])) != manifest[key]["sha256"]:
            raise ValueError(f"{key} changed during measurement")
    if source_snapshot(root) != manifest["source_snapshot"]:
        raise ValueError("Source changed during measurement; do not mix builds")
    if build_snapshot(Path(manifest["executable"]["path"])) != manifest["build"]:
        raise ValueError("Build/runtime changed during measurement")
    for item in manifest["source_images"]:
        if sha256(Path(item["frozen"]["path"])) != item["sha256"]:
            raise ValueError("Frozen image changed during measurement")


def summarize(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    summary = []
    requested = {tuple(cell) for cell in manifest["cells"]}
    for backend, mode, depth in cells(True):
        runs = [run for run in manifest["runs"]
                if (run["backend"], run["mode"], run["depth"]) == (backend, mode, depth)]
        passed = [run for run in runs if run["status"] == "passed"]
        complete = len(passed) == manifest["protocol"]["repetitions"]
        status = "PASS" if complete else ("FAILED" if any(run["status"] == "failed" for run in runs) else "PENDING")
        if complete and manifest.get("status") == "failed" and not manifest.get("evidence_validated"):
            status = "UNVERIFIED"
        row = {"backend": backend, "mode": mode, "depth": depth, "status": status,
               "requested": (backend, mode, depth) in requested,
               "successful_repetitions": len(passed),
               "required_repetitions": manifest["protocol"]["repetitions"],
               "metrics": {}}
        for metric in ("mean_ms", "median_ms", "p95_ms", "fps"):
            row["metrics"][metric] = variability([run["metrics"][metric] for run in passed])
        summary.append(row)
    return summary


def write_reports(output: Path, manifest: dict[str, Any]) -> None:
    rows = summarize(manifest)
    save_json(output / "summary.json", {"status": manifest["status"],
        "diagnostic_only": manifest.get("diagnostic_only", False),
        "evidence_validated": manifest.get("evidence_validated", False), "cells": rows})

    def number(row: dict[str, Any], metric: str) -> str:
        value = row["metrics"][metric].get("median")
        return f"{value:.3f}" if value is not None and row["status"] == "PASS" else row["status"]

    lines = ["# FP16 native / fused matrix", "", f"Run status: **{manifest['status']}**.", "",
             "This report contains only this run's immutable engine, executable and input selection.",
             "Historical pre-correctness-fix measurements are not merged into this table.", "",
             "## Sequential end-to-end latency", "",
             "| Backend | Mean ms | p50 ms | p95 ms | Successful repetitions |",
             "|---|---:|---:|---:|---:|"]
    if manifest.get("diagnostic_only"):
        lines[4:4] = ["**DIAGNOSTIC ONLY: short/nonstandard protocol; do not publish as benchmark results.**", ""]
    for row in rows:
        if row["mode"] == "latency":
            lines.append(f"| {row['backend']} | {number(row, 'mean_ms')} | {number(row, 'median_ms')} | {number(row, 'p95_ms')} | {row['successful_repetitions']}/{row['required_repetitions']} |")
    lines += ["", "## Streaming throughput", "",
              "| Backend | Depth | FPS | Across-run FPS min–max | FPS sample CV | Frame residence p50 / p95 ms |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["mode"] != "throughput":
            continue
        fps = row["metrics"]["fps"]
        spread = f"{fps['minimum']:.3f}–{fps['maximum']:.3f}" if fps and row["status"] == "PASS" else row["status"]
        cv = f"{fps['cv_percent']:.2f}%" if fps.get("cv_percent") is not None and row["status"] == "PASS" else "N/A"
        lines.append(f"| {row['backend']} | {row['depth']} | {number(row, 'fps')} | {spread} | {cv} | {number(row, 'median_ms')} / {number(row, 'p95_ms')} |")
    lines += ["", "## Same-depth comparison", ""]
    for depth in (1, 3):
        same = [row for row in rows if row["mode"] == "throughput" and row["depth"] == depth]
        if all(row["status"] == "PASS" for row in same):
            values = {row["backend"]: row["metrics"]["fps"]["median"] for row in same}
            ratio = values["fused-trt"] / values["native-trt"]
            lines.append(f"- Depth {depth}: fused/native throughput ratio **{ratio:.3f}×** ({(ratio - 1) * 100:+.2f}%).")
        else:
            lines.append(f"- Depth {depth}: **PENDING** — both backends require complete matched repetitions; use `-FullMatrix`.")
    lines += ["", "## Methodology", "",
              "- Same YOLO26M FP16 engine, batch 1, frozen encoded images and the same executable/OpenCV decoder; frames are decoded and preloaded before measurement.",
              "- End-to-end is an in-memory BGR image through preprocessing, inference and postprocessing to detection objects. Decode, engine setup, annotation and file output are excluded.",
              "- Square 640×640 RGB letterbox, normalization 1/255, confidence 0.25, maximum 300 detections; native CPU and fused CUDA implementations preserve TASK 1 semantics.",
              "- Stage timers disabled; CUDA graphs requested for both backends, with capture errors fatal (fused uses --require-cuda-graph). Per-run evidence records fused measured graph-use counts. Native graph-use telemetry is unavailable; its requested setting is recorded, not claimed as observed per-frame capture.",
              "- Streaming FPS = completed frames / measurement wall time, including pipeline fill and drain. Streaming residence time is NOT sequential latency and FPS is NOT its reciprocal.",
              "- Cells run serially; each repetition reverses cell order to reduce fixed-order bias. Warmup is repeated in every fresh process. Native depths 1 and 3 use the same worker scheduler.",
              "- Headline numbers are medians of per-run statistics, not pooled percentiles. JSON also contains mean, range, sample standard deviation and coefficient of variation for every metric.",
              "- Concurrent native host operations establish genuine independent in-flight work, not guaranteed simultaneous GPU kernels; hardware/resource constraints determine kernel overlap.",
              f"- Protocol: `{json.dumps(manifest['protocol'], sort_keys=True)}`.", "",
              "## Evidence and failures", "",
              "See `manifest.json`, `summary.json` and `runs/*/{command,process,result}.json`; each run preserves full application JSON and retained frame samples. Input/source/executable/library SHA-256 hashes and GPU snapshots are retained."]
    for run in manifest["runs"]:
        if run["status"] != "passed":
            lines += ["", f"- `{run['id']}`: {run['reason']}"]
    if manifest.get("error"):
        lines += ["", f"Runner error: {manifest['error']}"]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"schema_version": 1, "status": "preparing", "started_utc": utc_now(),
        "cells": cells(args.full_matrix), "runs": [],
        "protocol": {key: getattr(args, key) for key in ("source_frames", "warmup", "latency_iterations", "throughput_iterations", "repetitions", "device")}}
    manifest["diagnostic_only"] = any(getattr(args, key) != value for key, value in (
        ("source_frames", 128), ("warmup", 50), ("latency_iterations", 300), ("throughput_iterations", 1000), ("repetitions", 3)))
    try:
        executable, engine = args.executable.resolve(strict=True), args.engine.resolve(strict=True)
        images = source_images(args.source.resolve(strict=True), args.source_frames)
        if len(images) != args.source_frames:
            raise ValueError(f"Need {args.source_frames} source images; found {len(images)}")
        frozen = output / "inputs"
        frozen.mkdir()
        for index, item in enumerate(images):
            path = Path(item["path"])
            destination = frozen / f"{index:06d}{path.suffix.lower()}"
            shutil.copyfile(path, destination)
            if sha256(destination) != item["sha256"]:
                raise ValueError(f"Source image changed while copying: {path}")
            item["frozen"] = file_record(destination)
        manifest.update(status="running", executable=file_record(executable), engine=file_record(engine),
                        source_images=images, source_snapshot=source_snapshot(root), build=build_snapshot(executable),
                        hardware=hardware_snapshot(root),
                        git_head=capture(["git", "rev-parse", "HEAD"], root),
                        git_status=capture(["git", "status", "--short"], root),
                        executable_version=capture([str(executable), "--version"], root))
        if manifest["hardware"]["gpu"].get("exit_code") != 0:
            raise ValueError("nvidia-smi failed; no verified NVIDIA GPU prerequisite")
        save_json(output / "manifest.json", manifest)
        write_reports(output, manifest)
        for repetition in range(1, args.repetitions + 1):
            sequence = list(manifest["cells"])
            if repetition % 2 == 0:
                sequence.reverse()
            for backend, mode, depth in sequence:
                verify_evidence(manifest, root)
                identity = f"{backend}_{mode}_d{depth}_r{repetition}"
                folder = output / "runs" / identity
                iterations = args.latency_iterations if mode == "latency" else args.throughput_iterations
                command = [str(executable), "--mode", "benchmark", "--source", str(frozen),
                    "--model", str(engine), "--backend", backend, "--precision", "fp16", "--device", str(args.device),
                    "--confidence", "0.25", "--max-detections", "300", "--pipeline-depth", str(depth),
                    "--output-dir", str(folder / "app_output"), "--benchmark-mode", mode,
                    "--timing-scope", "end-to-end", "--warmup", str(args.warmup), "--iterations", str(iterations),
                    "--benchmark-source-frames", str(args.source_frames), "--failure-policy", "stop",
                    "--retain-samples", "--no-stage-timings"]
                if backend == "fused-trt":
                    command.append("--require-cuda-graph")
                print(f"START {identity}", flush=True)
                process = execute(command, folder, root, args.timeout_seconds)
                record: dict[str, Any] = {"id": identity, "backend": backend, "mode": mode, "depth": depth,
                    "repetition": repetition, "folder": str(folder.relative_to(output)), **process}
                errors = []
                if process["exit_code"] != 0 or process["timed_out"]:
                    errors.append(f"Process failed: exit={process['exit_code']}, timed_out={process['timed_out']}")
                else:
                    report, errors = read_benchmark(folder, iterations, args.warmup, mode, "end-to-end", depth, False)
                    if report:
                        errors += validate_report(report, backend, mode, depth, args.source_frames, iterations)
                        try:
                            actual = semantics(report, backend)
                            if (actual["engine"] != str(engine) or actual["input_shape"] != [1, 3, 640, 640]
                                    or actual["input_type"] != "float16" or actual["device"] != args.device
                                    or actual["cuda_graph_requested"] is not True):
                                errors.append("Wrong engine, batch/shape, precision, device or graph request")
                            baseline = manifest.setdefault("verified_semantics", actual)
                            if actual != baseline:
                                errors.append("Pre/postprocessing or execution semantics differ across cells")
                        except (KeyError, TypeError, ValueError) as error:
                            errors.append(f"Cannot verify application semantics: {error}")
                        record["metrics"] = {key: report["primary_latency"][key] for key in ("mean_ms", "median_ms", "p95_ms")}
                        record["metrics"]["fps"] = report["throughput"]["frames_per_second"]
                        record["maximum_active_workers"] = report.get("maximum_active_workers")
                        if backend == "fused-trt":
                            record["measured_graph_frames"] = sum(
                                sample.get("used_cuda_graph") is True for sample in report["samples"])
                            record["measured_non_graph_frames"] = sum(
                                sample.get("used_cuda_graph") is False for sample in report["samples"])
                        record["target"] = report.get("target")
                        record["warnings"] = report.get("warnings")
                record.update(status="failed" if errors else "passed", reason="; ".join(errors))
                if errors:
                    record["log_excerpt"] = failure_excerpt(folder)
                save_json(folder / "result.json", record)
                manifest["runs"].append(record)
                save_json(output / "manifest.json", manifest)
                write_reports(output, manifest)
                print(f"{record['status'].upper()} {identity}", flush=True)
                if errors:
                    raise RuntimeError(f"{identity}: {record['reason']} (see preserved logs)")
        verify_evidence(manifest, root)
        manifest["evidence_validated"] = True
        manifest["status"] = "completed"
        return 0
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        print(f"STOPPED: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        manifest["finished_utc"] = utc_now()
        save_json(output / "manifest.json", manifest)
        write_reports(output, manifest)
        print(f"Report: {output / 'README.md'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-matrix", action="store_true")
    for name, default in (("source-frames", 128), ("warmup", 50), ("latency-iterations", 300),
                          ("throughput-iterations", 1000), ("repetitions", 3), ("timeout-seconds", 1800)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    for name in ("source_frames", "warmup", "latency_iterations", "throughput_iterations", "repetitions", "timeout_seconds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.throughput_iterations < 3 or args.warmup < 3:
        parser.error("Warmup and throughput iterations must each exercise all 3 slots")
    if args.device < 0:
        parser.error("--device must be nonnegative")
    try:
        return run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
