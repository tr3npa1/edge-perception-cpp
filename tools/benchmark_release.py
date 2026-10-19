#!/usr/bin/env python3
"""Run the release benchmark matrix serially and preserve auditable evidence.

Requires only Python's standard library and a built edge_perception executable.
Run this on an otherwise idle machine. Exit 0 means the matrix finished without
failed cells, 2 means it finished with recorded failures, and 1 is a runner error.
Latency, streaming throughput, and instrumented diagnostics remain separate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff", ".webp"}
GPU_QUERY = (
    "timestamp,name,uuid,driver_version,pstate,temperature.gpu,utilization.gpu,"
    "utilization.memory,memory.used,memory.total,power.draw,power.limit,"
    "clocks.current.graphics,clocks.current.memory"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size,
            "sha256": sha256(path), "mtime_ns": stat.st_mtime_ns}


def save_json(path: Path, value: Any) -> None:
    """Replace only this runner's report, avoiding partially written progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class OutputLock:
    """Hold an OS lock for the whole suite, including resume and final reports.

    The lock file may outlive the process; the OS lock cannot. A crashed process
    therefore leaves no stale PID decision for the next runner to guess about.
    """

    def __init__(self, output: Path) -> None:
        self.output = output
        self.handle: Any = None
        self.owner = {"pid": os.getpid(), "acquired_utc": utc_now()}

    def __enter__(self) -> OutputLock:
        self.handle = (self.output / ".runner.lock").open("a+b")
        try:
            self.handle.seek(0, os.SEEK_END)
            if self.handle.tell() == 0:
                self.handle.write(b"\0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            raise RuntimeError(f"Another runner holds the output lock: {self.output}") from error
        try:
            save_json(self.output / ".runner.owner.json", self.owner)
        except BaseException:
            self.handle.close()
            raise
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback_value: Any) -> None:
        try:
            self.owner["released_utc"] = utc_now()
            save_json(self.output / ".runner.owner.json", self.owner)
        finally:
            # Closing releases flock/byte-range locks even while unwinding an error.
            self.handle.close()


def identities(records: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    """Compare content and paths, ignoring harmless timestamp changes."""
    result = []
    for record in records:
        result.append((record["path"], record.get("sha256")))
        result.extend(identities(record.get("aliases", [])))
    return sorted(result)


def refresh_file_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refreshed = []
    for record in records:
        try:
            refreshed.append(file_record(Path(record["path"])))
        except OSError as error:
            refreshed.append({"path": record["path"], "error": str(error)})
        refreshed.extend(refresh_file_records(record.get("aliases", [])))
    return refreshed


def capture(command: list[str], cwd: Path, timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
        return {"command": command, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "exit_code": None, "error": str(error)}


def code_snapshot(root: Path) -> dict[str, Any]:
    tracked = capture(["git", "ls-files", "-z"], root)
    paths = []
    if tracked.get("exit_code") == 0:
        for relative in tracked["stdout"].split("\0"):
            if (relative.startswith(("src/", "include/", "cmake/"))
                    or relative in {"CMakeLists.txt", "CMakePresets.json"}):
                path = root / relative
                if path.is_file():
                    paths.append({"path": relative, "sha256": sha256(path)})
    serialized = json.dumps(paths, sort_keys=True).encode("utf-8")
    return {"head": capture(["git", "rev-parse", "HEAD"], root),
            "status": capture(["git", "status", "--short"], root),
            "compiled_source_files": paths,
            "compiled_source_sha256": hashlib.sha256(serialized).hexdigest()}


def discover_artifacts(models: Path) -> list[dict[str, Any]]:
    """Cover distinct artifact bytes; name aliases never masquerade as new tests."""
    discovered: list[dict[str, Any]] = []
    by_content: dict[tuple[str, str, str], dict[str, Any]] = {}
    paths = list((models / "onnx").glob("*.onnx")) + list((models / "engine").rglob("*.engine"))
    # Prefer the canonical fp32 name over identical calibration-export aliases.
    for path in sorted(paths, key=lambda item: (len(item.name), str(item))):
        match = re.search(r"(?:^|_)(fp32|fp16|int8)(?:_|\.|$)", path.name.lower())
        if not match:
            raise ValueError(f"Cannot determine precision from model filename: {path}")
        precision = match.group(1)
        record = file_record(path)
        artifact_type = path.suffix.lstrip(".")
        key = (artifact_type, precision, record["sha256"])
        if key in by_content:
            by_content[key]["aliases"].append(record)
            continue
        artifact = {**record, "id": f"{artifact_type}_{path.stem}",
                    "artifact_type": artifact_type, "precision": precision, "aliases": []}
        by_content[key] = artifact
        discovered.append(artifact)
    if not discovered:
        raise ValueError(f"No ONNX or TensorRT artifacts found below {models}")
    return discovered


def source_images(source: Path, limit: int) -> list[dict[str, Any]]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [file_record(source)]
    if not source.is_dir():
        raise ValueError("--source must be an image or image directory for reproducible frame selection")
    paths = sorted((path for path in source.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS), key=str)
    if not paths:
        raise ValueError(f"No supported images found in {source}")
    return [file_record(path) for path in paths[:limit]]


def hardware_snapshot(root: Path) -> dict[str, Any]:
    result = {"platform": platform.platform(), "uname": dict(platform.uname()._asdict()),
              "logical_cpu_count": os.cpu_count(), "python": sys.version,
              "gpu": capture(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv"], root),
              "gpu_full": capture(["nvidia-smi", "-q"], root)}
    if os.name == "nt":
        result["cpu_memory"] = capture([
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json; "
            "Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json"
        ], root)
    return result


def build_snapshot(executable: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"adjacent_libraries": [file_record(path) for path in
        sorted(executable.parent.glob("*.dll"))], "cache": None}
    for parent in executable.parents:
        cache = parent / "CMakeCache.txt"
        if cache.is_file():
            relevant = [line for line in cache.read_text(encoding="utf-8", errors="replace").splitlines()
                        if not line.startswith(("//", "#")) and re.search(
                            r"VERSION|BUILD_TYPE|CXX_COMPILER|CUDA|TRT|TENSORRT|ONNXRUNTIME|OpenCV", line)]
            result["cache"] = {**file_record(cache), "configuration": relevant}
            break
    return result


def terminate_child(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        process.kill()
    process.wait()


def execute(command: list[str], folder: Path, root: Path, timeout: int) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "command.json", {"argv": command, "cwd": str(root)})
    save_json(folder / "gpu_before.json", capture(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv"], root))
    started = utc_now()
    monotonic_start = time.perf_counter()
    result: dict[str, Any] = {"started_utc": started, "timeout_seconds": timeout,
                              "timed_out": False, "exit_code": None}
    process = None
    try:
        with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr)
            try:
                result["exit_code"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                terminate_child(process)
                result["exit_code"] = process.returncode
    except OSError as error:
        result["launch_error"] = str(error)
    except BaseException:
        if process is not None:
            terminate_child(process)
        raise
    finally:
        result["finished_utc"] = utc_now()
        result["process_wall_seconds"] = time.perf_counter() - monotonic_start
        save_json(folder / "process.json", result)
        save_json(folder / "gpu_after.json", capture(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv"], root))
    return result


def failure_excerpt(folder: Path) -> str:
    pieces = []
    for name in ("stderr.log", "stdout.log"):
        path = folder / name
        if path.is_file():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            significant = [line for line in lines if re.search(
                r"error|fail|unsupported|incompat|invalid|exception|could not|cannot", line, re.I)]
            pieces.extend(significant[-8:] or lines[-3:])
    return "\n".join(pieces)[-6000:]


def read_benchmark(folder: Path, iterations: int, warmup: int, mode: str,
                   scope: str, depth: int, stages: bool) -> tuple[dict[str, Any], list[str]]:
    # This new run directory is the sole search root: earlier results cannot leak in.
    candidates = list((folder / "app_output").rglob("benchmark.json"))
    if len(candidates) != 1:
        return {}, [f"Expected one benchmark.json in this run; found {len(candidates)}"]
    try:
        report = json.loads(candidates[0].read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"Cannot read generated benchmark report: {error}"]
    errors = []
    if report.get("successful") is not True or report.get("status") != "completed":
        errors.append("Application report is not successful/completed")
    expected = {"measured_iterations": iterations, "warmup_iterations": warmup,
                "mode": "synchronous_latency" if mode == "latency" else "streaming_throughput",
                "timing_scope": "backend_execution" if scope == "backend" else "end_to_end",
                "pipeline_depth": depth, "collect_stage_timings": stages}
    for key, value in expected.items():
        actual = report.get("config", {}).get(key)
        if actual != value:
            errors.append(f"Config {key}: expected {value!r}, received {actual!r}")
    for phase, requested in (("measurement", iterations), ("warmup", warmup)):
        counts = report.get("counts", {}).get(phase, {})
        if counts.get("completed") != requested or counts.get("failed") != 0:
            errors.append(f"Incomplete or failed {phase}: {counts}")
    if report.get("primary_latency", {}).get("sample_count") != iterations:
        errors.append("Primary latency sample count does not match requested iterations")
    if len(report.get("samples", [])) != iterations:
        errors.append("Retained sample count does not match requested iterations")
    if report.get("throughput", {}).get("completed_frames") != iterations:
        errors.append("Throughput completed frame count does not match requested iterations")
    return report, errors


def export_aggregates(output: Path, records: list[dict[str, Any]], repetitions: int) -> None:
    rows = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        row = {key: record.get(key) for key in (
            "run_id", "cell_id", "artifact", "precision", "backend", "test", "depth",
            "repetition", "status", "exit_code", "timed_out", "process_wall_seconds", "reason")}
        row.update(record.get("metrics", {}))
        rows.append(row)
        if record["test"] not in {"smoke", "stage_diagnostic"}:
            key = tuple(record.get(field) for field in ("cell_id", "test", "depth"))
            groups.setdefault(key, []).append(record)
    aggregate = []
    for (cell_id, test, depth), attempts in groups.items():
        good = [record for record in attempts if record["status"] == "passed"]
        row = {"cell_id": cell_id, "artifact": attempts[0]["artifact"],
               "backend": attempts[0]["backend"], "precision": attempts[0]["precision"],
               "test": test, "depth": depth, "successful_repetitions": len(good),
               "expected_repetitions": repetitions,
               "complete": len(good) == repetitions and len(attempts) == repetitions}
        for metric in ("mean_ms", "median_ms", "p95_ms", "p99_ms", "fps"):
            values = [item["metrics"][metric] for item in good if item.get("metrics", {}).get(metric) is not None]
            if values:
                row[f"median_of_run_{metric}"] = statistics.median(values)
                row[f"minimum_run_{metric}"] = min(values)
                row[f"maximum_run_{metric}"] = max(values)
        aggregate.append(row)
    save_json(output / "aggregate" / "runs.json", rows)
    save_json(output / "aggregate" / "summary.json", aggregate)
    for name, values in (("runs.csv", rows), ("summary.csv", aggregate)):
        if values:
            keys = list(dict.fromkeys(key for row in values for key in row))
            with (output / "aggregate" / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(values)


def measurement_plan(backend: str, precision: str, graph_ablation: bool) -> list[tuple[str, int]]:
    """Only native/fused pipelines support multiple independent in-flight slots."""
    tests = [("latency_e2e_clean", 1), ("throughput_e2e_clean", 1)]
    if backend == "fused-trt":
        tests += [("throughput_e2e_clean", 2), ("throughput_e2e_clean", 3)]
        if precision == "fp16" and graph_ablation:
            tests += [("latency_e2e_graph_off", 1), ("throughput_e2e_graph_off", 3)]
    else:
        tests.append(("latency_backend_clean", 1))
        if backend == "native-trt":
            tests.append(("throughput_e2e_clean", 3))
    return tests


def run_locked(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    executable = args.executable.resolve(strict=True)
    source = args.source.resolve(strict=True)
    models = args.models_dir.resolve(strict=True)
    output = args.output.resolve()
    artifacts = discover_artifacts(models)
    images = source_images(source, args.source_frames)
    before = code_snapshot(root)
    binary = file_record(executable)
    manifest: dict[str, Any] = {
        "schema_version": 1, "status": "running", "started_utc": utc_now(),
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "executable": binary, "runner": file_record(Path(__file__)), "code_before": before,
        "hardware": hardware_snapshot(root), "build": build_snapshot(executable),
        "executable_version": capture([str(executable), "--version"], root),
        "artifacts": artifacts, "source_frames": images, "source_selection":
            "First N supported images in recursive lexicographic path order, matching CLI preloading; decode excluded from timing.",
        "measurement_notes": [
            "All application processes run serially; model/session startup and warmup excluded from reported metrics.",
            "End-to-end means in-memory BGR image to detection objects, excluding decode, annotation, and output writing.",
            "Streaming FPS is completed frames / measured wall time; it is not 1000 / per-frame latency.",
            "Clean headline runs disable stage timers; stage diagnostics are separate and run once.",
            "Aggregate percentiles are medians of per-run percentiles, not pooled percentiles.",
            "ORT TensorRT may partition across TensorRT/CUDA/CPU; recorded providers do not establish placement of every node.",
            "Smoke output validates executable inference and serialization, not dataset accuracy.",
        ],
        "cells": [], "runs": [], "unsupported_tests": [],
    }
    if args.resume:
        previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        identity_fields = ("repetitions", "warmup", "latency_iterations", "throughput_iterations",
                           "source_frames", "device", "only_backends", "graph_ablation", "stage_iterations")
        for field in identity_fields:
            if previous["settings"].get(field) != manifest["settings"].get(field):
                raise ValueError(f"Cannot resume with changed setting: {field}")
        if previous["executable"]["sha256"] != binary["sha256"]:
            raise ValueError("Cannot resume: executable SHA256 changed")
        if previous["code_before"]["compiled_source_sha256"] != before["compiled_source_sha256"]:
            raise ValueError("Cannot resume: compiled source SHA256 changed")
        if previous["runner"]["sha256"] != manifest["runner"]["sha256"]:
            raise ValueError("Cannot resume: benchmark runner SHA256 changed")
        if identities(previous["build"]["adjacent_libraries"]) != identities(manifest["build"]["adjacent_libraries"]):
            raise ValueError("Cannot resume: adjacent runtime DLLs changed")
        for name in ("artifacts", "source_frames"):
            if identities(previous[name]) != identities(manifest[name]):
                raise ValueError(f"Cannot resume: {name} changed")
        manifest = previous
        manifest.setdefault("resumptions", []).append({"utc": utc_now(), "hardware": hardware_snapshot(root)})
        manifest["status"] = "running"
        manifest.pop("runner_error", None)
    save_json(output / "manifest.json", manifest)
    completed_by_id = {item["run_id"]: item for item in manifest["runs"]}

    def persist() -> None:
        manifest["updated_utc"] = utc_now()
        manifest["completed_run_count"] = len(manifest["runs"])
        manifest["failed_run_count"] = sum(item["status"] != "passed" for item in manifest["runs"])
        save_json(output / "manifest.json", manifest)
        export_aggregates(output, manifest["runs"], args.repetitions)

    def one(artifact: dict[str, Any], backend: str, test: str, depth: int = 1,
            repetition: int = 1) -> dict[str, Any]:
        cell_id = f"{artifact['id']}__{backend}"
        run_id = f"{cell_id}__{test}__d{depth}__r{repetition}"
        if run_id in completed_by_id:
            previous_run = completed_by_id[run_id]
            result_path = output / previous_run["folder"] / "result.json"
            if not result_path.is_file() or json.loads(result_path.read_text(encoding="utf-8")) != previous_run:
                raise ValueError(f"Cannot resume: finalized result missing or changed for {run_id}")
            print(f"RESUME {run_id}: {previous_run['status']}", flush=True)
            return previous_run
        folder = output / "runs" / hashlib.sha256(run_id.encode()).hexdigest()[:12]
        if folder.exists():
            # A killed runner can leave an unfinished directory. Preserve it and
            # retry in a new directory, never reuse a partial application's JSON.
            attempt = 2
            original = folder
            while folder.exists():
                folder = original.with_name(original.name + f"_attempt{attempt}")
                attempt += 1
            manifest.setdefault("incomplete_attempts", []).append(str(original.relative_to(output)))
        smoke = test == "smoke"
        mode = "throughput" if test.startswith("throughput") else "latency"
        scope = "backend" if test == "latency_backend_clean" else "end-to-end"
        stages = test == "stage_diagnostic"
        iterations = args.stage_iterations if stages else (args.throughput_iterations if mode == "throughput" else args.latency_iterations)
        command = [str(executable), "--mode", "infer" if smoke else "benchmark", "--source",
                   images[0]["path"] if smoke else str(source), "--model", artifact["path"],
                   "--backend", backend, "--precision", artifact["precision"],
                   "--device", str(args.device), "--confidence", "0.25", "--max-detections", "300",
                   "--pipeline-depth", str(depth), "--ort-cache-dir", str(output / "ort_cache" / artifact["id"]),
                   "--output-dir", str(folder / "app_output")]
        if test.endswith("graph_off"):
            command.append("--no-cuda-graph")
        elif backend == "fused-trt":
            command.append("--require-cuda-graph")
        if not smoke:
            command += ["--benchmark-mode", mode, "--timing-scope", scope,
                        "--warmup", str(args.warmup), "--iterations", str(iterations),
                        "--benchmark-source-frames", str(args.source_frames),
                        "--failure-policy", "stop", "--retain-samples"]
            if not stages:
                command.append("--no-stage-timings")
        manifest["current_run"] = {"run_id": run_id, "started_utc": utc_now()}
        persist()
        print(f"START {run_id}", flush=True)
        process = execute(command, folder, root, args.timeout_seconds)
        record: dict[str, Any] = {"run_id": run_id, "cell_id": cell_id,
            "artifact": artifact["id"], "backend": backend, "precision": artifact["precision"],
            "test": test, "depth": depth, "repetition": repetition,
            "folder": str(folder.relative_to(output)), **process}
        errors = []
        if process["timed_out"]:
            errors.append(f"Process timed out after {args.timeout_seconds} seconds")
        if process["exit_code"] != 0:
            errors.append(f"Process exit code: {process['exit_code']}")
        if process.get("launch_error"):
            errors.append(process["launch_error"])
        if smoke:
            detection_file = folder / "app_output" / "detections.jsonl"
            try:
                detections = [json.loads(line) for line in detection_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(detections) != 1:
                    errors.append(f"Smoke expected one frame record; received {len(detections)}")
                record["smoke_frame_records"] = len(detections)
                annotated = [path for path in (folder / "app_output").rglob("*")
                             if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
                record["annotated_outputs"] = [str(path.relative_to(output)) for path in annotated]
                if not annotated:
                    errors.append("No annotated smoke image was saved")
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"Cannot validate smoke detection output: {error}")
        elif process["exit_code"] == 0 and not process["timed_out"]:
            report, report_errors = read_benchmark(folder, iterations, args.warmup, mode, scope, depth, stages)
            errors.extend(report_errors)
            if report:
                if backend in ("native-trt", "fused-trt"):
                    from benchmark_fp16_matrix import validate_report
                    errors.extend(validate_report(report, backend, mode, depth, len(images), iterations))
                if backend == "fused-trt" and not test.endswith("graph_off") and any(
                        sample.get("used_cuda_graph") is not True for sample in report.get("samples", [])):
                    errors.append("A fused measured frame did not use the required CUDA graph")
                primary = report.get("primary_latency", {})
                throughput = report.get("throughput", {})
                record["metrics"] = {key: primary.get(key) for key in (
                    "mean_ms", "median_ms", "p95_ms", "p99_ms", "minimum_ms", "maximum_ms", "sample_count")}
                record["metrics"].update({"fps": throughput.get("frames_per_second"),
                    "measurement_wall_ms": throughput.get("wall_time_ms"),
                    "completed_frames": throughput.get("completed_frames"),
                    "effective_pipeline_depth": report.get("effective_pipeline_depth")})
                graph_samples = [sample["used_cuda_graph"] for sample in report.get("samples", [])
                                 if sample.get("used_cuda_graph") is not None]
                record["metrics"]["cuda_graph_used_samples"] = sum(graph_samples) if graph_samples else None
                record["metrics"]["cuda_graph_observed_samples"] = len(graph_samples) if graph_samples else None
                record["runtime_info"] = report.get("target", {}).get("runtime_info", {})
                record["maximum_active_workers"] = report.get("maximum_active_workers")
                record["warnings"] = report.get("warnings", [])
                record["stage_statistics"] = report.get("stage_statistics", {}) if stages else None
        record["status"] = "failed" if errors else "passed"
        record["reason"] = "\n".join(errors)
        if errors:
            record["failure_log_excerpt"] = failure_excerpt(folder)
        save_json(folder / "result.json", record)
        manifest["runs"].append(record)
        completed_by_id[run_id] = record
        manifest["current_run"] = None
        persist()
        print(f"{record['status'].upper()} {run_id}" + (f": {errors[0]}" if errors else ""), flush=True)
        return record

    sampler = None
    telemetry_stdout = None
    telemetry_stderr = None
    telemetry_record: dict[str, Any] = {}
    try:
        segment = len(manifest.get("telemetry_segments", [])) + 1
        telemetry_path = output / f"gpu_telemetry_{segment:03d}.csv"
        telemetry_stdout = telemetry_path.open("wb")
        telemetry_stderr = telemetry_path.with_suffix(".stderr.log").open("wb")
        telemetry_record = {"csv": telemetry_path.name, "started_utc": utc_now(),
                            "interval_seconds": 1}
        manifest.setdefault("telemetry_segments", []).append(telemetry_record)
        try:
            sampler = subprocess.Popen(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv", "-l", "1"],
                                       cwd=root, stdout=telemetry_stdout, stderr=telemetry_stderr)
            telemetry_record["pid"] = sampler.pid
        except OSError as error:
            telemetry_record["error"] = str(error)
        cells = []
        for artifact in artifacts:
            backends = ("ort-cpu", "ort-cuda", "ort-trt") if artifact["artifact_type"] == "onnx" else ("native-trt", "fused-trt")
            for backend in backends:
                if not args.only_backends or backend in args.only_backends:
                    cells.append((artifact, backend))
        priority = {"fused-trt": 0, "native-trt": 1, "ort-cuda": 2, "ort-trt": 3, "ort-cpu": 4}
        precision_priority = {"fp16": 0, "fp32": 1, "int8": 2}
        cells.sort(key=lambda item: (priority[item[1]], precision_priority[item[0]["precision"]], item[0]["id"]))
        ready = []
        for artifact, backend in cells:
            cell_id = f"{artifact['id']}__{backend}"
            cell = next((item for item in manifest["cells"] if item["cell_id"] == cell_id), None)
            if cell is None:
                cell = {"cell_id": cell_id, "artifact": artifact["id"], "backend": backend,
                        "status": "smoke_pending"}
                manifest["cells"].append(cell)
            smoke = one(artifact, backend, "smoke")
            if smoke["status"] != "passed":
                cell.update({"status": "smoke_failed", "reason": smoke["reason"],
                             "failure_log_excerpt": smoke.get("failure_log_excerpt", ""),
                             "benchmark_disposition": "Not repeated because initial inference failed; no performance claim."})
                persist()
                continue
            cell["status"] = "benchmarking"
            tests = measurement_plan(backend, artifact['precision'], args.graph_ablation)
            if backend == "fused-trt":
                unsupported = {"cell_id": cell_id, "test": "latency_backend_clean",
                               "reason": "CLI explicitly exposes fused-trt timing only as end-to-end."}
                if unsupported not in manifest["unsupported_tests"]:
                    manifest["unsupported_tests"].append(unsupported)
            ready.append((artifact, backend, cell, tests, []))
        # Reverse cell order between repetitions to reduce fixed ordering bias.
        for repetition in range(1, args.repetitions + 1):
            for artifact, backend, cell, tests, outcomes in ready[::1 if repetition % 2 else -1]:
                for test, depth in tests:
                    outcomes.append(one(artifact, backend, test, depth, repetition))
        for artifact, backend, cell, tests, outcomes in ready:
            outcomes.append(one(artifact, backend, "stage_diagnostic"))
            cell["status"] = "passed" if all(item["status"] == "passed" for item in outcomes) else "benchmark_failed"
            persist()
        manifest["code_after"] = code_snapshot(root)
        manifest["executable_after"] = file_record(executable)
        manifest["compiled_sources_unchanged"] = (
            before["compiled_source_sha256"] == manifest["code_after"]["compiled_source_sha256"])
        manifest["executable_unchanged"] = binary["sha256"] == manifest["executable_after"]["sha256"]
        manifest["model_files_after"] = refresh_file_records(artifacts)
        manifest["model_files_unchanged"] = identities(artifacts) == identities(manifest["model_files_after"])
        try:
            manifest["source_frames_after"] = source_images(source, args.source_frames)
            manifest["source_frames_unchanged"] = identities(images) == identities(manifest["source_frames_after"])
        except (OSError, ValueError) as error:
            manifest["source_frames_after"] = {"error": str(error)}
            manifest["source_frames_unchanged"] = False
        manifest["runtime_libraries_after"] = [file_record(path) for path in sorted(executable.parent.glob("*.dll"))]
        manifest["runtime_libraries_unchanged"] = (
            identities(manifest["build"]["adjacent_libraries"]) == identities(manifest["runtime_libraries_after"]))
        manifest["hardware_after"] = hardware_snapshot(root)
        manifest["finished_utc"] = utc_now()
        unchanged = all(manifest[key] for key in (
            "compiled_sources_unchanged", "executable_unchanged", "model_files_unchanged",
            "source_frames_unchanged", "runtime_libraries_unchanged"))
        all_passed = all(cell["status"] == "passed" for cell in manifest["cells"])
        manifest["status"] = "completed" if unchanged else "invalidated_by_input_changes"
        manifest["all_cells_passed"] = all_passed
        persist()
        print(f"FINISHED {manifest['status']}: {len(manifest['runs'])} runs, "
              f"{manifest['failed_run_count']} failed; {output}", flush=True)
        return 0 if unchanged and all_passed else 2
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "runner_error"
        manifest["runner_error"] = traceback.format_exc()
        manifest["finished_utc"] = utc_now()
        persist()
        raise
    finally:
        if sampler is not None:
            sampler_running = sampler.poll() is None
            terminate_child(sampler)
            telemetry_record["running_at_shutdown"] = sampler_running
            telemetry_record["exit_code"] = sampler.returncode
        telemetry_record["finished_utc"] = utc_now()
        if telemetry_stdout is not None:
            telemetry_stdout.close()
        if telemetry_stderr is not None:
            telemetry_stderr.close()
        persist()


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise ValueError(f"Output already exists; choose a fresh directory: {output}")
    if args.resume and not (output / "manifest.json").is_file():
        raise ValueError("--resume requires an existing manifest.json")
    output.mkdir(parents=True, exist_ok=args.resume)
    with OutputLock(output):
        return run_locked(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing results are never overwritten")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--latency-iterations", type=int, default=300)
    parser.add_argument("--throughput-iterations", type=int, default=1000)
    parser.add_argument("--stage-iterations", type=int, default=100)
    parser.add_argument("--source-frames", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Continue finalized runs only after verifying all input identities")
    parser.add_argument("--only-backends", nargs="+", choices=["ort-cpu", "ort-cuda", "ort-trt", "native-trt", "fused-trt"],
                        help="Optional restricted matrix; omitted by default to test every backend")
    parser.add_argument("--graph-ablation", action=argparse.BooleanOptionalAction, default=True,
                        help="Also repeat fused FP16 latency and depth-3 throughput with CUDA graphs disabled")
    args = parser.parse_args()
    for field in ("repetitions", "latency_iterations", "throughput_iterations", "stage_iterations", "source_frames", "timeout_seconds"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.warmup < 0 or args.device < 0:
        parser.error("--warmup and --device must be non-negative")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print("Benchmark runner interrupted; completed evidence is preserved.", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
