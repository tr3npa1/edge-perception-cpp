"""Export a completed all-model run as local, reviewable documentation.

No inference, build, upload, or Git operations. Existing bundles require
--replace; source evidence is read-only. Only named generated files are written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_text(path, text):
    """Keep published hashes stable across Windows and Linux Git checkouts."""
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(scrub(rows))
    return stream.getvalue()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()
                if k.lower() not in {"hostname", "node", "uuid", "pid"}}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        value = value.replace("\\", "/")
        value = re.sub(re.escape(ROOT.as_posix()) + "/?", "", value, flags=re.I)
        value = re.sub(r"(?i)\b[A-Z]:/[^\n\"'<>;,)]*", "<external-path>", value)
        value = re.sub(r"/(?:home|Users)/[^\s\"']+", "<external-path>", value)
        return re.sub(r"GPU-[a-fA-F0-9-]+", "<gpu-id>", value)
    return value


def validate(root):
    manifest = read(root / "pipeline/manifest.json")
    summary = read(root / "pipeline/summary.json")
    performance = read(root / "performance/manifest.json")
    require(manifest["status"] == "completed" and manifest.get("evidence_validated")
            and not manifest.get("diagnostic_only"), "Need a validated, completed full run")
    require(summary["status"] == "completed" and summary["evidence_validated"], "Summary is incomplete")
    require(not summary["failed_stages"], "Execution failures require review before export")
    require(performance["status"] == "completed", "Performance matrix is incomplete")
    for name in ("compiled_sources_unchanged", "executable_unchanged", "model_files_unchanged",
                 "source_frames_unchanged", "runtime_libraries_unchanged"):
        require(performance.get(name) is True, f"Invalid performance evidence: {name}")
    records = {}
    for stage in manifest["stages"].values():
        require(stage["status"] == "completed", "Unfinished coordinator stage")
        records.update({r["path"]: r for r in stage["outputs"]})
    for record in records.values():
        require(digest(record["path"]) == record["sha256"], f"Changed evidence: {record['path']}")
    for run in performance["runs"]:
        require(read(root / "performance" / run["folder"] / "result.json") == run,
                f"Finalized performance record changed: {run['run_id']}")
    for row in summary["performance"]:
        require(row["successful_runs"] == row["required_runs"] == 3, "Incomplete repetitions")
    require(len(summary["quality"]) == 24 and len(summary["parity"]) == 42,
            "Unexpected all-model coverage; review the export schema")
    require(all(q["images"] == 20000 for q in summary["quality"].values()), "Incomplete quality split")
    require(summary["quality"] == {n[8:]: s["result"] for n, s in manifest["stages"].items()
                                   if n.startswith("quality_") and not n.startswith("quality_predict_")},
            "Quality summary differs from finalized stages")
    require(summary["parity"] == {n: s["result"] for n, s in manifest["stages"].items()
                                  if n.startswith("compare_")}, "Parity summary differs from finalized stages")
    failed = [r for r in performance["runs"] if r["status"] != "passed"]
    require(len(performance["runs"]) == 202 and len(failed) == 2 and all(
        r["precision"] == "int8" and r["backend"] == "ort-trt" and r["test"] == "smoke"
        and "does not route" in r.get("failure_log_excerpt", "") for r in failed),
        "Unexpected execution coverage/failures; review before publishing")
    for row in summary["performance"]:
        matches = [r for r in performance["runs"] if all(r[k] == row[k] for k in ("artifact", "backend", "test", "depth"))]
        require(len(matches) == 3, "Summary group does not have three saved runs")
        for metric, stats in row["statistics"].items():
            values = [r["metrics"][metric] for r in matches]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            expected = {"median": statistics.median(values), "minimum": min(values), "maximum": max(values),
                        "mean": mean, "sample_stddev": sd, "cv_percent": 100 * sd / mean}
            require(all(math.isclose(stats[k], v, rel_tol=1e-9, abs_tol=1e-9) for k, v in expected.items()),
                    f"Summary statistics differ from saved runs: {row['artifact']} / {metric}")
    for metric, stats in summary["pytorch_timing"].items():
        values = [manifest["stages"][f"pytorch_timing_r{i}"]["result"][metric] for i in range(1, 4)]
        require(math.isclose(stats["median"], statistics.median(values), rel_tol=1e-9), "PyTorch summary mismatch")
    return manifest, summary, performance, len(records)


def table(headers, rows):
    def line(row):
        return "| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *map(line, rows)])


def label(artifact, backend):
    return artifact.replace("engine_yolo26m_bdd100k_", "TRT ").replace(
        "onnx_yolo26m_bdd100k_", "ONNX ").upper() + " / " + backend


def export(root, output, replace=False):
    m, s, p, verified = validate(root)
    require(not output.exists() or replace, "Output exists; use --replace after reviewing its contents")
    output.mkdir(parents=True, exist_ok=True)
    generated = []

    def write(name, text):
        write_text(output / name, text)
        generated.append(name)

    def json_file(name, value):
        write(name, json.dumps(scrub(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def csv_file(name, rows):
        write(name, csv_text(rows))

    quality = [{"path": name, "images": q["images"], "targets": q["targets"],
                "map50": q["metrics/mAP50(B)"] * 100, "map50_95": q["metrics/mAP50-95(B)"] * 100}
               for name, q in s["quality"].items()]
    parity = [{"comparison": name, "matched_images": q["matched_images"], "images": q["images"],
               "passed": q["passed"], "same_artifact": q["same_artifact"]}
              for name, q in s["parity"].items()]
    runs = [{k: r.get(k) for k in ("run_id", "cell_id", "artifact", "backend", "precision", "test",
                                  "depth", "repetition", "status", "reason", "metrics",
                                  "maximum_active_workers", "warnings", "stage_statistics")}
            for r in p["runs"]]
    aggregate = read(root / "performance/aggregate/summary.json")
    gpu = list(csv.DictReader(io.StringIO(m["hardware"]["gpu"]["stdout"]), skipinitialspace=True))[0]
    cpu = json.loads("[" + m["hardware"]["cpu_memory"]["stdout"].strip().replace("}\n{", "},{").replace("}\r\n{", "},{") + "]")
    hardware = {"gpu": gpu["name"], "gpu_memory": gpu["memory.total [MiB]"],
                "driver": gpu["driver_version"], "cpu": cpu[0]["Name"].strip(),
                "ram_bytes": cpu[1]["TotalPhysicalMemory"], "platform": m["hardware"]["platform"]}
    preparations = []
    for folder in sorted(root.glob("prep_*")):
        record = read(folder / "preparation.json")
        preparations.append({"path": str(folder.relative_to(root)), "state": record["state"],
                             "steps": [{"name": x["name"], "exit_code": x["exit_code"]} for x in record["steps"]]})
    validation = {"status": m["status"], "evidence_validated": m["evidence_validated"],
                  "verified_evidence_files": verified, "coordinator_jobs": len(m["stages"]),
                  "quality_paths": len(quality), "parity_comparisons": len(parity),
                  "parity_passed": sum(x["passed"] for x in parity),
                  "performance_executions": len(runs),
                  "performance_passed": sum(x["status"] == "passed" for x in runs),
                  "preparation": preparations,
                  "gpu_int8_regressions_passed": m["stages"]["gpu_int8_regressions"]["result"]["passed"],
                  "docker": "Not validated by this native Windows benchmark",
                  "remote_ci": "Not established by local benchmark evidence"}
    provenance = {"schema_version": 1, "run_id": root.name, "finished_utc": m["finished_utc"],
                  "local_evidence": str(root.relative_to(ROOT)), "hardware": hardware,
                  "packages": m["identity"]["packages"], "measured_identity": m["identity"],
                  "performance_settings": p["settings"],
                  "source_files": [{"path": str((root / x).relative_to(ROOT)), "sha256": digest(root / x)}
                                   for x in ("pipeline/manifest.json", "pipeline/summary.json", "performance/manifest.json")],
                  "resumed": bool(p.get("resumptions")),
                  "scope": "Native Windows; local source snapshot identified by hashes, not a clean-commit claim"}
    json_file("accuracy.json", {"scale": "0-100 in CSV; original metric fields are 0-1", "paths": s["quality"]})
    csv_file("accuracy.csv", quality)
    json_file("parity.json", s["parity"])
    csv_file("parity.csv", parity)
    json_file("performance.json", {"hardware": hardware, "settings": p["settings"], "cells": p["cells"],
                                   "aggregate": s["performance"], "runs": runs, "pytorch_timing": s["pytorch_timing"]})
    csv_file("summary.csv", aggregate)
    csv_file("runs.csv", [{k: v for k, v in r.items() if k not in ("metrics", "warnings", "stage_statistics")}
                          | (r["metrics"] or {}) for r in runs])
    json_file("provenance.json", provenance)
    json_file("validation.json", validation)

    # Fixed first smoke image, not selected for appearance or detection count.
    examples = []
    for index, (precision, backend) in enumerate((("fp16", "fused-trt"), ("fp16", "native-trt"), ("int8", "fused-trt")), 1):
        run = next(r for r in p["runs"] if r["precision"] == precision and r["backend"] == backend and r["test"] == "smoke")
        source = sorted((root / "performance" / run["folder"] / "app_output").glob("*_detections.jpg"))[0]
        name = f"example-{index}.jpg"
        shutil.copy2(source, output / name)
        generated.append(name)
        examples.append({"file": name, "backend": backend, "precision": precision,
                         "source": str(source.relative_to(ROOT)), "sha256": digest(source),
                         "selection": "First saved image of the fixed-input smoke run; illustrative, not an accuracy test"})
    json_file("examples.json", examples)

    # Static scientific plots; values and ranges are taken directly from aggregates.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, test, depth, metric, title, xlabel in (
        ("latency.svg", "latency_e2e_clean", 1, "mean_ms", "Sequential inference latency", "Milliseconds (lower is better)"),
        ("throughput.svg", "throughput_e2e_clean", 3, "fps", "Streaming throughput at depth 3", "Frames per second (higher is better)")):
        rows = [r for r in s["performance"] if r["backend"] in ("native-trt", "fused-trt") and r["test"] == test and r["depth"] == depth]
        rows.sort(key=lambda r: ({"fp32": 0, "fp16": 1, "int8": 2}[r["artifact"].rsplit("_", 1)[1]], r["backend"] == "fused-trt"))
        values = [r["statistics"][metric]["median"] for r in rows]
        bounds = [[v-r["statistics"][metric]["minimum"] for v, r in zip(values, rows)],
                  [r["statistics"][metric]["maximum"]-v for v, r in zip(values, rows)]]
        with plt.rc_context({"svg.fonttype": "none", "font.size": 10, "svg.hashsalt": "edge-perception"}):
            fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
            ax.barh(range(len(rows)), values, xerr=bounds, capsize=3,
                    color=["#2563eb" if r["backend"] == "native-trt" else "#0f766e" for r in rows])
            ax.set_yticks(range(len(rows)), [r["artifact"].rsplit("_", 1)[1].upper()+" / "+r["backend"] for r in rows])
            ax.invert_yaxis()
            ax.set_xlim(0, max(r["statistics"][metric]["maximum"] for r in rows)*1.2)
            for i, value in enumerate(values):
                ax.text(rows[i]["statistics"][metric]["maximum"]+ax.get_xlim()[1]*.015, i, f"{value:.2f}", va="center")
            ax.set_title(title + "\nRTX 2060 · batch 1 · 640×640", loc="left", pad=14)
            ax.set_xlabel(xlabel + "; median of 3 runs, whiskers show min–max")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="x", alpha=.18)
            fig.savefig(output/name, metadata={"Date": None})
            plt.close(fig)
        # Matplotlib may emit platform line endings and trailing path whitespace.
        svg = (output / name).read_text(encoding="utf-8")
        write_text(output / name, "\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
        generated.append(name)

    quality_table = table(["Path", "Images", "mAP50", "mAP50–95"],
                          [[x["path"], x["images"], f"{x['map50']:.4f}", f"{x['map50_95']:.4f}"] for x in quality])
    parity_table = table(["Comparison", "Images matched", "Gate"],
                         [[x["comparison"], f"{x['matched_images']}/{x['images']}", "PASS" if x["passed"] else "FAIL"] for x in parity])
    perf_rows = []
    for r in s["performance"]:
        t = r["statistics"]; f = t["fps"]
        perf_rows.append([label(r["artifact"], r["backend"]), r["test"], r["depth"],
                          " / ".join(f"{t[k]['median']:.3f}" for k in ("mean_ms", "median_ms", "p95_ms")),
                          f"{f['median']:.3f}", f"{f['minimum']:.3f}–{f['maximum']:.3f}", f"{f['cv_percent']:.2f}%", "3/3"])
    perf_table = table(["Path", "Measurement", "Depth", "Mean / p50 / p95 (ms)", "FPS", "FPS range", "FPS CV", "Runs"], perf_rows)
    report = f"""# Deployment evaluation

[Project](../../README.md) · [Interpretation](../RESULTS_ANALYSIS.md) · [Reproduction](../BENCHMARKING.md)

Measured run: `{root.name}`. All 120 coordinator jobs completed; source/input
identities were validated at completion. This bundle was exported from verified
saved evidence, not a new inference run. Complete does not mean every parity gate passed.

## Measurement protocol

- {hardware['gpu']}, {hardware['cpu']}, native Windows, batch 1, square 640×640.
- 24 quality paths, each using the full prepared 20,000-image test split;
  367,728 targets. This is not an official BDD100K leaderboard evaluation.
- Quality confidence 0.001, maximum 300 detections, common matching/AP code.
- Parity: 64 validation images, seed 42, confidence 0.25; unchanged strict gates.
- Performance: 128 fixed decoded validation images; 50 warmups; three repetitions
  of 300 sequential or 1,000 streaming frames. Stage diagnostics use 100 frames.
- Sequential timing covers decoded BGR pixels through detection objects. Decode,
  initialization, annotation and disk output are excluded. Streaming frame
  residence includes scheduling/backpressure and is not sequential latency.
- Statistics are medians of per-run statistics, not pooled percentiles. FPS is
  completed frames / measured wall time. Run ranges are not confidence intervals.
- Runs resumed after power interruptions; source/model identities remained unchanged.
  GPU clocks were not locked; no statistical significance claim is made.

## Detection quality

Scores use a 0–100 scale. Python rows are artifact-level evaluations; C++ rows
measure the complete corresponding preprocessing/inference/postprocessing path.

{quality_table}

## Sequential latency and streaming throughput

`latency_*` is sequential. The millisecond columns for `throughput_*` describe
frame residence under streaming, not single-frame inference latency. Backend-only
timings and graph-off ablations are separate measurements, not substitutions for
end-to-end results. Every row below has all three repetitions.

{perf_table}

![Sequential TensorRT latency with run ranges](latency.svg)

![Depth-3 TensorRT throughput with run ranges](throughput.svg)

PyTorch API sequential mean/p50/p95: **{' / '.join(f'{s["pytorch_timing"][k]["median"]:.3f}' for k in ('mean_ms', 'p50_ms', 'p95_ms'))} ms**;
{s['pytorch_timing']['fps']['median']:.3f} FPS. This separately synchronized Python path includes API overhead.

## Strict parity

22 comparisons pass; 20 fail. Same-artifact parity, PyTorch agreement and dataset
mAP are distinct properties. See [gate definitions](fp16-correctness.md).

{parity_table}

## Execution coverage and limitations

All 16 supported C++ paths completed inference and performance measurement.
Of 202 performance executions, 200 passed. Two INT8 ONNX / ORT TensorRT smoke
checks were rejected by project routing policy; no timing is assigned to them.
This does not establish a universal TensorRT limitation. Native TensorRT INT8
uses the separately built engine and completes normally.

Native depth-3 runs recorded three active workers for every precision and
repetition. This verifies concurrent host workers, not guaranteed GPU kernel overlap.
Docker execution, remote CI status and external camera/device-input throughput
are not established by this native decoded-image benchmark. No QAT was run.

## Annotated outputs

The first saved image from each named smoke run is shown below. Selection is
fixed, not based on visual quality. These examples demonstrate output generation;
accuracy is established by the full test split above. Source imagery remains
subject to BDD100K's dataset terms.

| Fused FP16 | Native FP16 | Fused INT8 |
|---|---|---|
| ![Fused FP16 detections](example-1.jpg) | ![Native FP16 detections](example-2.jpg) | ![Fused INT8 detections](example-3.jpg) |

## Evidence files

- [accuracy.csv](accuracy.csv) / [accuracy.json](accuracy.json): 24 quality paths, with per-class metrics in JSON.
- [parity.csv](parity.csv) / [parity.json](parity.json): every strict result and its thresholds.
- [summary.csv](summary.csv) / [runs.csv](runs.csv): aggregate and per-execution timing records.
- [performance.json](performance.json): variability, stage diagnostics, failures and PyTorch timing.
- [provenance.json](provenance.json): artifact/source hashes, versions, settings and source-manifest hashes.
- [validation.json](validation.json): saved validation outcomes and scope.
- [examples.json](examples.json): image selection and checksums.
- [SHA256SUMS.txt](SHA256SUMS.txt): generated-bundle checksums (identity, not independent reproduction).

Raw predictions, per-frame samples, logs, engines and caches remain in the ignored
local run directory. Paths in provenance identify that local evidence; they are
not downloadable GitHub assets. No model weights or dataset download is bundled.
"""
    write("README.md", report)
    write("SHA256SUMS.txt", "".join(f"{digest(output/name)}  {name}\n" for name in sorted(generated)))
    print(f"Exported {len(generated)} files; verified {verified} source evidence files. No inference or upload.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Completed all-model run directory")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results")
    parser.add_argument("--replace", action="store_true", help="Replace only this exporter's named generated files")
    args = parser.parse_args()
    export(args.source.resolve(), args.output.resolve(), args.replace)
