"""Curate a completed release measurement suite into reviewable repository docs.

This command never runs inference or uploads anything. It refuses incomplete
matrices and changed measurement inputs. Recorded runtime/parity failures are
published as failures, not treated as missing data or converted into passes.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import statistics
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("precision", "recall", "map50", "map75", "map50_95")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def scrub(value):
    """Keep public evidence useful without publishing workstation identifiers."""
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()
                if key.lower() not in {"hostname", "node", "uuid", "pid"}}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        value = value.replace("\\", "/")
        value = re.sub(re.escape(ROOT.as_posix()), ".", value, flags=re.I)
        value = re.sub(r"(?i)\b[A-Z]:/[^\n\"'<>;,)]*", "<external-path>", value)
        value = re.sub(r"/(?:home|Users)/[^\s\"']+", "<external-path>", value)
        return re.sub(r"GPU-[a-fA-F0-9-]+", "<gpu-id>", value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(scrub(value), indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in scrub(rows):
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_record(record: dict) -> None:
    path = Path(record["path"])
    require(path.is_file(), f"Recorded input is missing: {path}")
    require(digest(path).lower() == record["sha256"].lower(), f"Recorded input changed: {path}")


def validate(performance: Path, accuracy: Path):
    pm, am = read(performance / "manifest.json"), read(accuracy / "manifest.json")
    require(pm.get("status") == "completed", "Performance matrix has not completed.")
    require(pm.get("compiled_sources_unchanged") is True and pm.get("executable_unchanged") is True,
            "Performance source/executable identity checks did not pass.")
    require(not pm["settings"].get("only_backends"), "A restricted backend matrix cannot be published as the full suite.")
    require(len(pm.get("cells", [])) > 0 and len(pm.get("runs", [])) > 0, "Performance matrix is empty.")
    require(all(cell["status"] in {"passed", "smoke_failed", "benchmark_failed"} for cell in pm["cells"]),
            "Performance matrix contains an unfinished cell.")
    for record in [pm["executable"], *pm["artifacts"], *pm["source_frames"]]:
        verify_record(record)
    for item in pm["code_before"]["compiled_source_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], f"Compiled source changed: {item['path']}")
    for run in pm["runs"]:
        require(read(performance / run["folder"] / "result.json") == run,
                f"Finalized run record changed: {run['run_id']}")
    require(am.get("executable_sha256") == pm["executable"]["sha256"],
            "Accuracy and performance used different executables.")
    require(am.get("split_image_count") == 20000 and am["settings"].get("split") == "test",
            "Expected the complete 20,000-image test split.")
    for relative, expected in am["source_sha256"].items():
        require(digest(ROOT / relative) == expected, f"Evaluation source changed: {relative}")
    for artifact in am["artifacts"]:
        if artifact["exists"]:
            verify_record(artifact)
    require(len(am["artifacts"]) >= 8, "Expected at least eight accuracy artifact attempts.")
    ar, cr = read(accuracy / "accuracy_results.json"), read(accuracy / "cpp_parity_results.json")
    require(len({item["label"] for item in ar}) == len(ar), "Duplicate accuracy result labels.")
    require({item["label"] for item in ar} == {item["label"] for item in am["artifacts"]},
            "Some declared accuracy artifacts have not been attempted.")
    for item in ar:
        require(item.get("status") == "missing_artifact" or item.get("process", {}).get("completed") is True,
                f"Unfinished accuracy attempt: {item['label']}")
    expected_cpp = {artifact["label"] + "__" + backend for artifact in am["artifacts"]
                    if artifact["type"] != "pytorch"
                    for backend in (("ort-cpu", "ort-cuda", "ort-trt") if artifact["type"] == "onnx"
                                    else ("native-trt", "fused-trt"))}
    require(len(cr) == len(expected_cpp) and {item["label"] for item in cr} == expected_cpp,
            "C++ parity coverage is incomplete.")
    for item in cr:
        require(item.get("status") in {"missing_artifact", "unsupported"}
                or item.get("process", {}).get("completed") is True,
                f"Unfinished C++ attempt: {item['label']}")
    return pm, am, ar, cr


def hardware(pm: dict) -> dict:
    raw = pm["hardware"]
    gpu_rows = list(csv.DictReader(io.StringIO(raw.get("gpu", {}).get("stdout", ""))))
    gpu = [{key.strip(): value.strip() for key, value in row.items()
            if key.strip() not in {"uuid", "timestamp"} and value is not None} for row in gpu_rows]
    cpu = []
    remaining = raw.get("cpu_memory", {}).get("stdout", "").strip()
    while remaining:
        try:
            value, end = json.JSONDecoder().raw_decode(remaining)
            cpu.append(value)
            remaining = remaining[end:].strip()
        except json.JSONDecodeError:
            break
    return {"platform": raw.get("platform"), "logical_cpu_count": raw.get("logical_cpu_count"),
            "cpu_and_memory": cpu, "gpu": gpu, "python": raw.get("python"),
            "application_version": pm.get("executable_version", {}).get("stdout", "").strip()}


def accuracy_rows(reports: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, providers = [], []
    for report in reports:
        evaluation = report.get("evaluation") or {}
        exported = (evaluation.get("artifacts") or [{}])[0]
        validation = evaluation.get("pytorch_evaluation") if report["label"] == "pytorch" else exported.get("validation")
        validation = validation or {}
        metrics = validation.get("metrics") or {}
        parity = exported.get("parity") or {}
        rows.append({"artifact": report["label"], "status": "measured" if validation.get("success") else "failed",
                     "process_exit_code": report.get("process", {}).get("return_code"),
                     **{key: metrics.get(key) for key in METRICS},
                     "strict_export_parity": parity.get("passed"),
                     "parity_images_passed": parity.get("passed_image_count"),
                     "parity_images_total": parity.get("image_count"),
                     "error": validation.get("error") or report.get("process", {}).get("error")
                     or report.get("status") or ("No evaluation summary produced" if not evaluation else None)})
        for check in exported.get("ort_backend_checks", []):
            skipped = str(check.get("error") or "").startswith("skipped_unsupported_combo:")
            providers.append({"artifact": report["label"], "backend": check["backend"],
                              "status": "unsupported" if skipped else "passed" if check.get("success") else "failed",
                              "success": check.get("success"), "actual_providers": check.get("actual_providers"),
                              "error": check.get("error")})
    return rows, providers


def parity_rows(reports: list[dict]) -> list[dict]:
    rows = []
    for report in reports:
        comparisons = report.get("comparisons") or {"unavailable": {}}
        for reference, comparison in comparisons.items():
            rows.append({"configuration": report["label"], "reference": reference,
                         "status": report.get("status") or ("passed" if comparison.get("passed") else "failed"),
                         "images": comparison.get("image_count"),
                         "passed_images": comparison.get("passed_image_count"),
                         "mean_matched_iou": comparison.get("mean_matched_iou"),
                         "max_box_abs_diff": comparison.get("max_box_abs_diff"),
                         "max_conf_abs_diff": comparison.get("max_conf_abs_diff"),
                         "reason": report.get("reason") or report.get("error") or comparison.get("error")
                         or comparison.get("reason")})
    return rows


def validation_evidence(folder: Path) -> dict:
    suites, builds = [], []
    for path in sorted(folder.glob("*.xml")):
        tree = ET.parse(path).getroot()
        for suite in ([tree] if tree.tag == "testsuite" else tree.findall("testsuite")):
            cases = []
            for case in suite.findall("testcase"):
                failure = case.find("failure")
                error = case.find("error")
                cases.append({"name": case.get("name"), "class": case.get("classname"),
                              "seconds": case.get("time"),
                              "status": "failed" if failure is not None or error is not None else
                              "skipped" if case.find("skipped") is not None else "passed",
                              "message": (failure.get("message") if failure is not None else
                                          error.get("message") if error is not None else None)})
            suites.append({"file": path.name, "tests": suite.get("tests"), "failures": suite.get("failures"),
                           "errors": suite.get("errors", "0"), "skipped": suite.get("skipped", "0"),
                           "seconds": suite.get("time"), "cases": cases, "source_sha256": digest(path)})
    for path in sorted(folder.glob("*-build.json")):
        builds.append({"file": path.name, "steps": read(path), "source_sha256": digest(path)})
    required = {"cpu-tests.xml", "nvidia-tests.xml", "python-unit.xml", "python-integration.xml"}
    require(required.issubset({item["file"] for item in suites}), "Required fresh test evidence is missing.")
    return {"test_suites": suites, "builds": builds,
            "docker": read(folder / "docker-status.json") if (folder / "docker-status.json").is_file() else None,
            "dataset": read(folder / "dataset_summary.json") if (folder / "dataset_summary.json").is_file() else None}


def examples(accuracy: Path, cpp_reports: list[dict], output: Path) -> list[dict]:
    """Choose deterministic examples with different detected class coverage."""
    eligible = [item for item in cpp_reports if item.get("process", {}).get("return_code") == 0]
    eligible.sort(key=lambda item: (item["label"] != "engine_fp16__fused-trt", item["label"]))
    for report in eligible:
        folder = accuracy / "cpp_outputs" / report["label"]
        detection_path = folder / "detections.jsonl"
        if not detection_path.is_file():
            continue
        rows = [json.loads(line) for line in detection_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        candidates = []
        for row in rows:
            name = Path(row["source"]).stem
            images = sorted(folder.glob(name + "_detections.*"))
            if images and row.get("detections"):
                candidates.append((row, images[0]))
        if len(candidates) < 3:
            continue
        selected, classes = [], set()
        for index in range(3):
            candidates.sort(key=lambda pair: (-len({d["class_id"] for d in pair[0]["detections"]} - classes),
                                               -len({d["class_id"] for d in pair[0]["detections"]}),
                                               -len(pair[0]["detections"]), pair[1].name))
            row, image = candidates.pop(0)
            ids = {d["class_id"] for d in row["detections"]}
            classes.update(ids)
            destination = output / (f"example-{index + 1}" + image.suffix.lower())
            shutil.copyfile(image, destination)
            selected.append({"file": destination.name, "configuration": report["label"],
                             "source_name": Path(row["source"]).name, "output_sha256": digest(image),
                             "detections": row["detections"], "class_ids": sorted(ids)})
        return selected
    return []


def f(value, percent=False) -> str:
    return "—" if value is None else f"{float(value) * (100 if percent else 1):.2f}"


def label(value: str) -> str:
    return value.replace("onnx_yolo26m_bdd100k_", "ONNX ").replace("engine_yolo26m_bdd100k_", "TensorRT ").replace("__", " / ")


def table(headers: list[str], rows: list[list]) -> str:
    def cell(value):
        return str(scrub(value)).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def plots(output: Path, summary: list[dict]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figures = []
    for test, metric, axis_label, filename in (
        ("latency_e2e_clean", "mean_ms", "Median of run mean latency (ms; lower is better)", "latency.svg"),
        ("throughput_e2e_clean", "fps", "Median of run throughput (FPS; higher is better)", "throughput.svg"),
    ):
        rows = [item for item in summary if item["test"] == test and item["complete"]]
        rows.sort(key=lambda item: item[f"median_of_run_{metric}"])
        if not rows:
            continue
        plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
        fig, ax = plt.subplots(figsize=(11, max(4, len(rows) * 0.35)))
        values = [item[f"median_of_run_{metric}"] for item in rows]
        errors = [[value - item[f"minimum_run_{metric}"] for value, item in zip(values, rows)],
                  [item[f"maximum_run_{metric}"] - value for value, item in zip(values, rows)]]
        names = [label(item["cell_id"]) + (f" (depth {item['depth']})" if test.startswith("throughput") else "")
                 for item in rows]
        ax.barh(names, values, xerr=errors, color="#147d92", capsize=2)
        ax.set_xlabel(axis_label)
        ax.set_title("YOLO26M · 640 × 640 · batch 1\nWhiskers: minimum–maximum across complete repetitions")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(output / filename, metadata={"Date": None})
        plt.close(fig)
        figures.append(filename)
    return figures


def report(pm, am, summary, ar, pr, providers, tests, demos, figures) -> str:
    settings = pm["settings"]
    machine = hardware(pm)
    gpu = (machine["gpu"] or [{}])[0]
    cpu = next((item for item in machine["cpu_and_memory"] if "Name" in item), {})
    memory = next((item.get("TotalPhysicalMemory") for item in machine["cpu_and_memory"]
                   if "TotalPhysicalMemory" in item), None)
    passed = sum(run["status"] == "passed" for run in pm["runs"])
    lines = ["# Deployment results", "",
             "Measured deployment performance, detection quality, compatibility, and validation evidence for YOLO26M.",
             "The source, executable, model, and input identities are recorded in [provenance.json](provenance.json).",
             "All attempted configurations are retained, including failed runtime initialization and strict parity mismatches.", "",
             "## Coverage and interpretation", "",
             f"- {len(pm['runs'])} finalized performance attempts: {passed} passed and {len(pm['runs']) - passed} failed initial inference checks, across {len(pm['cells'])} artifact/backend combinations.",
             f"- {len(ar)} artifact quality evaluations on the full prepared test split; strict C++ parity covers {len(pm['cells'])} combinations, including unsupported and failed attempts.",
             "- The suite completed its scheduled attempts; this is not an all-tests-passed or production-certification claim.",
             "- Fastest measured configuration: fused TensorRT FP16. Its strict same-artifact parity failed; no full-test-set C++ fused mAP was measured.",
             "- Docker container execution was not validated in this suite. See the build evidence and [interpretation notes](../RESULTS_ANALYSIS.md).", "",
             "## Measurement protocol", "",
             f"- Input: batch 1, 640 × 640 model tensor; {len(pm['source_frames'])} preloaded source images reused by the benchmark.",
             f"- Per configuration: {settings['repetitions']} repetitions, {settings['warmup']} warmups, "
             f"{settings['latency_iterations']} latency iterations and {settings['throughput_iterations']} throughput iterations per repetition.",
             "- Latency covers an in-memory BGR image through preprocessing, inference, and detection objects. Image decoding, model startup, annotation, and file writing are excluded.",
             "- Streaming FPS is completed frames divided by measured wall time. Concurrent-frame latency is reported separately; reciprocal latency is not streaming throughput.",
             "- Headline timing disables stage instrumentation. Stage diagnostics are separate 100-frame runs.",
             "- Tables show medians of per-run means/percentiles and the range across repetitions. They are not pooled percentiles or confidence intervals.",
             "- Explicit ORT TensorRT sessions can partition work across TensorRT, CUDA, and CPU. Provider availability does not prove placement of every graph node.", "",
             "## Hardware and software", "",
             table(["Component", "Recorded configuration"], [
                 ["GPU", f"{gpu.get('name', 'unavailable')} / {gpu.get('memory.total [MiB]', 'unavailable')}"],
                 ["CPU", str(cpu.get("Name", "unavailable")).strip()],
                 ["RAM", f"{memory / 2**30:.2f} GiB usable" if memory else "unavailable"],
                 ["OS", machine.get("platform")],
                 ["NVIDIA driver", gpu.get("driver_version")],
                 ["Application", machine.get("application_version")],
                 ["Python evaluation", machine.get("python")],
                 ["Evaluation packages", "; ".join(f"{key} {value}" for key, value in am["versions"].items())],
             ]),
             "Python evaluation package versions and native build configuration are retained in the provenance report.", "",
             "## Latency", "", "All values are milliseconds; smaller is better. `p50` is the per-run median.", ""]
    latency = [item for item in summary if item["test"] == "latency_e2e_clean"]
    lines += [table(["Configuration", "Mean", "p50", "p95", "p99", "Run-mean range", "Repetitions"],
                   [[label(item["cell_id"]), *[f(item.get(f"median_of_run_{key}")) for key in ("mean_ms", "median_ms", "p95_ms", "p99_ms")],
                     f(item.get("minimum_run_mean_ms")) + "–" + f(item.get("maximum_run_mean_ms")),
                     f"{item['successful_repetitions']}/{item['expected_repetitions']}"] for item in latency])]
    if "latency.svg" in figures:
        lines += ["![Latency by backend and artifact](latency.svg)", ""]
    lines += ["## Streaming throughput", "", "Every tested pipeline depth is shown. FPS and individual-frame latency describe different properties.", "",
              table(["Configuration", "Depth", "FPS", "Run FPS range", "Frame mean, ms", "Frame p95, ms", "Repetitions"],
                    [[label(item["cell_id"]), item["depth"], f(item.get("median_of_run_fps")),
                      f(item.get("minimum_run_fps")) + "–" + f(item.get("maximum_run_fps")),
                      f(item.get("median_of_run_mean_ms")), f(item.get("median_of_run_p95_ms")),
                      f"{item['successful_repetitions']}/{item['expected_repetitions']}"]
                     for item in summary if item["test"] == "throughput_e2e_clean"])]
    if "throughput.svg" in figures:
        lines += ["![Streaming throughput across tested pipeline depths](throughput.svg)", ""]
    lines += ["## Backend timing and CUDA graph ablation", "",
              "Backend timing for the generic engine includes transfers and synchronization required by its public run call. It is not isolated kernel time. Fused CLI benchmarking exposes only the complete pipeline.", "",
              table(["Configuration", "Experiment", "Depth", "Mean, ms", "p95, ms", "FPS", "Repetitions"],
                    [[label(item["cell_id"]), item["test"], item["depth"], f(item.get("median_of_run_mean_ms")),
                      f(item.get("median_of_run_p95_ms")), f(item.get("median_of_run_fps")),
                      f"{item['successful_repetitions']}/{item['expected_repetitions']}"]
                     for item in summary if item["test"] not in {"latency_e2e_clean", "throughput_e2e_clean"}]),
              "Stage measurements and graph-use counts are in [performance.json](performance.json). Stages can overlap and their durations must not be summed to reconstruct total latency.", "",
              "## Detection quality", "",
              f"Evaluation uses the complete {am['split_image_count']:,}-image **test** split of the prepared Kaggle BDD100K YOLO dataset, "
              "at 640 × 640 and batch 1. This split name follows the prepared dataset; it is not a claim of official BDD100K leaderboard evaluation.",
              "Scores below use a 0–100 scale. mAP describes object-detection quality, not percentage of images classified correctly. "
              "Per-class metrics and raw 0–1 values are retained in [accuracy.json](accuracy.json).", "",
              table(["Artifact", "Status", "Precision", "Recall", "mAP50", "mAP75", "mAP50–95", "Strict export parity"],
                    [[item["artifact"], item["status"], *[f(item[key], True) for key in METRICS],
                      "reference" if item["artifact"] == "pytorch" else
                      "passed" if item["strict_export_parity"] is True else
                      "failed" if item["strict_export_parity"] is False else "unavailable"] for item in ar]),
              "A successful quality measurement can coexist with failed strict parity. Process exit codes and provider failures remain visible in the supporting reports.", "",
              "These are Python/Ultralytics artifact evaluations, not full-dataset measurements of each C++ backend. "
              "In particular, the TensorRT FP16 quality score must not be presented as measured fused-pipeline mAP.", "",
              "## Strict C++ parity", "",
              f"The fixed seed {am['parity_seed']} selects {am['parity_image_count']} images. Comparisons cover C++ output versus PyTorch "
              "and versus the same artifact through Python. Thresholds are preserved in [parity.json](parity.json); failures are not relaxed into passes.", "",
              table(["Configuration", "Reference", "Status", "Images passing", "Mean matched IoU", "Max box difference, px", "Max confidence difference"],
                    [[item["configuration"], item["reference"], item["status"],
                      f"{item['passed_images']}/{item['images']}" if item["images"] is not None else "—",
                      f(item["mean_matched_iou"]), f(item["max_box_abs_diff"]), f(item["max_conf_abs_diff"])] for item in pr]),
              "The original ONNX INT8 artifact produces no detections in this sample. Its same-artifact 64/64 parity passes "
              "compare empty outputs and do not indicate useful detection quality; its full-test-set mAP is zero.", "",
              "## Failures and compatibility", ""]
    failures = [run for run in pm["runs"] if run["status"] != "passed"]
    if failures:
        lines += [table(["Configuration / test", "Recorded failure"],
                        [[label(item["cell_id"]) + " / " + item["test"],
                          (item.get("failure_log_excerpt") or item["reason"])[-1800:]] for item in failures])]
    else:
        lines += ["All attempted performance runs completed successfully.", ""]
    lines += ["Failed initial inference prevents repeated timing for that configuration; no speed is assigned to it. "
              "The C++ application intentionally rejects QDQ INT8 ONNX with the TensorRT EP before provider execution. "
              "Python marks those provider checks unsupported; they are not successful TensorRT measurements. "
              "The saved INT8 engine has an Ultralytics metadata prefix that the Python loader strips but the C++ loaders do not; "
              "the deserialization error alone does not establish a TensorRT version mismatch. See [failure analysis](../RESULTS_ANALYSIS.md).", "",
              table(["Configuration", "Unavailable experiment", "Reason"],
                    [[label(item["cell_id"]), item["test"], item["reason"]] for item in pm.get("unsupported_tests", [])]),
              *( [table(["Quality artifact", "Failure"], [[item["artifact"], item["error"] or "See accuracy.json"]
                                                          for item in ar if item["status"] != "measured"])]
                 if any(item["status"] != "measured" for item in ar) else [] ),
              "Provider checks, including failures within otherwise successful quality evaluations:", "",
              table(["Artifact", "Provider request", "Status", "Selected providers / error"],
                    [[item["artifact"], item["backend"], item["status"],
                      ", ".join(item["actual_providers"] or []) if item["status"] == "passed" else item["error"]] for item in providers]),
              "## Build and test evidence", "",
              table(["Suite", "Tests", "Failures", "Errors", "Skipped"],
                    [[item[key] for key in ("file", "tests", "failures", "errors", "skipped")]
                     for item in tests["test_suites"]]),
              "Individual test cases and native configure/build/test outcomes are in [validation.json](validation.json). "
              "These are local results; they do not establish remote CI or container validation.", "",
              "Docker probe: `" + str((tests.get("docker") or {}).get("status", "not_recorded")) + "`. "
              "Container benchmarks executed: `" + str((tests.get("docker") or {}).get("container_benchmarks_executed", False)).lower() + "`.", "",
              "## Annotated outputs", "",
              "The examples below are actual C++ outputs from the fixed parity sample. A deterministic greedy selection prioritizes "
              "different detected class coverage, then detection count and filename. They illustrate output behavior and are not a substitute for test-set metrics.", ""]
    lines += ["Examples use BDD100K source imagery, which remains subject to its dataset terms; annotations were produced by this project. "
              "Fused examples demonstrate execution, not a strict parity pass.", ""]
    for demo in demos:
        lines += [f"![Detections for {demo['source_name']}]({demo['file']})", "",
                  f"`{demo['configuration']}` · `{demo['source_name']}` · {len(demo['detections'])} detections.", ""]
    if not demos:
        lines += ["Three eligible annotated outputs were unavailable; see the C++ execution failures above.", ""]
    lines += ["## Reproduce and inspect", "",
              "[All individual runs](runs.csv) · [Aggregated configurations](summary.csv) · [Accuracy CSV](accuracy.csv) · "
              "[Parity CSV](parity.csv) · [Provenance](provenance.json) · [Output checksums](SHA256SUMS.txt)", "",
              "See the [reproduction guide](../BENCHMARKING.md). Use [benchmark_release.py](../../tools/benchmark_release.py) and [evaluate_release.py](../../tools/evaluate_release.py) "
              "with the recorded arguments and artifact hashes. [publish_results.py](../../tools/publish_results.py) regenerates this publication only from a completed, verified suite. "
              "Full raw samples, process logs, and predictions remain local in the repository-relative evidence locations listed in provenance; "
              "those ignored directories are not included in a GitHub clone. This bundle includes compact records, aggregate metrics, "
              "artifact hashes, and selected annotated outputs, not the model files or complete raw predictions.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "benchmarks/release_20261011")
    parser.add_argument("--performance", type=Path, help="Override the source/performance directory")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    performance, accuracy = (args.performance or source / "performance").resolve(), source / "accuracy"
    pm, am, ar, cr = validate(performance, accuracy)
    tests = validation_evidence(source / "validation")
    summaries, runs = read(performance / "aggregate/summary.json"), read(performance / "aggregate/runs.json")
    require(len(runs) == len(pm["runs"]), "Aggregate run count is stale.")
    expected_run_ids = {item["run_id"] for item in pm["runs"]}
    require({item["run_id"] for item in runs} == expected_run_ids, "Aggregate runs do not match the manifest.")
    if args.validate_only:
        print(f"Verified {len(runs)} performance runs, {len(ar)} accuracy attempts, and {len(cr)} C++ configurations.")
        return 0
    require(not output.exists(), f"Refusing to overwrite an existing publication: {output}")
    output.mkdir(parents=True)
    accuracy_table, providers = accuracy_rows(ar)
    parity_table = parity_rows(cr)
    for name, rows in (("runs.csv", runs), ("summary.csv", summaries),
                       ("accuracy.csv", accuracy_table), ("parity.csv", parity_table)):
        write_csv(output / name, rows)
    write_json(output / "accuracy.json", {"rows": accuracy_table, "provider_checks": providers, "reports": ar})
    write_json(output / "parity.json", {"rows": parity_table, "reports": cr})
    write_json(output / "validation.json", tests)
    write_json(output / "performance.json", {"summary": summaries, "cells": pm["cells"],
               "unsupported_tests": pm.get("unsupported_tests", []),
               "stage_diagnostics": [item for item in pm["runs"] if item["test"] == "stage_diagnostic"],
               "failures": [item for item in pm["runs"] if item["status"] != "passed"],
               "runs": runs})
    provenance = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
                  "performance_started_utc": pm["started_utc"], "performance_finished_utc": pm["finished_utc"],
                  "performance_resumptions": pm.get("resumptions", []),
                  "incomplete_attempts": pm.get("incomplete_attempts", []),
                  "accuracy_manifest_created_utc": am["created_utc"],
                  "raw_evidence": {"performance": str(performance.relative_to(ROOT)),
                                   "accuracy": str(accuracy.relative_to(ROOT)),
                                   "validation": str((source / "validation").relative_to(ROOT))},
                  "git_head": pm["code_before"]["head"].get("stdout", "").strip(),
                  "compiled_source_sha256": pm["code_before"]["compiled_source_sha256"],
                  "compiled_source_files": pm["code_before"]["compiled_source_files"],
                  "evaluation_source_sha256": am["source_sha256"],
                  "executable": pm["executable"], "benchmark_runner": pm["runner"],
                  "publisher_sha256": digest(Path(__file__)), "build": pm["build"],
                  "hardware": hardware(pm), "evaluation_packages": am["versions"],
                  "benchmark_settings": pm["settings"], "benchmark_notes": pm["measurement_notes"],
                  "model_artifacts": pm["artifacts"], "quality_artifacts": am["artifacts"],
                  "performance_source_images": pm["source_frames"],
                  "evaluation_settings": am["settings"], "dataset": am["dataset"],
                  "dataset_yaml_sha256": am["data_yaml_sha256"],
                  "dataset_path_and_size_sha256": am["image_path_and_size_sha256"],
                  "parity_images": am["parity_images"], "parity_seed": am["parity_seed"],
                  "suite_manifest_sha256": {"performance": digest(performance / "manifest.json"),
                                             "accuracy": digest(accuracy / "manifest.json")}}
    write_json(output / "provenance.json", provenance)
    demos = examples(accuracy, cr, output)
    write_json(output / "examples.json", demos)
    figures = plots(output, summaries)
    (output / "README.md").write_text(report(pm, am, summaries, accuracy_table, parity_table,
                                            providers, tests, demos, figures), encoding="utf-8")
    checksums = "".join(f"{digest(path)}  {path.name}\n" for path in sorted(output.iterdir()) if path.is_file())
    (output / "SHA256SUMS.txt").write_text(checksums, encoding="utf-8")
    print(f"Prepared {output}; {len(runs)} runs, {len(accuracy_table)} artifacts, {len(demos)} annotated examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
