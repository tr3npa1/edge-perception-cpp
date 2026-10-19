"""Targeted FP16 correctness checks; never run throughput or full-dataset tests.

Reuse a recorded image selection, isolate model processes, and retain the
original strict thresholds. Each output directory is independent of the old
release measurements. Run reference workers separately to bound GPU memory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABELS = ("pytorch", "onnx_fp32", "onnx_fp16", "engine_fp16")
STRICT = dict(match_iou_threshold=0.95, min_mean_iou=0.99,
              max_box_abs_diff=1.5, max_conf_abs_diff=0.01, max_count_diff=0)
PYTORCH = dict(match_iou_threshold=0.90, min_mean_iou=0.98,
               max_box_abs_diff=2.0, max_conf_abs_diff=0.02, max_count_diff=1)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def identity():
    files = [*ROOT.glob("src/*"), *ROOT.glob("include/*.hpp"),
             ROOT / "training/evaluate.py", Path(__file__)]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
            for path in sorted(files) if path.is_file()}


def sample(args):
    original = read(args.manifest)
    images = original["parity_images"]
    artifacts = [item for item in original["artifacts"] if item["label"] in LABELS]
    for item in [*images, *artifacts]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError(f"Recorded input changed: {item['path']}")
    if len({Path(item["path"]).name for item in images}) != len(images):
        raise ValueError("Image filenames must be unique")
    record = dict(images=images, artifacts=artifacts, seed=original["parity_seed"],
                  source_manifest_sha256=digest(args.manifest))
    target = args.output / "sample.json"
    if target.exists() and read(target) != record:
        raise ValueError("Existing sample differs; choose a fresh output directory")
    if not target.exists():
        save(target, record)
    return record


def reference(args, selection):
    from ultralytics import YOLO
    from training.evaluate import predict_one_image

    target = args.output / "references" / f"{args.reference}.json"
    if target.exists():
        raise FileExistsError(target)
    artifact = next(item for item in selection["artifacts"] if item["label"] == args.reference)
    model = YOLO(artifact["path"])
    detections = {}
    for image in selection["images"]:
        path = Path(image["path"])
        detections[path.name] = [asdict(item) for item in predict_one_image(
            model=model, image_path=path, imgsz=640, conf=0.25,
            iou=0.7, max_det=300, device="0")]
    save(target, dict(artifact=artifact, detections=detections,
                     source_sha256=identity(), rect=False,
                     completed_utc=datetime.now(timezone.utc).isoformat()))
    print(f"Reference {args.reference}: {len(detections)} images", flush=True)


def run_cpp(args, selection):
    artifact = next(item for item in selection["artifacts"] if item["label"] == "engine_fp16")
    source = args.output / "images"
    source.mkdir(exist_ok=True)
    for item in selection["images"]:
        target = source / Path(item["path"]).name
        if not target.exists():
            shutil.copyfile(item["path"], target)
        if digest(target) != item["sha256"]:
            raise ValueError(f"Staged input changed: {target}")
    variants = {
        "native": ("native-trt", []),
        "fused": ("fused-trt", []),
        "fused_graph_off": ("fused-trt", ["--no-cuda-graph"]),
        "fused_cpu_postprocess": ("fused-trt", ["--debug-parity"]),
    }
    for name, (backend, extra) in variants.items():
        output = args.output / "cpp" / name
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        command = [str(args.executable), "--mode", "infer", "--source", str(source),
                   "--backend", backend, "--model", artifact["path"], "--precision", "fp16",
                   "--pipeline-depth", "1", "--confidence", "0.25", "--max-detections", "300",
                   "--output-dir", str(output), *extra]
        with (output / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        save(output / "process.json", dict(command=command, exit_code=process.returncode,
             executable_sha256=digest(args.executable), source_sha256=identity()))
        if process.returncode:
            raise RuntimeError(f"{name} exited {process.returncode}; see {output / 'console.log'}")
        print(f"C++ {name}: completed", flush=True)


def compare(args, selection):
    from training.evaluate import Detection, compare_detection_sets, summarize_parity

    predictions = {}
    for path in (args.output / "references").glob("*.json"):
        predictions[path.stem] = read(path)["detections"]
    for folder in (args.output / "cpp").iterdir():
        rows = [json.loads(line) for line in (folder / "detections.jsonl").read_text().splitlines() if line]
        predictions[folder.name] = {Path(row["source"]).name: [
            dict(class_id=item["class_id"], confidence=item["confidence"], xyxy=item["box"])
            for item in row["detections"]] for row in rows}
        if len(rows) != len(predictions[folder.name]):
            raise ValueError("Duplicate C++ image records")
    names = {Path(item["path"]).name for item in selection["images"]}
    for name, values in predictions.items():
        if set(values) != names:
            raise ValueError(f"Incomplete image coverage: {name}")
    pairs = [(name, "pytorch", PYTORCH) for name in predictions if name != "pytorch"]
    pairs += [(name, "engine_fp16", STRICT) for name in ("native", "fused", "fused_graph_off", "fused_cpu_postprocess")]
    pairs += [(name, "native", STRICT) for name in ("fused", "fused_graph_off", "fused_cpu_postprocess")]
    results = []
    for candidate, reference_name, thresholds in pairs:
        details = [compare_detection_sets(image_path=Path(item["path"]),
            reference_detections=[Detection(**entry) for entry in predictions[reference_name][Path(item["path"]).name]],
            artifact_detections=[Detection(**entry) for entry in predictions[candidate][Path(item["path"]).name]],
            **thresholds) for item in selection["images"]]
        summary = summarize_parity(label=candidate, artifact_type="correctness",
            artifact_path=args.executable, details=details, thresholds=thresholds)
        result = dict(candidate=candidate, reference=reference_name, **asdict(summary))
        results.append(result)
        save(args.output / "comparisons" / f"{candidate}_vs_{reference_name}.json",
             [asdict(item) for item in details])
        print(f"{candidate} vs {reference_name}: {summary.passed_image_count}/{summary.image_count}")
    save(args.output / "summary.json", dict(results=results, source_sha256=identity(),
         note="64-image correctness only; not a throughput or full-test-set quality benchmark."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/release_20261011/accuracy/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path, default=ROOT / "build/fp16-correctness/bin/Release/edge_perception.exe")
    parser.add_argument("--phase", choices=("reference", "cpp", "compare"), required=True)
    parser.add_argument("--reference", choices=LABELS)
    args = parser.parse_args()
    if args.phase == "reference" and not args.reference:
        parser.error("--reference is required for the reference phase")
    args.output = args.output.resolve()
    args.executable = args.executable.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", MPLBACKEND="Agg")
    selection = sample(args)
    {"reference": reference, "cpp": run_cpp, "compare": compare}[args.phase](args, selection)


if __name__ == "__main__":
    main()
