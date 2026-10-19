"""Score the fixed 64-image FP16 correctness sample; never a full-test mAP claim."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_release import environment, git, package_versions, sha256, stamp, write_json


def score_predictions(predictions: dict, targets: dict, names: dict, output: Path) -> dict:
    """Use Ultralytics' validator matching and AP implementation, on CPU."""
    import numpy as np
    import torch
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils.metrics import DetMetrics
    from training.evaluate import to_jsonable

    if set(predictions) != set(targets):
        raise ValueError("Predictions must contain exactly one entry for every selected image.")
    validator = DetectionValidator(save_dir=output, args={"plots": False, "device": "cpu"})
    metrics = DetMetrics(names=names)
    for name, target in targets.items():
        rows = predictions[name]
        pred = {"bboxes": torch.tensor([x["xyxy"] for x in rows], dtype=torch.float32).reshape(-1, 4),
                "cls": torch.tensor([x["class_id"] for x in rows], dtype=torch.float32)}
        gt = {"bboxes": torch.tensor(target["xyxy"], dtype=torch.float32).reshape(-1, 4),
              "cls": torch.tensor(target["classes"], dtype=torch.float32)}
        metrics.update_stats({**validator._process_batch(pred, gt),
                              "conf": np.array([x["confidence"] for x in rows]),
                              "pred_cls": pred["cls"].numpy(), "target_cls": gt["cls"].numpy(),
                              "target_img": np.unique(gt["cls"].numpy()), "im_name": name})
    metrics.process(plot=False)
    return to_jsonable({"images": len(targets), "ground_truth_boxes": sum(len(t["classes"]) for t in targets.values()),
            "detections": sum(map(len, predictions.values())), **metrics.results_dict,
            "map75": float(metrics.box.all_ap[:, 5].mean()) if metrics.box.all_ap.size else 0.0,
            "per_class": metrics.summary()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=ROOT / "benchmarks/fp16_correctness_20261016/sample.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/bdd100k_yolo/bdd100k.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--weights", type=Path, help="Optionally also evaluate this PyTorch checkpoint.")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    args.executable, args.sample, args.output = (p.resolve() for p in (args.executable, args.sample, args.output))
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {args.output}")
    if not args.executable.is_file():
        raise FileNotFoundError(args.executable)
    os.environ.update(YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", MPLBACKEND="Agg")

    import cv2
    import numpy as np
    import yaml
    from ultralytics.utils.ops import xywh2xyxy

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    entries = sample["images"]
    if len(entries) != 64 or len({Path(x["path"]).name for x in entries}) != 64:
        raise ValueError("This check requires exactly 64 uniquely named sample images.")
    artifact = next(x for x in sample["artifacts"] if x["label"] == "engine_fp16")
    engine = Path(artifact["path"]).resolve()
    if sha256(engine) != artifact["sha256"]:
        raise ValueError("Engine content differs from the recorded sample artifact.")
    names = yaml.safe_load(args.data.read_text(encoding="utf-8"))["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    targets, input_records = {}, []
    for entry in entries:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Image content changed: {path}")
        label = path.parent.parent / "labels" / (path.stem + ".txt")
        rows = np.array([list(map(float, line.split())) for line in label.read_text().splitlines()
                         if line.strip()], dtype=np.float32).reshape(-1, 5)
        if not np.isfinite(rows).all() or (rows[:, 1:] < 0).any() or (rows[:, 1:] > 1).any():
            raise ValueError(f"Invalid normalized YOLO labels: {label}")
        if any(c != int(c) or int(c) not in names for c in rows[:, 0]):
            raise ValueError(f"Invalid class IDs: {label}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        h, w = image.shape[:2]
        boxes = xywh2xyxy(rows[:, 1:]) * np.array([w, h, w, h], dtype=np.float32)
        targets[path.name] = {"classes": rows[:, 0].tolist(), "xyxy": boxes.tolist()}
        input_records.append({**entry, "label_path": str(label), "label_sha256": sha256(label),
                              "width": w, "height": h})
    sources = sorted(set(git("ls-files", "src", "include", "training", "CMakeLists.txt").splitlines()
                         + ["tools/evaluate_release.py", "tools/check_fp16_sample_quality.py"]))
    provenance = {"created_utc": stamp(), "scope": "fixed_64_image_sample_only_not_full_test_accuracy",
                  "sample": str(args.sample), "sample_sha256": sha256(args.sample), "inputs": input_records,
                  "engine": artifact, "data_sha256": sha256(args.data), "class_names": names,
                  "executable": str(args.executable), "executable_sha256": sha256(args.executable),
                  "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
                  "source_sha256": {p: sha256(ROOT / p) for p in sources}, "versions": package_versions(),
                  "settings": {"confidence": 0.001, "imgsz": 640, "rect": False, "max_det": 300,
                               "iou": 0.7, "metric_iou": "0.50:0.05:0.95", "device": args.device,
                               "metric_coordinates": "original_image_xyxy", "metric_device": "cpu",
                               "precision_recall": "Ultralytics best-mean-F1 operating point"},
                  "method": "Ultralytics DetectionValidator._process_batch and DetMetrics.process"}
    if args.weights:
        provenance["pytorch"] = {"path": str(args.weights.resolve()), "sha256": sha256(args.weights)}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "provenance.json", provenance)
    write_json(args.output / "ground_truth.json", targets)
    source = args.output / "images"
    source.mkdir()
    for entry in entries:
        shutil.copy2(entry["path"], source / Path(entry["path"]).name)

    results = {}
    for backend in ("native-trt", "fused-trt"):
        output = args.output / backend
        output.mkdir()
        command = [str(args.executable), "--mode", "infer", "--source", str(source),
                   "--backend", backend, "--model", str(engine), "--precision", "fp16",
                   "--confidence", "0.001", "--max-detections", "300", "--pipeline-depth", "1",
                   "--output-dir", str(output)]
        write_json(output / "command.json", command)
        with (output / "console.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=args.timeout)
        rows = [json.loads(line) for line in (output / "detections.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != len(entries):
            raise ValueError(f"{backend} did not produce exactly 64 output rows.")
        predictions = {Path(row["source"]).name: [{"class_id": d["class_id"], "confidence": d["confidence"],
                       "xyxy": d["box"]} for d in row["detections"]] for row in rows}
        write_json(output / "predictions.json", predictions)
        results[backend] = score_predictions(predictions, targets, names, output)
        write_json(args.output / "partial_metrics.json", results)
        print(f"{backend}: {len(entries)} images; mAP50-95={results[backend]['metrics/mAP50-95(B)']:.6f}", flush=True)

    from ultralytics import YOLO
    from training.evaluate import predict_one_image
    import gc
    import torch

    for label, path in [("python-engine", engine)] + ([("pytorch", args.weights)] if args.weights else []):
        model = YOLO(str(path))
        predictions = {Path(x["path"]).name: [asdict(d) for d in predict_one_image(
            model=model, image_path=Path(x["path"]), imgsz=640, conf=0.001, iou=0.7,
            max_det=300, device=args.device)] for x in entries}
        output = args.output / label
        write_json(output / "predictions.json", predictions)
        results[label] = score_predictions(predictions, targets, names, output)
        write_json(args.output / "partial_metrics.json", results)
        print(f"{label}: {len(entries)} images; mAP50-95={results[label]['metrics/mAP50-95(B)']:.6f}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / "summary.json", {"scope": provenance["scope"], "completed_utc": stamp(),
               "provenance_sha256": sha256(args.output / "provenance.json"), "metrics": results})


if __name__ == "__main__":
    main()
