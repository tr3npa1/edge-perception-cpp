"""Run auditable accuracy and C++ detection-parity checks in isolated processes.

The accuracy phase evaluates the complete configured split. The C++ phase uses
a fixed, recorded image sample and preserves every generated detection output.
Failed artifacts remain in the report; one failure does not hide later results.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def package_versions() -> dict[str, str]:
    names = ("ultralytics", "torch", "torchvision", "onnx", "onnxruntime-gpu",
             "tensorrt_cu12", "numpy", "opencv-python", "pytest")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", PYTHONUNBUFFERED="1",
               PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    return env


def artifacts(args: argparse.Namespace) -> list[dict]:
    entries = [
        ("pytorch", "pytorch", args.weights, "fp32"),
        ("onnx_fp32", "onnx", ROOT / "models/onnx/yolo26m_bdd100k_fp32.onnx", "fp32"),
        ("onnx_fp16", "onnx", ROOT / "models/onnx/yolo26m_bdd100k_fp16.onnx", "fp16"),
        ("onnx_int8", "onnx", ROOT / "models/onnx/yolo26m_bdd100k_int8.onnx", "int8"),
        ("onnx_allconv_int8", "onnx", ROOT / "models/onnx/yolo26m_bdd100k_allconv_int8.onnx", "int8"),
        ("engine_fp32", "engine", ROOT / "models/engine/yolo26m_bdd100k_fp32.engine", "fp32"),
        ("engine_fp16", "engine", ROOT / "models/engine/yolo26m_bdd100k_fp16.engine", "fp16"),
        ("engine_int8", "engine", ROOT / "models/engine/yolo26m_bdd100k_int8.engine", "int8"),
    ]
    for path in args.extra_engine:
        precision = "int8" if "int8" in path.stem else "fp16" if "fp16" in path.stem else "fp32"
        entries.append(("extra_" + path.stem, "engine", path, precision))
    return [dict(label=label, type=kind, path=str(path.resolve()), precision=precision,
                 exists=path.is_file(), sha256=sha256(path) if path.is_file() else None,
                 bytes=path.stat().st_size if path.is_file() else None)
            for label, kind, path, precision in entries]


def build_manifest(args: argparse.Namespace) -> dict:
    from training.evaluate import collect_images, select_parity_images, validate_dataset_yaml

    dataset = validate_dataset_yaml(args.data, args.split)
    images = collect_images(dataset.split_image_roots)
    selected = select_parity_images(dataset_info=dataset, image_count=args.parity_images,
                                    seed=args.seed)
    if not images:
        raise ValueError("The requested evaluation split is empty.")
    inventory = "\n".join(str(path.relative_to(Path(dataset.dataset_root))).replace("\\", "/")
                          + ":" + str(path.stat().st_size) for path in images)
    source_files = [Path(name) for name in git("ls-files").splitlines()
                    if name.startswith(("src/", "include/", "training/", "tests/"))
                    or name in {"CMakeLists.txt", "CMakePresets.json"}]
    source_files.append(Path("tools/evaluate_release.py"))
    source_hashes = {str(path).replace("\\", "/"): sha256(ROOT / path)
                     for path in source_files if (ROOT / path).is_file()}
    return dict(created_utc=stamp(), git_head=git("rev-parse", "HEAD"),
                git_status=git("status", "--short"), source_sha256=source_hashes,
                versions=package_versions(), python=sys.version, executable=str(args.executable),
                executable_sha256=sha256(args.executable) if args.executable.is_file() else None,
                data_yaml=str(args.data), data_yaml_sha256=sha256(args.data),
                dataset=asdict(dataset), split_image_count=len(images),
                image_path_and_size_sha256=hashlib.sha256(inventory.encode()).hexdigest(),
                artifacts=artifacts(args), parity_seed=args.seed,
                parity_image_count=len(selected), parity_images=[dict(path=str(path), sha256=sha256(path))
                                                               for path in selected],
                settings=dict(split=args.split, imgsz=640, batch=1, device=args.device,
                              workers=args.workers, map_conf=0.001, parity_conf=0.25,
                              iou=0.7, max_det=300),
                notes=["Accuracy uses every image in the configured split.",
                       "Python validation speed is not the C++ deployment benchmark.",
                       "ONNX INT8 QDQ + ORT TensorRT is an explicitly unsupported combination.",
                       "Parity is reported even when it fails; thresholds are never adjusted to obtain a pass."])


def run_command(command: list[str], directory: Path, timeout: int, resume: bool) -> dict:
    result_path = directory / "process.json"
    if resume and result_path.is_file():
        result = read_json(result_path)
        if result.get("command") == command and result.get("completed"):
            print(f"Already recorded: {directory.name}", flush=True)
            return result
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing run: {directory}; use --resume or a new output.")
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(command=command, cwd=str(ROOT), started_utc=stamp(), completed=False)
    write_json(result_path, result)
    started = time.perf_counter()
    print(f"Starting {directory.name}; log: {directory / 'console.log'}", flush=True)
    with (directory / "console.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run(command, cwd=ROOT, env=environment(), stdout=log,
                                     stderr=subprocess.STDOUT, timeout=timeout, check=False)
            result.update(return_code=process.returncode, completed=True)
        except subprocess.TimeoutExpired:
            result.update(return_code=None, completed=True, error=f"Timed out after {timeout} seconds")
        except OSError as error:
            result.update(return_code=None, completed=True, error=str(error))
    result.update(finished_utc=stamp(), wall_seconds=time.perf_counter() - started)
    write_json(result_path, result)
    print(f"Finished {directory.name}: return_code={result['return_code']}", flush=True)
    return result


def evaluation_command(args: argparse.Namespace, artifact: dict) -> list[str]:
    command = [sys.executable, "-m", "training.evaluate", "--weights", str(args.weights),
               "--data", str(args.data), "--split", args.split, "--imgsz", "640", "--batch", "1",
               "--device", args.device, "--workers", str(args.workers),
               "--output", str(args.output / "evaluations"), "--name", artifact["label"],
               "--no-mlflow", "--continue-on-error", "--strict", "--save-json", "--plots",
               "--conf", "0.001", "--iou", "0.7", "--max-det", "300",
               "--parity-images", str(args.parity_images), "--parity-seed", str(args.seed),
               "--parity-conf", "0.25", "--parity-iou", "0.7",
               "--parity-match-iou", "0.90", "--parity-min-mean-iou", "0.98",
               "--parity-max-box-abs-diff", "2.0", "--parity-max-conf-abs-diff", "0.02",
               "--parity-max-count-diff", "1", "--ort-run-dummy",
               "--ort-tensorrt-cache-dir", str(args.output / "ort_cache")]
    if artifact["type"] == "pytorch":
        command += ["--eval-pytorch", "--no-eval-artifacts", "--no-parity"]
    else:
        command += ["--no-eval-pytorch", "--eval-artifacts", "--parity",
                    "--" + artifact["type"], artifact["path"]]
    command += ["--check-ort-cpu", "--check-ort-cuda", "--check-ort-tensorrt"]
    return command


def run_accuracy(args: argparse.Namespace, manifest: dict) -> list[dict]:
    results_path = args.output / "accuracy_results.json"
    previous = read_json(results_path) if results_path.is_file() else []
    reports_by_label = {item["label"]: item for item in previous}

    def save_reports() -> list[dict]:
        reports = [reports_by_label[item["label"]] for item in manifest["artifacts"]
                   if item["label"] in reports_by_label]
        write_json(results_path, reports)
        return reports

    for artifact in manifest["artifacts"]:
        if args.only_artifacts and artifact["label"] not in args.only_artifacts:
            continue
        if not artifact["exists"]:
            reports_by_label[artifact["label"]] = dict(label=artifact["label"], status="missing_artifact", artifact=artifact)
            save_reports()
            continue
        process = run_command(evaluation_command(args, artifact),
                              args.output / "processes" / artifact["label"],
                              args.timeout, args.resume)
        summary_path = args.output / "evaluations" / artifact["label"] / "evaluation_summary.json"
        report = dict(label=artifact["label"], artifact=artifact, process=process,
                      summary_path=str(summary_path),
                      evaluation=read_json(summary_path) if summary_path.is_file() else None)
        reports_by_label[artifact["label"]] = report
        save_reports()
    return save_reports()


def predict_reference(args: argparse.Namespace) -> None:
    """Worker: retain one model at a time, then let process exit release the GPU."""
    os.environ.update({key: value for key, value in environment().items()
                       if key in {"YOLO_AUTOINSTALL", "YOLO_OFFLINE", "MPLBACKEND"}})
    from ultralytics import YOLO
    from training.evaluate import predict_one_image

    manifest = read_json(args.output / "manifest.json")
    artifact = next(item for item in manifest["artifacts"] if item["label"] == args.reference)
    model = YOLO(artifact["path"])
    results = {}
    for index, item in enumerate(manifest["parity_images"], start=1):
        path = Path(item["path"])
        detections = predict_one_image(model=model, image_path=path, imgsz=640,
                                       conf=0.25, iou=0.7, max_det=300, device=args.device)
        results[path.name] = [asdict(detection) for detection in detections]
        print(f"Reference {artifact['label']} {index}/{len(manifest['parity_images'])}", flush=True)
    write_json(args.output / "references" / (artifact["label"] + ".json"),
               dict(artifact=artifact, detections=results, finished_utc=stamp()))


def compare_cpp(args: argparse.Namespace, manifest: dict, artifact: dict,
                output: Path) -> dict:
    from training.evaluate import Detection, compare_detection_sets, summarize_parity

    rows = [json.loads(line) for line in (output / "detections.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    predictions = {Path(row["source"]).name: [Detection(int(item["class_id"]),
                    float(item["confidence"]), [float(value) for value in item["box"]])
                    for item in row["detections"]] for row in rows}
    expected_names = {Path(item["path"]).name for item in manifest["parity_images"]}
    if len(rows) != len(expected_names) or set(predictions) != expected_names:
        raise ValueError("C++ detections do not contain exactly one row for every selected image.")
    comparisons = {}
    for reference_label in ("pytorch", artifact["label"]):
        reference_file = args.output / "references" / (reference_label + ".json")
        if not reference_file.is_file():
            comparisons[reference_label] = dict(success=False, error="Python reference generation failed")
            continue
        references = read_json(reference_file)["detections"]
        thresholds = (dict(match_iou_threshold=0.90, min_mean_iou=0.98,
                           max_box_abs_diff=2.0, max_conf_abs_diff=0.02, max_count_diff=1)
                      if reference_label == "pytorch" else
                      dict(match_iou_threshold=0.95, min_mean_iou=0.99,
                           max_box_abs_diff=1.5, max_conf_abs_diff=0.01, max_count_diff=0))
        details = [compare_detection_sets(image_path=Path(item["path"]),
                   reference_detections=[Detection(**entry) for entry in references[Path(item["path"]).name]],
                   artifact_detections=predictions[Path(item["path"]).name], **thresholds)
                   for item in manifest["parity_images"]]
        summary = summarize_parity(label=output.name, artifact_type="cpp", artifact_path=Path(artifact["path"]),
                                   details=details, thresholds=thresholds)
        comparisons[reference_label] = asdict(summary)
        write_json(output / ("parity_vs_" + reference_label + "_details.json"), [asdict(item) for item in details])
    write_json(output / "parity_summary.json", comparisons)
    return comparisons


def run_cpp(args: argparse.Namespace, manifest: dict) -> list[dict]:
    if not args.executable.is_file():
        raise FileNotFoundError(f"C++ executable not found: {args.executable}")
    source = args.output / "parity_images"
    source.mkdir(parents=True, exist_ok=True)
    if len({Path(item["path"]).name for item in manifest["parity_images"]}) != len(manifest["parity_images"]):
        raise ValueError("Selected source filenames must be unique for directory inference.")
    for item in manifest["parity_images"]:
        destination = source / Path(item["path"]).name
        if not destination.is_file():
            shutil.copy2(item["path"], destination)
        if sha256(destination) != item["sha256"]:
            raise ValueError(f"Parity image content changed: {destination}")
    for artifact in manifest["artifacts"]:
        if artifact["exists"]:
            command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                       "--reference", artifact["label"], "--device", args.device]
            run_command(command, args.output / "reference_processes" / artifact["label"],
                        args.timeout, args.resume)
    reports = []
    for artifact in manifest["artifacts"]:
        if artifact["type"] == "pytorch":
            continue
        backends = ("ort-cpu", "ort-cuda", "ort-trt") if artifact["type"] == "onnx" else ("native-trt", "fused-trt")
        for backend in backends:
            label = artifact["label"] + "__" + backend
            if not artifact["exists"]:
                reports.append(dict(label=label, status="missing_artifact"))
                continue
            if backend == "ort-trt" and artifact["precision"] == "int8":
                reports.append(dict(label=label, status="unsupported",
                                    reason="ONNX INT8 QDQ is unsupported by this project's ORT TensorRT path."))
                continue
            output = args.output / "cpp_outputs" / label
            command = [str(args.executable), "--mode", "infer", "--source", str(source),
                       "--backend", backend, "--model", artifact["path"],
                       "--precision", artifact["precision"], "--confidence", "0.25",
                       "--max-detections", "300", "--output-dir", str(output),
                       "--pipeline-depth", "1", "--ort-cache-dir", str(args.output / "cpp_ort_cache")]
            process = run_command(command, args.output / "cpp_processes" / label, args.timeout, args.resume)
            report = dict(label=label, artifact=artifact, backend=backend, process=process)
            try:
                if process.get("return_code") != 0:
                    raise RuntimeError("C++ inference process failed; see preserved console.log")
                report["comparisons"] = compare_cpp(args, manifest, artifact, output)
                selection_path = output / "backend_selection.json"
                report["backend_selection"] = read_json(selection_path) if selection_path.is_file() else None
            except Exception as error:
                report["error"] = str(error)
            reports.append(report)
            write_json(args.output / "cpp_parity_results.json", reports)
    write_json(args.output / "cpp_parity_results.json", reports)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/release_20261011/accuracy")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/train/yolo26m_bdd100k/weights/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/bdd100k_yolo/bdd100k.yaml")
    parser.add_argument("--executable", type=Path, default=ROOT / "build/nvidia-release/bin/Release/edge_perception.exe")
    parser.add_argument("--extra-engine", type=Path, action="append", default=[])
    parser.add_argument("--only-artifacts", nargs="+",
                        help="Run accuracy only for these labels; retain the complete input manifest and earlier results.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--parity-images", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--phase", choices=("accuracy", "cpp", "all"), default="all")
    parser.add_argument("--plan", action="store_true", help="Record inputs and commands without loading any models.")
    parser.add_argument("--resume", action="store_true", help="Reuse recorded completed attempts; failed attempts stay visible.")
    parser.add_argument("--reference", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for field in ("output", "weights", "data", "executable"):
        setattr(args, field, getattr(args, field).resolve())
    if args.reference:
        predict_reference(args)
        return 0
    if args.parity_images <= 0 or args.workers < 0 or args.timeout <= 0:
        parser.error("parity-images and timeout must be positive; workers must be nonnegative")
    if args.only_artifacts and args.phase != "accuracy":
        parser.error("--only-artifacts requires --phase accuracy")
    manifest = build_manifest(args)
    if args.only_artifacts:
        unknown = set(args.only_artifacts) - {item["label"] for item in manifest["artifacts"]}
        if unknown:
            parser.error("Unknown artifact labels: " + ", ".join(sorted(unknown)))
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        for key in ("source_sha256", "artifacts", "data_yaml_sha256", "image_path_and_size_sha256",
                    "parity_images", "settings", "executable_sha256", "versions"):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f"Recorded inputs changed ({key}); choose a new output directory.")
        manifest = previous
    else:
        write_json(manifest_path, manifest)
    write_json(args.output / "evaluation_commands.json",
               {item["label"]: evaluation_command(args, item) for item in manifest["artifacts"]})
    if args.plan:
        print(json.dumps(dict(manifest=str(manifest_path), images=manifest["split_image_count"],
                              artifacts=len(manifest["artifacts"]), parity_images=manifest["parity_image_count"])))
        return 0
    if args.phase in ("accuracy", "all"):
        run_accuracy(args, manifest)
    if args.phase in ("cpp", "all"):
        run_cpp(args, manifest)
    write_json(args.output / "completion.json", dict(finished_utc=stamp(), phase=args.phase,
               note="All scheduled attempts completed. Read per-artifact reports for failures and parity mismatches."))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
