"""
Evaluate YOLO26M checkpoints and exported deployment artifacts.

This script is the evaluation entrypoint for edge-perception-cpp.

Expected PyTorch model input:
    runs/detect/runs/train/<run_name>/weights/best.pt

Expected dataset input:
    data/processed/bdd100k_yolo/bdd100k.yaml

Optional exported artifacts:
    models/onnx/yolo26m_bdd100k_fp32.onnx
    models/onnx/yolo26m_bdd100k_fp16.onnx
    models/onnx/yolo26m_bdd100k_int8.onnx
    models/engine/yolo26m_bdd100k_fp32.engine
    models/engine/yolo26m_bdd100k_fp16.engine
    models/engine/yolo26m_bdd100k_int8.engine

Responsibilities:
- evaluate PyTorch YOLO26M checkpoint quality
- optionally evaluate exported ONNX artifacts with Ultralytics validation
- optionally evaluate native TensorRT engine artifacts with Ultralytics validation
- report mAP@50, mAP@75, mAP@50-95, precision, recall, and per-class metrics
- optionally check ONNX Runtime CPU/CUDA/TensorRT Execution Provider session readiness
- optionally compare final detections from PyTorch against ONNX and TensorRT artifacts
- log evaluation parameters, metrics, checks, parity results, and JSON artifacts to MLflow
- save local JSON summaries for all evaluations, backend checks, and parity checks

This script does not train.
This script does not export models.
This script does not benchmark deployment latency.
This script does not replace tests/test_parity.py.
This script does not replace tests/test_inference.cpp.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger("evaluate_yolo26m")

DEFAULT_WEIGHTS = Path("runs/train/yolo26m_bdd100k/weights/best.pt")
DEFAULT_DATA_YAML = Path("data/processed/bdd100k_yolo/bdd100k.yaml")
DEFAULT_OUTPUT_DIR = Path("runs/evaluate")
DEFAULT_RUN_NAME = "yolo26m_bdd100k_eval"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "edge-perception-cpp"
DEFAULT_ORT_TRT_CACHE_DIR = Path("models/ort_trt_cache")

KAGGLE_PATH_MARKER = "/kaggle/input"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetInfo:
    """Resolved dataset information used for evaluation."""

    yaml_path: str
    dataset_root: str
    split: str
    split_image_roots: list[str]
    class_count: int
    class_names: list[str]


@dataclass(frozen=True)
class Detection:
    """One final postprocessed detection."""

    class_id: int
    confidence: float
    xyxy: list[float]


@dataclass(frozen=True)
class ModelEvaluation:
    """Validation result for one model artifact."""

    label: str
    artifact_type: str
    path: str
    success: bool
    seconds: float
    metrics: dict[str, Any] | None
    output_dir: str | None
    error: str | None


@dataclass(frozen=True)
class OrtBackendCheck:
    """ONNX Runtime provider/session check result for one ONNX artifact."""

    label: str
    onnx_path: str
    backend: str
    provider_request: list[Any]
    success: bool
    available_providers: list[str] | None
    actual_providers: list[str] | None
    inputs: list[dict[str, Any]] | None
    outputs: list[dict[str, Any]] | None
    dummy_inference: dict[str, Any] | None
    seconds: float
    error: str | None


@dataclass(frozen=True)
class ImageParityResult:
    """PyTorch-vs-artifact final-detection parity result for one image."""

    image_path: str
    reference_detection_count: int
    artifact_detection_count: int
    matched_detection_count: int
    unmatched_reference_count: int
    unmatched_artifact_count: int
    mean_matched_iou: float | None
    min_matched_iou: float | None
    mean_box_abs_diff: float | None
    max_box_abs_diff: float | None
    mean_conf_abs_diff: float | None
    max_conf_abs_diff: float | None
    passed: bool


@dataclass(frozen=True)
class ParitySummary:
    """Aggregated final-detection parity result for one exported artifact."""

    label: str
    artifact_type: str
    artifact_path: str
    enabled: bool
    success: bool
    passed: bool | None
    reason: str
    image_count: int
    passed_image_count: int
    failed_image_count: int
    total_reference_detections: int
    total_artifact_detections: int
    total_matched_detections: int
    mean_matched_iou: float | None
    min_matched_iou: float | None
    mean_box_abs_diff: float | None
    max_box_abs_diff: float | None
    mean_conf_abs_diff: float | None
    max_conf_abs_diff: float | None
    thresholds: dict[str, Any]
    error: str | None


@dataclass(frozen=True)
class ArtifactReport:
    """All evaluation/check/parity outputs for one exported artifact."""

    label: str
    artifact_type: str
    path: str
    validation: ModelEvaluation | None
    ort_backend_checks: list[OrtBackendCheck]
    parity: ParitySummary | None


@dataclass(frozen=True)
class EvaluationSummary:
    """Top-level evaluation summary."""

    created_unix_time: float
    duration_seconds: float
    output_dir: str
    weights: str
    data_yaml: str
    split: str
    imgsz: int
    batch: int
    device: str
    mlflow_enabled: bool
    mlflow_run_id: str | None
    dataset: DatasetInfo
    pytorch_evaluation: ModelEvaluation | None
    artifacts: list[ArtifactReport]
    args: dict[str, Any]


def setup_logging() -> None:
    """Configure readable console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate YOLO26M PyTorch, ONNX, and TensorRT artifacts."
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to trained PyTorch checkpoint, usually best.pt.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Prepared local BDD100K dataset YAML.",
    )

    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Evaluation image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Evaluation batch size. Default is 1 because exported ONNX/TensorRT artifacts are usually static batch=1.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Evaluation device, for example '0' or 'cpu'.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for evaluation outputs.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Evaluation run name.",
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing into an existing evaluation directory.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold used during mAP validation.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold used during validation NMS.",
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image.",
    )

    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use half precision during PyTorch validation if supported.",
    )

    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save Ultralytics evaluation plots.",
    )

    parser.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask Ultralytics to save JSON predictions when supported.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose Ultralytics validation logging.",
    )

    parser.add_argument(
        "--eval-pytorch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run PyTorch best.pt mAP evaluation.",
    )

    parser.add_argument(
        "--eval-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Ultralytics mAP validation for provided ONNX/TensorRT artifacts.",
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        nargs="*",
        default=[],
        help="One or more ONNX artifacts to evaluate/check.",
    )

    parser.add_argument(
        "--engine",
        type=Path,
        nargs="*",
        default=[],
        help="One or more native TensorRT .engine artifacts to evaluate/check.",
    )

    parser.add_argument(
        "--check-ort-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check ONNX Runtime CPUExecutionProvider sessions for ONNX artifacts.",
    )

    parser.add_argument(
        "--check-ort-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Check ONNX Runtime CUDAExecutionProvider sessions for ONNX artifacts.",
    )

    parser.add_argument(
        "--check-ort-tensorrt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Check ONNX Runtime TensorRTExecutionProvider sessions for ONNX artifacts. "
            "Disabled by default because QDQ INT8 ONNX models can emit TensorRT parser logs "
            "even when normal ONNX/CUDA validation succeeds."
        ),
    )

    parser.add_argument(
        "--ort-tensorrt-cache-dir",
        type=Path,
        default=DEFAULT_ORT_TRT_CACHE_DIR,
        help="Engine-cache directory for ONNX Runtime TensorRT EP session checks.",
    )

    parser.add_argument(
        "--ort-run-dummy",
        action="store_true",
        help="Run one dummy inference during ONNX Runtime provider checks.",
    )

    parser.add_argument(
        "--parity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compare PyTorch final detections against each provided exported artifact. "
            "Disabled by default because backend/NMS/INT8 numerical differences can make "
            "strict parity fail even when mAP validation succeeds. Enable with --parity "
            "when debugging deployment equivalence."
        ),
    )

    parser.add_argument(
        "--parity-images",
        type=int,
        default=32,
        help="Number of split images used for final-detection parity.",
    )

    parser.add_argument(
        "--parity-seed",
        type=int,
        default=42,
        help="Random seed used to select parity images.",
    )

    parser.add_argument(
        "--parity-conf",
        type=float,
        default=0.25,
        help="Confidence threshold used for parity predictions.",
    )

    parser.add_argument(
        "--parity-iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold used for parity predictions.",
    )

    parser.add_argument(
        "--parity-match-iou",
        type=float,
        default=0.90,
        help="Minimum IoU required to match reference and artifact detections.",
    )

    parser.add_argument(
        "--parity-min-mean-iou",
        type=float,
        default=0.95,
        help="Minimum mean matched IoU required per image.",
    )

    parser.add_argument(
        "--parity-max-box-abs-diff",
        type=float,
        default=3.0,
        help="Maximum allowed absolute box-coordinate difference in pixels.",
    )

    parser.add_argument(
        "--parity-max-conf-abs-diff",
        type=float,
        default=0.05,
        help="Maximum allowed confidence difference for matched detections.",
    )

    parser.add_argument(
        "--parity-max-count-diff",
        type=int,
        default=1,
        help="Maximum allowed unmatched detection count per image.",
    )

    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue evaluating other artifacts if one artifact/backend fails.",
    )

    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return non-zero exit code if any optional artifact validation/check/parity fails.",
    )

    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log this evaluation run to MLflow.",
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. If omitted, MLflow uses its default/local configuration.",
    )

    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=DEFAULT_MLFLOW_EXPERIMENT_NAME,
        help="MLflow experiment name for evaluation runs.",
    )

    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="MLflow run name for this evaluation.",
    )

    parser.add_argument(
        "--mlflow-log-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log local evaluation JSON/results directory as MLflow artifacts.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/configuration and exit without evaluating or logging to MLflow.",
    )

    return parser.parse_args()


def args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Convert parsed arguments into JSON-serializable values."""

    output: dict[str, Any] = {}

    for key, value in vars(args).items():
        if isinstance(value, Path):
            output[key] = str(value)
        elif isinstance(value, list):
            output[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            output[key] = value

    return output


def to_jsonable(value: Any) -> Any:
    """Convert common Python, tensor, and array objects into JSON-safe values."""

    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    if hasattr(value, "detach"):
        return to_jsonable(value.detach().cpu().numpy())

    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())

    try:
        parsed = float(value)
    except Exception:
        return str(value)

    return parsed if math.isfinite(parsed) else None


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(payload), file, indent=2)


def safe_name(text: str) -> str:
    """Create a filesystem-safe short name."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    cleaned = cleaned.strip("._")

    return cleaned or "artifact"


def safe_mlflow_key(text: str) -> str:
    """Create a conservative MLflow metric/parameter key."""

    cleaned = re.sub(r"[^A-Za-z0-9_./ -]+", "_", text)
    cleaned = cleaned.strip(" ./_")
    cleaned = cleaned.replace(" ", "_")

    return cleaned or "value"


def truncate_param_value(value: Any, max_length: int = 500) -> str:
    """Convert a parameter value to a safe short string for MLflow."""

    text = str(to_jsonable(value))

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries for metric/parameter logging."""

    flattened: dict[str, Any] = {}

    for key, value in payload.items():
        joined_key = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            flattened.update(flatten_dict(value, joined_key))
        else:
            flattened[joined_key] = value

    return flattened


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file as a dictionary."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}, got {type(payload).__name__}.")

    return payload


def normalize_class_names(names_payload: Any) -> list[str]:
    """Normalize YOLO class names from list or dictionary format."""

    if isinstance(names_payload, list):
        class_names = [str(name) for name in names_payload]

        if not class_names:
            raise ValueError("Dataset YAML 'names' list is empty.")

        return class_names

    if isinstance(names_payload, dict):
        class_names_by_id: dict[int, str] = {}

        for key, value in names_payload.items():
            class_id = int(key)
            class_names_by_id[class_id] = str(value)

        expected_ids = list(range(len(class_names_by_id)))
        actual_ids = sorted(class_names_by_id)

        if actual_ids != expected_ids:
            raise ValueError(
                "Dataset YAML class IDs must be contiguous from 0. "
                f"Expected {expected_ids}, got {actual_ids}."
            )

        return [class_names_by_id[index] for index in expected_ids]

    raise ValueError("Dataset YAML must contain 'names' as a list or dictionary.")


def ensure_not_raw_kaggle_yaml(yaml_payload: dict[str, Any]) -> None:
    """Reject Kaggle-only YAML paths before evaluation starts."""

    path_fields = [
        yaml_payload.get("path"),
        yaml_payload.get("train"),
        yaml_payload.get("val"),
        yaml_payload.get("test"),
    ]

    for field in path_fields:
        if field is None:
            continue

        values = [str(item) for item in field] if isinstance(field, list) else [str(field)]

        for value in values:
            normalized_value = value.replace("\\", "/")

            if KAGGLE_PATH_MARKER in normalized_value:
                raise ValueError(
                    "Dataset YAML still contains Kaggle-only paths. "
                    "Run data/dataset.py and evaluate with "
                    "data/processed/bdd100k_yolo/bdd100k.yaml."
                )


def resolve_dataset_root(data_yaml: Path, yaml_payload: dict[str, Any]) -> Path:
    """Resolve the dataset root from a YOLO dataset YAML."""

    raw_path = yaml_payload.get("path")

    if raw_path is None:
        return data_yaml.parent.resolve()

    dataset_root = Path(str(raw_path))

    if dataset_root.is_absolute():
        return dataset_root.resolve()

    cwd_candidate = dataset_root.resolve()

    if cwd_candidate.exists():
        return cwd_candidate

    return (data_yaml.parent / dataset_root).resolve()


def split_entry_to_list(split_entry: Any) -> list[str]:
    """Normalize one YOLO split entry to a list of path strings."""

    if split_entry is None:
        return []

    if isinstance(split_entry, str):
        return [split_entry]

    if isinstance(split_entry, list):
        if not all(isinstance(item, str) for item in split_entry):
            raise ValueError("Dataset split list entries must all be strings.")

        return list(split_entry)

    raise ValueError(
        "Dataset split entries must be strings or lists of strings. "
        f"Got {type(split_entry).__name__}."
    )


def resolve_split_paths(split_entry: Any, dataset_root: Path) -> list[Path]:
    """Resolve one split entry from a YOLO dataset YAML."""

    resolved_paths: list[Path] = []

    for item in split_entry_to_list(split_entry):
        split_path = Path(item)

        if split_path.is_absolute():
            resolved_paths.append(split_path.resolve())
        else:
            resolved_paths.append((dataset_root / split_path).resolve())

    return resolved_paths


def validate_dataset_yaml(data_yaml: Path, split: str) -> DatasetInfo:
    """Validate the prepared local dataset YAML before evaluation."""

    data_yaml = data_yaml.resolve()
    yaml_payload = read_yaml(data_yaml)

    ensure_not_raw_kaggle_yaml(yaml_payload)

    class_names = normalize_class_names(yaml_payload.get("names"))
    nc_value = yaml_payload.get("nc")

    if nc_value is not None and int(nc_value) != len(class_names):
        raise ValueError(
            f"Dataset YAML nc={nc_value} does not match len(names)={len(class_names)}."
        )

    dataset_root = resolve_dataset_root(data_yaml, yaml_payload)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    split_paths = resolve_split_paths(yaml_payload.get(split), dataset_root)

    if not split_paths:
        raise ValueError(f"Dataset YAML does not define split: {split}")

    for split_path in split_paths:
        if not split_path.is_dir():
            raise FileNotFoundError(f"{split} image directory not found: {split_path}")

    return DatasetInfo(
        yaml_path=str(data_yaml),
        dataset_root=str(dataset_root),
        split=split,
        split_image_roots=[str(path) for path in split_paths],
        class_count=len(class_names),
        class_names=class_names,
    )


def collect_images(image_roots: list[str]) -> list[Path]:
    """Collect image paths from one or more image root directories."""

    images: list[Path] = []

    for root_text in image_roots:
        root = Path(root_text)

        if not root.is_dir():
            raise FileNotFoundError(f"Image root not found: {root}")

        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path.resolve())

    return sorted(images)


def make_output_dir(base_output: Path, name: str, exist_ok: bool) -> Path:
    """Create an evaluation output directory."""

    base_output = base_output.resolve()
    output_dir = base_output / name

    if output_dir.exists() and not exist_ok:
        index = 2

        while True:
            candidate = base_output / f"{name}{index}"

            if not candidate.exists():
                output_dir = candidate
                break

            index += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def as_float(value: Any) -> float | None:
    """Convert a value to finite float when possible."""

    if value is None:
        return None

    try:
        parsed = float(value)
    except Exception:
        return None

    if not math.isfinite(parsed):
        return None

    return parsed


def array_to_list(value: Any) -> list[Any]:
    """Convert array-like objects to a Python list."""

    if value is None:
        return []

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, list):
        return value

    return []


def mean(values: list[float]) -> float | None:
    """Return mean for non-empty numeric list."""

    if not values:
        return None

    return sum(values) / len(values)


def compute_map75_from_all_ap(all_ap_payload: Any) -> float | None:
    """
    Compute mean AP@75 from all_ap when direct map75 is unavailable.

    Common Ultralytics all_ap layout:
        [class_count][10 IoU thresholds]

    Threshold index 5 corresponds to IoU 0.75.
    """

    all_ap = array_to_list(all_ap_payload)
    ap75_values: list[float] = []

    for class_values in all_ap:
        class_ap_values = array_to_list(class_values)

        if len(class_ap_values) <= 5:
            continue

        value = as_float(class_ap_values[5])

        if value is not None:
            ap75_values.append(value)

    return mean(ap75_values)


def extract_per_class_metrics(metrics: Any, class_names: list[str]) -> dict[str, Any]:
    """Extract per-class detection metrics when available."""

    box = getattr(metrics, "box", None)

    if box is None:
        return {}

    precision_values = array_to_list(getattr(box, "p", None))
    recall_values = array_to_list(getattr(box, "r", None))
    map_values = array_to_list(getattr(box, "maps", None))
    all_ap_values = array_to_list(getattr(box, "all_ap", None))

    per_class: dict[str, Any] = {}

    for class_id, class_name in enumerate(class_names):
        row: dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "precision": None,
            "recall": None,
            "ap50": None,
            "ap75": None,
            "ap50_95": None,
        }

        if class_id < len(precision_values):
            row["precision"] = as_float(precision_values[class_id])

        if class_id < len(recall_values):
            row["recall"] = as_float(recall_values[class_id])

        if class_id < len(map_values):
            row["ap50_95"] = as_float(map_values[class_id])

        if class_id < len(all_ap_values):
            class_ap_values = array_to_list(all_ap_values[class_id])

            if len(class_ap_values) > 0:
                row["ap50"] = as_float(class_ap_values[0])

            if len(class_ap_values) > 5:
                row["ap75"] = as_float(class_ap_values[5])

        per_class[str(class_id)] = row

    return per_class


def extract_metrics(metrics: Any, class_names: list[str]) -> dict[str, Any]:
    """Extract important Ultralytics validation metrics."""

    box = getattr(metrics, "box", None)

    summary: dict[str, Any] = {
        "precision": None,
        "recall": None,
        "map50": None,
        "map75": None,
        "map50_95": None,
        "fitness": None,
        "speed": to_jsonable(getattr(metrics, "speed", None)),
        "results_dict": to_jsonable(getattr(metrics, "results_dict", None)),
        "per_class": {},
    }

    if box is not None:
        summary["precision"] = as_float(getattr(box, "mp", None))
        summary["recall"] = as_float(getattr(box, "mr", None))
        summary["map50"] = as_float(getattr(box, "map50", None))
        summary["map75"] = as_float(getattr(box, "map75", None))
        summary["map50_95"] = as_float(getattr(box, "map", None))
        summary["per_class"] = extract_per_class_metrics(metrics, class_names)

        if summary["map75"] is None:
            summary["map75"] = compute_map75_from_all_ap(getattr(box, "all_ap", None))

    fitness = getattr(metrics, "fitness", None)

    if callable(fitness):
        summary["fitness"] = as_float(fitness())
    else:
        summary["fitness"] = as_float(fitness)

    return summary


def run_ultralytics_validation(
    *,
    label: str,
    artifact_type: str,
    model_path: Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    output_dir: Path,
    conf: float,
    iou: float,
    max_det: int,
    half: bool,
    plots: bool,
    save_json: bool,
    verbose: bool,
    class_names: list[str],
) -> ModelEvaluation:
    """Run Ultralytics validation for one model artifact."""

    from ultralytics import YOLO

    start_time = time.time()
    run_name = safe_name(f"{label}_val")
    run_output_dir = output_dir / run_name

    try:
        LOGGER.info("Running validation for %s:", label)
        LOGGER.info("  type: %s", artifact_type)
        LOGGER.info("  path: %s", model_path)

        model = YOLO(str(model_path))

        metrics = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            project=str(output_dir),
            name=run_name,
            exist_ok=True,
            conf=conf,
            iou=iou,
            max_det=max_det,
            half=half if artifact_type == "pytorch" else False,
            plots=plots,
            save_json=save_json,
            verbose=verbose,
        )

        extracted_metrics = extract_metrics(metrics, class_names)

        return ModelEvaluation(
            label=label,
            artifact_type=artifact_type,
            path=str(model_path),
            success=True,
            seconds=time.time() - start_time,
            metrics=extracted_metrics,
            output_dir=str(run_output_dir),
            error=None,
        )
    except Exception as error:
        return ModelEvaluation(
            label=label,
            artifact_type=artifact_type,
            path=str(model_path),
            success=False,
            seconds=time.time() - start_time,
            metrics=None,
            output_dir=str(run_output_dir),
            error=str(error),
        )


def build_provider_request(
    *,
    backend: str,
    cache_dir: Path,
) -> list[Any]:
    """Build ONNX Runtime provider request for a backend profile."""

    if backend == "ort_cpu":
        return ["CPUExecutionProvider"]

    if backend == "ort_cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if backend == "ort_tensorrt":
        provider_options = {
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": str(cache_dir),
        }

        return [
            ("TensorrtExecutionProvider", provider_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    raise ValueError(f"Unsupported ONNX Runtime backend: {backend}")


def required_provider_name(backend: str) -> str:
    """Return required provider name for a backend profile."""

    if backend == "ort_cpu":
        return "CPUExecutionProvider"

    if backend == "ort_cuda":
        return "CUDAExecutionProvider"

    if backend == "ort_tensorrt":
        return "TensorrtExecutionProvider"

    raise ValueError(f"Unsupported ONNX Runtime backend: {backend}")


def ort_type_to_numpy_dtype(ort_type: str) -> Any:
    """Map an ONNX Runtime tensor type string to a NumPy dtype."""

    import numpy as np

    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(uint16)": np.uint16,
        "tensor(int16)": np.int16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(bool)": np.bool_,
    }

    return mapping.get(ort_type, np.float32)


def ort_shape_to_dummy_shape(shape: list[Any], batch: int, imgsz: int) -> list[int]:
    """Convert ONNX Runtime input shape metadata into a dummy tensor shape."""

    dummy_shape: list[int] = []

    for index, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            dummy_shape.append(dim)
            continue

        if index == 0:
            dummy_shape.append(batch)
        elif index == 1:
            dummy_shape.append(3)
        elif index in (2, 3):
            dummy_shape.append(imgsz)
        else:
            dummy_shape.append(1)

    return dummy_shape


def check_ort_backend(
    *,
    label: str,
    onnx_path: Path,
    backend: str,
    cache_dir: Path,
    run_dummy: bool,
    batch: int,
    imgsz: int,
) -> OrtBackendCheck:
    """Check ONNX Runtime session creation for one ONNX backend profile."""

    start_time = time.time()
    provider_request = build_provider_request(backend=backend, cache_dir=cache_dir)

    try:
        import numpy as np
        import onnxruntime as ort

        _ = np
        available_providers = ort.get_available_providers()
        required_provider = required_provider_name(backend)

        if required_provider not in available_providers:
            return OrtBackendCheck(
                label=label,
                onnx_path=str(onnx_path),
                backend=backend,
                provider_request=provider_request,
                success=False,
                available_providers=available_providers,
                actual_providers=None,
                inputs=None,
                outputs=None,
                dummy_inference=None,
                seconds=time.time() - start_time,
                error=f"{required_provider} is not available.",
            )

        if backend == "ort_tensorrt":
            cache_dir.mkdir(parents=True, exist_ok=True)

        session = ort.InferenceSession(
            str(onnx_path),
            providers=provider_request,
        )

        inputs = [
            {
                "name": item.name,
                "shape": list(item.shape),
                "type": item.type,
            }
            for item in session.get_inputs()
        ]

        outputs = [
            {
                "name": item.name,
                "shape": list(item.shape),
                "type": item.type,
            }
            for item in session.get_outputs()
        ]

        dummy_inference: dict[str, Any] | None = None

        if run_dummy:
            if not inputs:
                raise RuntimeError("ONNX Runtime session has no inputs.")

            first_input = session.get_inputs()[0]
            dummy_shape = ort_shape_to_dummy_shape(
                shape=list(first_input.shape),
                batch=batch,
                imgsz=imgsz,
            )
            dummy_dtype = ort_type_to_numpy_dtype(first_input.type)

            dummy_input = np.zeros(dummy_shape, dtype=dummy_dtype)
            output_arrays = session.run(None, {first_input.name: dummy_input})

            dummy_inference = {
                "input_name": first_input.name,
                "input_shape": dummy_shape,
                "input_dtype": str(dummy_input.dtype),
                "output_shapes": [list(array.shape) for array in output_arrays],
                "output_dtypes": [str(array.dtype) for array in output_arrays],
            }

        return OrtBackendCheck(
            label=label,
            onnx_path=str(onnx_path),
            backend=backend,
            provider_request=provider_request,
            success=True,
            available_providers=available_providers,
            actual_providers=session.get_providers(),
            inputs=inputs,
            outputs=outputs,
            dummy_inference=dummy_inference,
            seconds=time.time() - start_time,
            error=None,
        )
    except Exception as error:
        available_providers = None

        try:
            import onnxruntime as ort

            available_providers = ort.get_available_providers()
        except Exception:
            pass

        return OrtBackendCheck(
            label=label,
            onnx_path=str(onnx_path),
            backend=backend,
            provider_request=provider_request,
            success=False,
            available_providers=available_providers,
            actual_providers=None,
            inputs=None,
            outputs=None,
            dummy_inference=None,
            seconds=time.time() - start_time,
            error=str(error),
        )


def requested_ort_backends(args: argparse.Namespace) -> list[str]:
    """Return requested ONNX Runtime backend checks."""

    backends: list[str] = []

    if args.check_ort_cpu:
        backends.append("ort_cpu")

    if args.check_ort_cuda:
        backends.append("ort_cuda")

    if args.check_ort_tensorrt:
        backends.append("ort_tensorrt")

    return backends


def run_ort_backend_checks(
    *,
    label: str,
    onnx_path: Path,
    args: argparse.Namespace,
) -> list[OrtBackendCheck]:
    """Run requested ONNX Runtime backend checks for one ONNX artifact."""

    checks: list[OrtBackendCheck] = []

    for backend in requested_ort_backends(args):
        LOGGER.info("Checking %s for %s", backend, onnx_path)

        check = check_ort_backend(
            label=label,
            onnx_path=onnx_path,
            backend=backend,
            cache_dir=args.ort_tensorrt_cache_dir,
            run_dummy=args.ort_run_dummy,
            batch=1,
            imgsz=args.imgsz,
        )

        checks.append(check)

        if check.success:
            LOGGER.info("  %s passed.", backend)
        else:
            LOGGER.warning("  %s failed: %s", backend, check.error)

    return checks


def extract_detections(result: Any) -> list[Detection]:
    """Extract final detections from one Ultralytics Results object."""

    boxes = getattr(result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        return []

    xyxy_array = boxes.xyxy.detach().cpu().numpy()
    conf_array = boxes.conf.detach().cpu().numpy()
    cls_array = boxes.cls.detach().cpu().numpy()

    detections: list[Detection] = []

    for xyxy, confidence, class_id in zip(xyxy_array, conf_array, cls_array):
        detections.append(
            Detection(
                class_id=int(class_id),
                confidence=float(confidence),
                xyxy=[float(value) for value in xyxy.tolist()],
            )
        )

    detections.sort(key=lambda item: (item.class_id, -item.confidence))
    return detections


def predict_one_image(
    *,
    model: Any,
    image_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> list[Detection]:
    """Run prediction on one image and return final detections."""

    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
    )

    if not results:
        return []

    return extract_detections(results[0])


def box_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU for two xyxy boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def box_abs_diffs(box_a: list[float], box_b: list[float]) -> list[float]:
    """Return absolute xyxy coordinate differences."""

    return [abs(a - b) for a, b in zip(box_a, box_b)]


def compare_detection_sets(
    *,
    image_path: Path,
    reference_detections: list[Detection],
    artifact_detections: list[Detection],
    match_iou_threshold: float,
    min_mean_iou: float,
    max_box_abs_diff: float,
    max_conf_abs_diff: float,
    max_count_diff: int,
) -> ImageParityResult:
    """Compare PyTorch and artifact final detections for one image."""

    unmatched_artifact_indices = set(range(len(artifact_detections)))
    matched_ious: list[float] = []
    matched_box_abs_diffs: list[float] = []
    matched_conf_abs_diffs: list[float] = []
    matched_reference_indices: set[int] = set()

    reference_order = sorted(
        range(len(reference_detections)),
        key=lambda index: reference_detections[index].confidence,
        reverse=True,
    )

    for reference_index in reference_order:
        reference_detection = reference_detections[reference_index]

        best_artifact_index: int | None = None
        best_iou = 0.0

        for artifact_index in unmatched_artifact_indices:
            artifact_detection = artifact_detections[artifact_index]

            if reference_detection.class_id != artifact_detection.class_id:
                continue

            candidate_iou = box_iou_xyxy(reference_detection.xyxy, artifact_detection.xyxy)

            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_artifact_index = artifact_index

        if best_artifact_index is None or best_iou < match_iou_threshold:
            continue

        artifact_detection = artifact_detections[best_artifact_index]

        unmatched_artifact_indices.remove(best_artifact_index)
        matched_reference_indices.add(reference_index)
        matched_ious.append(best_iou)
        matched_box_abs_diffs.extend(
            box_abs_diffs(reference_detection.xyxy, artifact_detection.xyxy)
        )
        matched_conf_abs_diffs.append(
            abs(reference_detection.confidence - artifact_detection.confidence)
        )

    unmatched_reference_count = len(reference_detections) - len(matched_reference_indices)
    unmatched_artifact_count = len(unmatched_artifact_indices)

    mean_matched_iou = mean(matched_ious)
    min_matched_iou = min(matched_ious) if matched_ious else None
    mean_box_diff = mean(matched_box_abs_diffs)
    max_box_diff = max(matched_box_abs_diffs) if matched_box_abs_diffs else None
    mean_conf_diff = mean(matched_conf_abs_diffs)
    max_conf_diff = max(matched_conf_abs_diffs) if matched_conf_abs_diffs else None

    both_empty = len(reference_detections) == 0 and len(artifact_detections) == 0

    count_ok = abs(len(reference_detections) - len(artifact_detections)) <= max_count_diff
    unmatched_ok = (
        unmatched_reference_count <= max_count_diff
        and unmatched_artifact_count <= max_count_diff
    )
    iou_ok = both_empty or (
        mean_matched_iou is not None and mean_matched_iou >= min_mean_iou
    )
    box_ok = both_empty or (
        max_box_diff is not None and max_box_diff <= max_box_abs_diff
    )
    conf_ok = both_empty or (
        max_conf_diff is not None and max_conf_diff <= max_conf_abs_diff
    )

    passed = count_ok and unmatched_ok and iou_ok and box_ok and conf_ok

    return ImageParityResult(
        image_path=str(image_path),
        reference_detection_count=len(reference_detections),
        artifact_detection_count=len(artifact_detections),
        matched_detection_count=len(matched_ious),
        unmatched_reference_count=unmatched_reference_count,
        unmatched_artifact_count=unmatched_artifact_count,
        mean_matched_iou=mean_matched_iou,
        min_matched_iou=min_matched_iou,
        mean_box_abs_diff=mean_box_diff,
        max_box_abs_diff=max_box_diff,
        mean_conf_abs_diff=mean_conf_diff,
        max_conf_abs_diff=max_conf_diff,
        passed=passed,
    )


def select_parity_images(
    *,
    dataset_info: DatasetInfo,
    image_count: int,
    seed: int,
) -> list[Path]:
    """Select deterministic random images for parity checking."""

    all_images = collect_images(dataset_info.split_image_roots)

    if not all_images:
        raise ValueError(f"No images found for split: {dataset_info.split}")

    rng = random.Random(seed)
    shuffled_images = list(all_images)
    rng.shuffle(shuffled_images)

    return shuffled_images[: min(image_count, len(shuffled_images))]


def parity_thresholds(args: argparse.Namespace) -> dict[str, Any]:
    """Return parity threshold configuration."""

    return {
        "parity_conf": args.parity_conf,
        "parity_iou": args.parity_iou,
        "max_det": args.max_det,
        "match_iou_threshold": args.parity_match_iou,
        "min_mean_iou": args.parity_min_mean_iou,
        "max_box_abs_diff": args.parity_max_box_abs_diff,
        "max_conf_abs_diff": args.parity_max_conf_abs_diff,
        "max_count_diff": args.parity_max_count_diff,
    }


def summarize_parity(
    *,
    label: str,
    artifact_type: str,
    artifact_path: Path,
    details: list[ImageParityResult],
    thresholds: dict[str, Any],
) -> ParitySummary:
    """Aggregate per-image parity results."""

    passed_count = sum(1 for item in details if item.passed)
    failed_count = len(details) - passed_count

    mean_ious = [
        item.mean_matched_iou
        for item in details
        if item.mean_matched_iou is not None
    ]
    min_ious = [
        item.min_matched_iou
        for item in details
        if item.min_matched_iou is not None
    ]
    mean_box_diffs = [
        item.mean_box_abs_diff
        for item in details
        if item.mean_box_abs_diff is not None
    ]
    max_box_diffs = [
        item.max_box_abs_diff
        for item in details
        if item.max_box_abs_diff is not None
    ]
    mean_conf_diffs = [
        item.mean_conf_abs_diff
        for item in details
        if item.mean_conf_abs_diff is not None
    ]
    max_conf_diffs = [
        item.max_conf_abs_diff
        for item in details
        if item.max_conf_abs_diff is not None
    ]

    return ParitySummary(
        label=label,
        artifact_type=artifact_type,
        artifact_path=str(artifact_path),
        enabled=True,
        success=True,
        passed=failed_count == 0,
        reason="completed",
        image_count=len(details),
        passed_image_count=passed_count,
        failed_image_count=failed_count,
        total_reference_detections=sum(item.reference_detection_count for item in details),
        total_artifact_detections=sum(item.artifact_detection_count for item in details),
        total_matched_detections=sum(item.matched_detection_count for item in details),
        mean_matched_iou=mean(mean_ious),
        min_matched_iou=min(min_ious) if min_ious else None,
        mean_box_abs_diff=mean(mean_box_diffs),
        max_box_abs_diff=max(max_box_diffs) if max_box_diffs else None,
        mean_conf_abs_diff=mean(mean_conf_diffs),
        max_conf_abs_diff=max(max_conf_diffs) if max_conf_diffs else None,
        thresholds=thresholds,
        error=None,
    )


def disabled_parity_summary(
    *,
    label: str,
    artifact_type: str,
    artifact_path: Path,
    reason: str,
    thresholds: dict[str, Any],
) -> ParitySummary:
    """Build a disabled parity summary."""

    return ParitySummary(
        label=label,
        artifact_type=artifact_type,
        artifact_path=str(artifact_path),
        enabled=False,
        success=True,
        passed=None,
        reason=reason,
        image_count=0,
        passed_image_count=0,
        failed_image_count=0,
        total_reference_detections=0,
        total_artifact_detections=0,
        total_matched_detections=0,
        mean_matched_iou=None,
        min_matched_iou=None,
        mean_box_abs_diff=None,
        max_box_abs_diff=None,
        mean_conf_abs_diff=None,
        max_conf_abs_diff=None,
        thresholds=thresholds,
        error=None,
    )


def failed_parity_summary(
    *,
    label: str,
    artifact_type: str,
    artifact_path: Path,
    error: Exception,
    thresholds: dict[str, Any],
) -> ParitySummary:
    """Build a failed parity summary."""

    return ParitySummary(
        label=label,
        artifact_type=artifact_type,
        artifact_path=str(artifact_path),
        enabled=True,
        success=False,
        passed=False,
        reason="failed",
        image_count=0,
        passed_image_count=0,
        failed_image_count=0,
        total_reference_detections=0,
        total_artifact_detections=0,
        total_matched_detections=0,
        mean_matched_iou=None,
        min_matched_iou=None,
        mean_box_abs_diff=None,
        max_box_abs_diff=None,
        mean_conf_abs_diff=None,
        max_conf_abs_diff=None,
        thresholds=thresholds,
        error=str(error),
    )


def run_final_detection_parity(
    *,
    label: str,
    artifact_type: str,
    reference_path: Path,
    artifact_path: Path,
    dataset_info: DatasetInfo,
    args: argparse.Namespace,
) -> tuple[ParitySummary, list[ImageParityResult]]:
    """Run PyTorch-vs-artifact final-detection parity."""

    from ultralytics import YOLO

    selected_images = select_parity_images(
        dataset_info=dataset_info,
        image_count=args.parity_images,
        seed=args.parity_seed,
    )

    LOGGER.info("Running final-detection parity for %s", label)
    LOGGER.info("  reference: %s", reference_path)
    LOGGER.info("  artifact: %s", artifact_path)
    LOGGER.info("  images: %d", len(selected_images))

    reference_model = YOLO(str(reference_path))
    artifact_model = YOLO(str(artifact_path))

    details: list[ImageParityResult] = []

    for index, image_path in enumerate(selected_images, start=1):
        LOGGER.info("Parity image %d/%d: %s", index, len(selected_images), image_path)

        reference_detections = predict_one_image(
            model=reference_model,
            image_path=image_path,
            imgsz=args.imgsz,
            conf=args.parity_conf,
            iou=args.parity_iou,
            max_det=args.max_det,
            device=args.device,
        )

        artifact_detections = predict_one_image(
            model=artifact_model,
            image_path=image_path,
            imgsz=args.imgsz,
            conf=args.parity_conf,
            iou=args.parity_iou,
            max_det=args.max_det,
            device=args.device,
        )

        details.append(
            compare_detection_sets(
                image_path=image_path,
                reference_detections=reference_detections,
                artifact_detections=artifact_detections,
                match_iou_threshold=args.parity_match_iou,
                min_mean_iou=args.parity_min_mean_iou,
                max_box_abs_diff=args.parity_max_box_abs_diff,
                max_conf_abs_diff=args.parity_max_conf_abs_diff,
                max_count_diff=args.parity_max_count_diff,
            )
        )

    summary = summarize_parity(
        label=label,
        artifact_type=artifact_type,
        artifact_path=artifact_path,
        details=details,
        thresholds=parity_thresholds(args),
    )

    return summary, details


def validate_required_paths(args: argparse.Namespace) -> None:
    """Validate input paths."""

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    for onnx_path in args.onnx:
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX artifact not found: {onnx_path}")

        if onnx_path.suffix.lower() != ".onnx":
            raise ValueError(f"Expected .onnx artifact, got: {onnx_path}")

    for engine_path in args.engine:
        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine artifact not found: {engine_path}")

        if engine_path.suffix.lower() != ".engine":
            raise ValueError(f"Expected .engine artifact, got: {engine_path}")


def artifact_label(path: Path, prefix: str) -> str:
    """Build a stable artifact label."""

    return safe_name(f"{prefix}_{path.stem}")


def log_model_evaluation(evaluation: ModelEvaluation) -> None:
    """Log model evaluation result to console."""

    if not evaluation.success:
        LOGGER.error("%s validation failed: %s", evaluation.label, evaluation.error)
        return

    metrics = evaluation.metrics or {}

    LOGGER.info("%s validation metrics:", evaluation.label)
    LOGGER.info("  precision: %s", metrics.get("precision"))
    LOGGER.info("  recall: %s", metrics.get("recall"))
    LOGGER.info("  mAP@50: %s", metrics.get("map50"))
    LOGGER.info("  mAP@75: %s", metrics.get("map75"))
    LOGGER.info("  mAP@50-95: %s", metrics.get("map50_95"))
    LOGGER.info("  fitness: %s", metrics.get("fitness"))


def evaluate_artifact(
    *,
    artifact_path: Path,
    artifact_type: str,
    label: str,
    data_yaml: Path,
    dataset_info: DatasetInfo,
    output_dir: Path,
    args: argparse.Namespace,
) -> ArtifactReport:
    """Evaluate one ONNX or TensorRT engine artifact."""

    validation: ModelEvaluation | None = None
    ort_checks: list[OrtBackendCheck] = []
    parity_summary: ParitySummary | None = None
    parity_details: list[ImageParityResult] = []

    if args.eval_artifacts:
        validation = run_ultralytics_validation(
            label=label,
            artifact_type=artifact_type,
            model_path=artifact_path,
            data_yaml=data_yaml,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            output_dir=output_dir,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            half=False,
            plots=args.plots,
            save_json=args.save_json,
            verbose=args.verbose,
            class_names=dataset_info.class_names,
        )

        log_model_evaluation(validation)

    if artifact_type == "onnx":
        ort_checks = run_ort_backend_checks(
            label=label,
            onnx_path=artifact_path,
            args=args,
        )

    if args.parity:
        try:
            parity_summary, parity_details = run_final_detection_parity(
                label=label,
                artifact_type=artifact_type,
                reference_path=args.weights,
                artifact_path=artifact_path,
                dataset_info=dataset_info,
                args=args,
            )
        except Exception as error:
            parity_summary = failed_parity_summary(
                label=label,
                artifact_type=artifact_type,
                artifact_path=artifact_path,
                error=error,
                thresholds=parity_thresholds(args),
            )

            if not args.continue_on_error:
                raise
    else:
        parity_summary = disabled_parity_summary(
            label=label,
            artifact_type=artifact_type,
            artifact_path=artifact_path,
            reason="disabled_by_user",
            thresholds=parity_thresholds(args),
        )

    artifact_dir = output_dir / "artifacts" / label
    write_json(artifact_dir / "validation.json", asdict(validation) if validation else None)
    write_json(artifact_dir / "ort_backend_checks.json", [asdict(item) for item in ort_checks])
    write_json(artifact_dir / "parity_summary.json", asdict(parity_summary))
    write_json(artifact_dir / "parity_details.json", [asdict(item) for item in parity_details])

    return ArtifactReport(
        label=label,
        artifact_type=artifact_type,
        path=str(artifact_path),
        validation=validation,
        ort_backend_checks=ort_checks,
        parity=parity_summary,
    )


def has_failure(summary: EvaluationSummary) -> bool:
    """Return True if any requested evaluation/check/parity failed."""

    if summary.pytorch_evaluation is not None and not summary.pytorch_evaluation.success:
        return True

    for artifact in summary.artifacts:
        if artifact.validation is not None and not artifact.validation.success:
            return True

        for check in artifact.ort_backend_checks:
            if not check.success:
                return True

        if artifact.parity is not None:
            if not artifact.parity.success:
                return True

            if artifact.parity.passed is False:
                return True

    return False


def import_mlflow_or_raise() -> Any:
    """Import MLflow with a clear project-level error message."""

    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow logging is enabled, but mlflow is not installed. "
            "Install dependencies with:\n\n"
            "    pip install -r requirements.txt\n\n"
            "or disable evaluation logging with --no-mlflow."
        ) from error

    return mlflow


def start_mlflow_run(args: argparse.Namespace) -> tuple[Any | None, Any | None]:
    """Start an MLflow run unless disabled."""

    if not args.mlflow or args.dry_run:
        return None, None

    mlflow = import_mlflow_or_raise()

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    mlflow.set_experiment(args.mlflow_experiment_name)
    active_run = mlflow.start_run(run_name=args.mlflow_run_name)

    mlflow.set_tags(
        {
            "project": "edge-perception-cpp",
            "stage": "evaluation",
            "model_family": "YOLO26M",
            "dataset": "BDD100K YOLO",
            "split": args.split,
        }
    )

    return mlflow, active_run


def safe_mlflow_log_param(mlflow: Any, key: str, value: Any) -> None:
    """Log one MLflow parameter without breaking evaluation on logging quirks."""

    try:
        mlflow.log_param(safe_mlflow_key(key), truncate_param_value(value))
    except Exception as error:
        LOGGER.warning("Failed to log MLflow param %s: %s", key, error)


def safe_mlflow_log_metric(mlflow: Any, key: str, value: Any) -> None:
    """Log one numeric MLflow metric if possible."""

    numeric_value = as_float(value)

    if numeric_value is None:
        return

    try:
        mlflow.log_metric(safe_mlflow_key(key), numeric_value)
    except Exception as error:
        LOGGER.warning("Failed to log MLflow metric %s: %s", key, error)


def log_initial_mlflow_params(
    *,
    mlflow: Any | None,
    args: argparse.Namespace,
    dataset_info: DatasetInfo,
    output_dir: Path,
) -> None:
    """Log run configuration parameters to MLflow."""

    if mlflow is None:
        return

    params = {
        "weights": args.weights,
        "data_yaml": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "half": args.half,
        "plots": args.plots,
        "save_json": args.save_json,
        "eval_pytorch": args.eval_pytorch,
        "eval_artifacts": args.eval_artifacts,
        "onnx_count": len(args.onnx),
        "engine_count": len(args.engine),
        "onnx_paths": args.onnx,
        "engine_paths": args.engine,
        "check_ort_cpu": args.check_ort_cpu,
        "check_ort_cuda": args.check_ort_cuda,
        "check_ort_tensorrt": args.check_ort_tensorrt,
        "ort_tensorrt_cache_dir": args.ort_tensorrt_cache_dir,
        "ort_run_dummy": args.ort_run_dummy,
        "parity": args.parity,
        "parity_images": args.parity_images,
        "parity_seed": args.parity_seed,
        "parity_conf": args.parity_conf,
        "parity_iou": args.parity_iou,
        "parity_match_iou": args.parity_match_iou,
        "parity_min_mean_iou": args.parity_min_mean_iou,
        "parity_max_box_abs_diff": args.parity_max_box_abs_diff,
        "parity_max_conf_abs_diff": args.parity_max_conf_abs_diff,
        "parity_max_count_diff": args.parity_max_count_diff,
        "continue_on_error": args.continue_on_error,
        "strict": args.strict,
        "output_dir": output_dir,
        "dataset_root": dataset_info.dataset_root,
        "class_count": dataset_info.class_count,
        "class_names": dataset_info.class_names,
    }

    for key, value in params.items():
        safe_mlflow_log_param(mlflow, key, value)


def log_evaluation_metrics_to_mlflow(
    *,
    mlflow: Any | None,
    evaluation: ModelEvaluation,
    prefix: str,
) -> None:
    """Log model validation metrics to MLflow."""

    if mlflow is None:
        return

    safe_mlflow_log_metric(mlflow, f"{prefix}/success", 1.0 if evaluation.success else 0.0)
    safe_mlflow_log_metric(mlflow, f"{prefix}/seconds", evaluation.seconds)

    if evaluation.error:
        safe_mlflow_log_param(mlflow, f"{prefix}/error", evaluation.error)

    if not evaluation.metrics:
        return

    for metric_name in ["precision", "recall", "map50", "map75", "map50_95", "fitness"]:
        safe_mlflow_log_metric(
            mlflow,
            f"{prefix}/{metric_name}",
            evaluation.metrics.get(metric_name),
        )

    speed = evaluation.metrics.get("speed")

    if isinstance(speed, dict):
        for key, value in speed.items():
            safe_mlflow_log_metric(mlflow, f"{prefix}/speed/{key}", value)

    per_class = evaluation.metrics.get("per_class")

    if isinstance(per_class, dict):
        for class_id, class_metrics in per_class.items():
            if not isinstance(class_metrics, dict):
                continue

            class_prefix = f"{prefix}/class_{class_id}"

            for metric_name in ["precision", "recall", "ap50", "ap75", "ap50_95"]:
                safe_mlflow_log_metric(
                    mlflow,
                    f"{class_prefix}/{metric_name}",
                    class_metrics.get(metric_name),
                )


def log_ort_checks_to_mlflow(
    *,
    mlflow: Any | None,
    checks: list[OrtBackendCheck],
    prefix: str,
) -> None:
    """Log ONNX Runtime backend check results to MLflow."""

    if mlflow is None:
        return

    for check in checks:
        check_prefix = f"{prefix}/ort/{check.backend}"
        safe_mlflow_log_metric(mlflow, f"{check_prefix}/success", 1.0 if check.success else 0.0)
        safe_mlflow_log_metric(mlflow, f"{check_prefix}/seconds", check.seconds)

        if check.error:
            safe_mlflow_log_param(mlflow, f"{check_prefix}/error", check.error)

        if check.available_providers is not None:
            safe_mlflow_log_param(
                mlflow,
                f"{check_prefix}/available_providers",
                check.available_providers,
            )

        if check.actual_providers is not None:
            safe_mlflow_log_param(
                mlflow,
                f"{check_prefix}/actual_providers",
                check.actual_providers,
            )


def log_parity_to_mlflow(
    *,
    mlflow: Any | None,
    parity: ParitySummary | None,
    prefix: str,
) -> None:
    """Log parity summary to MLflow."""

    if mlflow is None or parity is None:
        return

    safe_mlflow_log_metric(mlflow, f"{prefix}/parity/enabled", 1.0 if parity.enabled else 0.0)
    safe_mlflow_log_metric(mlflow, f"{prefix}/parity/success", 1.0 if parity.success else 0.0)

    if parity.passed is not None:
        safe_mlflow_log_metric(mlflow, f"{prefix}/parity/passed", 1.0 if parity.passed else 0.0)

    for metric_name in [
        "image_count",
        "passed_image_count",
        "failed_image_count",
        "total_reference_detections",
        "total_artifact_detections",
        "total_matched_detections",
        "mean_matched_iou",
        "min_matched_iou",
        "mean_box_abs_diff",
        "max_box_abs_diff",
        "mean_conf_abs_diff",
        "max_conf_abs_diff",
    ]:
        safe_mlflow_log_metric(mlflow, f"{prefix}/parity/{metric_name}", getattr(parity, metric_name))

    safe_mlflow_log_param(mlflow, f"{prefix}/parity/reason", parity.reason)

    if parity.error:
        safe_mlflow_log_param(mlflow, f"{prefix}/parity/error", parity.error)


def log_artifact_report_to_mlflow(
    *,
    mlflow: Any | None,
    report: ArtifactReport,
) -> None:
    """Log one artifact report to MLflow."""

    if mlflow is None:
        return

    prefix = f"artifacts/{report.label}"

    safe_mlflow_log_param(mlflow, f"{prefix}/type", report.artifact_type)
    safe_mlflow_log_param(mlflow, f"{prefix}/path", report.path)

    if report.validation is not None:
        log_evaluation_metrics_to_mlflow(
            mlflow=mlflow,
            evaluation=report.validation,
            prefix=f"{prefix}/validation",
        )

    log_ort_checks_to_mlflow(
        mlflow=mlflow,
        checks=report.ort_backend_checks,
        prefix=prefix,
    )

    log_parity_to_mlflow(
        mlflow=mlflow,
        parity=report.parity,
        prefix=prefix,
    )


def log_summary_to_mlflow(
    *,
    mlflow: Any | None,
    summary: EvaluationSummary,
    output_dir: Path,
    log_artifacts: bool,
) -> None:
    """Log final evaluation summary and local files to MLflow."""

    if mlflow is None:
        return

    safe_mlflow_log_metric(mlflow, "summary/duration_seconds", summary.duration_seconds)
    safe_mlflow_log_metric(mlflow, "summary/artifact_count", len(summary.artifacts))
    safe_mlflow_log_metric(mlflow, "summary/has_failure", 1.0 if has_failure(summary) else 0.0)

    if log_artifacts:
        try:
            mlflow.log_artifacts(str(output_dir), artifact_path="evaluation_outputs")
        except Exception as error:
            LOGGER.warning("Failed to log evaluation output directory to MLflow: %s", error)


def end_mlflow_run(mlflow: Any | None, status: str) -> None:
    """End active MLflow run if logging is enabled."""

    if mlflow is None:
        return

    try:
        mlflow.end_run(status=status)
    except Exception as error:
        LOGGER.warning("Failed to end MLflow run cleanly: %s", error)


def run(args: argparse.Namespace) -> int:
    """Run evaluation."""

    start_time = time.time()
    mlflow = None
    mlflow_run = None
    mlflow_run_id: str | None = None
    mlflow_status = "FINISHED"

    args.weights = args.weights.resolve()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    args.ort_tensorrt_cache_dir = args.ort_tensorrt_cache_dir.resolve()
    args.onnx = [path.resolve() for path in args.onnx]
    args.engine = [path.resolve() for path in args.engine]

    validate_required_paths(args)

    output_dir = make_output_dir(
        base_output=args.output,
        name=args.name,
        exist_ok=args.exist_ok,
    )

    dataset_info = validate_dataset_yaml(args.data, args.split)

    LOGGER.info("Evaluation configuration:")
    LOGGER.info("  weights: %s", args.weights)
    LOGGER.info("  data: %s", args.data)
    LOGGER.info("  split: %s", args.split)
    LOGGER.info("  output: %s", output_dir)
    LOGGER.info("  ONNX artifacts: %s", args.onnx)
    LOGGER.info("  TensorRT engines: %s", args.engine)
    LOGGER.info("  evaluate PyTorch: %s", args.eval_pytorch)
    LOGGER.info("  evaluate artifacts: %s", args.eval_artifacts)
    LOGGER.info("  parity: %s", args.parity)
    LOGGER.info("  ORT backend checks: %s", requested_ort_backends(args))
    LOGGER.info("  MLflow enabled: %s", args.mlflow)

    if args.dry_run:
        LOGGER.info("Dry run passed. No evaluation started and no MLflow run created.")
        return 0

    try:
        import ultralytics  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Install dependencies with:\n\n"
            "    pip install -r requirements.txt"
        ) from error

    try:
        mlflow, mlflow_run = start_mlflow_run(args)
        mlflow_run_id = mlflow_run.info.run_id if mlflow_run is not None else None

        log_initial_mlflow_params(
            mlflow=mlflow,
            args=args,
            dataset_info=dataset_info,
            output_dir=output_dir,
        )

        pytorch_evaluation: ModelEvaluation | None = None

        if args.eval_pytorch:
            pytorch_evaluation = run_ultralytics_validation(
                label="pytorch_best",
                artifact_type="pytorch",
                model_path=args.weights,
                data_yaml=args.data,
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                output_dir=output_dir,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                half=args.half,
                plots=args.plots,
                save_json=args.save_json,
                verbose=args.verbose,
                class_names=dataset_info.class_names,
            )

            log_model_evaluation(pytorch_evaluation)
            write_json(output_dir / "pytorch_metrics.json", asdict(pytorch_evaluation))
            log_evaluation_metrics_to_mlflow(
                mlflow=mlflow,
                evaluation=pytorch_evaluation,
                prefix="pytorch/validation",
            )

            if not pytorch_evaluation.success and not args.continue_on_error:
                raise RuntimeError(f"PyTorch evaluation failed: {pytorch_evaluation.error}")

        artifacts: list[ArtifactReport] = []

        for onnx_path in args.onnx:
            label = artifact_label(onnx_path, "onnx")

            try:
                report = evaluate_artifact(
                    artifact_path=onnx_path,
                    artifact_type="onnx",
                    label=label,
                    data_yaml=args.data,
                    dataset_info=dataset_info,
                    output_dir=output_dir,
                    args=args,
                )
            except Exception as error:
                if not args.continue_on_error:
                    raise

                report = ArtifactReport(
                    label=label,
                    artifact_type="onnx",
                    path=str(onnx_path),
                    validation=None,
                    ort_backend_checks=[],
                    parity=failed_parity_summary(
                        label=label,
                        artifact_type="onnx",
                        artifact_path=onnx_path,
                        error=error,
                        thresholds=parity_thresholds(args),
                    ),
                )

            artifacts.append(report)
            log_artifact_report_to_mlflow(mlflow=mlflow, report=report)

        for engine_path in args.engine:
            label = artifact_label(engine_path, "engine")

            try:
                report = evaluate_artifact(
                    artifact_path=engine_path,
                    artifact_type="engine",
                    label=label,
                    data_yaml=args.data,
                    dataset_info=dataset_info,
                    output_dir=output_dir,
                    args=args,
                )
            except Exception as error:
                if not args.continue_on_error:
                    raise

                report = ArtifactReport(
                    label=label,
                    artifact_type="engine",
                    path=str(engine_path),
                    validation=None,
                    ort_backend_checks=[],
                    parity=failed_parity_summary(
                        label=label,
                        artifact_type="engine",
                        artifact_path=engine_path,
                        error=error,
                        thresholds=parity_thresholds(args),
                    ),
                )

            artifacts.append(report)
            log_artifact_report_to_mlflow(mlflow=mlflow, report=report)

        summary = EvaluationSummary(
            created_unix_time=time.time(),
            duration_seconds=time.time() - start_time,
            output_dir=str(output_dir),
            weights=str(args.weights),
            data_yaml=str(args.data),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            mlflow_enabled=args.mlflow,
            mlflow_run_id=mlflow_run_id,
            dataset=dataset_info,
            pytorch_evaluation=pytorch_evaluation,
            artifacts=artifacts,
            args=args_to_dict(args),
        )

        write_json(output_dir / "evaluation_summary.json", asdict(summary))
        write_json(output_dir / "artifact_reports.json", [asdict(item) for item in artifacts])

        log_summary_to_mlflow(
            mlflow=mlflow,
            summary=summary,
            output_dir=output_dir,
            log_artifacts=args.mlflow_log_artifacts,
        )

        LOGGER.info("Evaluation completed.")
        LOGGER.info("Output directory: %s", output_dir)

        if pytorch_evaluation is not None and pytorch_evaluation.metrics is not None:
            LOGGER.info("PyTorch mAP@50: %s", pytorch_evaluation.metrics.get("map50"))
            LOGGER.info("PyTorch mAP@75: %s", pytorch_evaluation.metrics.get("map75"))
            LOGGER.info("PyTorch mAP@50-95: %s", pytorch_evaluation.metrics.get("map50_95"))

        failed = has_failure(summary)

        if failed:
            LOGGER.warning("One or more evaluations/checks/parity tests failed.")
            mlflow_status = "FAILED" if args.strict else "FINISHED"

            if args.strict:
                return 2

        return 0
    except Exception:
        mlflow_status = "FAILED"
        raise
    finally:
        end_mlflow_run(mlflow, mlflow_status)


def main() -> None:
    """Command-line entrypoint."""

    setup_logging()
    args = parse_args()

    try:
        exit_code = run(args)
    except Exception as error:
        LOGGER.error("evaluate.py failed: %s", error)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
