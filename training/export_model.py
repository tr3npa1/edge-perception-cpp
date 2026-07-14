"""
Export YOLO26M PyTorch checkpoints to ONNX and TensorRT deployment artifacts.

This script is the export entrypoint for edge-perception-cpp.

Expected PyTorch model input:
    runs/tain/yolo26m_bdd100k/weights/best.pt

Expected dataset input for INT8 calibration:
    data/processed/bdd100k_yolo/bdd100k.yaml

Responsibilities:
- export base ONNX FP32
- export base ONNX FP16
- export ONNX INT8/PTQ QDQ artifacts for ONNX Runtime CPU/CUDA validation
- record ONNX Runtime CPU/CUDA/TensorRT-EP backend metadata for ONNX artifacts
- document that ONNX INT8 QDQ + ORT TensorRT EP is skipped for this YOLO26M graph
- export native TensorRT FP32 engine
- export native TensorRT FP16 engine
- export native TensorRT INT8 engine with calibration data
- write per-artifact metadata JSON
- write global export summary JSON
- inspect ONNX graph inputs/outputs
- inspect ONNX Runtime session inputs/outputs when requested
- optionally smoke-load TensorRT engine artifacts

This script does not train.
This script does not evaluate mAP.
This script does not run PyTorch-vs-ONNX parity.
This script does not benchmark deployment latency.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger("export_yolo26m_model")

DEFAULT_WEIGHTS = Path("runs/train/yolo26m_bdd100k/weights/best.pt")
DEFAULT_DATA_YAML = Path("data/processed/bdd100k_yolo/bdd100k.yaml")
DEFAULT_ONNX_DIR = Path("models/onnx")
DEFAULT_ENGINE_DIR = Path("models/engine")
DEFAULT_ORT_TRT_CACHE_DIR = Path("models/ort_trt_cache")
DEFAULT_BASENAME = "yolo26m_bdd100k"

KAGGLE_PATH_MARKER = "/kaggle/input"

ORT_TRT_INT8_QDQ_SKIP_REASON = (
    "skipped_unsupported_combo: ONNX INT8 QDQ artifacts are intended for ONNX "
    "Runtime CPU/CUDA checks in this project. ORT TensorRT EP INT8 is skipped "
    "for this YOLO26M graph; native TensorRT .engine artifacts are the supported "
    "TensorRT INT8 deployment path."
)

ONNX_VARIANTS = ("onnx_fp32", "onnx_fp16", "onnx_int8")
ENGINE_VARIANTS = ("engine_fp32", "engine_fp16", "engine_int8")
ALL_VARIANTS = ONNX_VARIANTS + ENGINE_VARIANTS


@dataclass(frozen=True)
class DatasetInfo:
    """Dataset information used for export and INT8 calibration."""

    yaml_path: str
    dataset_root: str
    class_count: int
    class_names: list[str]
    train: str | list[str] | None
    val: str | list[str] | None
    test: str | list[str] | None


@dataclass(frozen=True)
class ExportArtifact:
    """Metadata for one exported deployment artifact."""

    variant: str
    export_format: str
    precision: str
    quantize: int | None
    success: bool
    output_path: str | None
    file_size_bytes: int | None
    export_seconds: float | None
    export_kwargs: dict[str, Any]
    error: str | None
    onnx_info: dict[str, Any] | None
    onnxruntime_info: dict[str, Any] | None
    backend_profiles: dict[str, Any] | None
    engine_info: dict[str, Any] | None


@dataclass(frozen=True)
class ExportSummary:
    """Top-level export summary."""

    created_unix_time: float
    duration_seconds: float
    weights: str
    data_yaml: str | None
    onnx_dir: str
    engine_dir: str
    ort_tensorrt_cache_dir: str
    basename: str
    requested_variants: list[str]
    successful_variants: list[str]
    failed_variants: list[str]
    exports: list[ExportArtifact]
    dataset: DatasetInfo | None
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
        description="Export YOLO26M to ONNX and TensorRT deployment artifacts."
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
        help="Prepared BDD100K YAML used for INT8 calibration.",
    )

    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=DEFAULT_ONNX_DIR,
        help="Output directory for ONNX files.",
    )

    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=DEFAULT_ENGINE_DIR,
        help="Output directory for native TensorRT engine files.",
    )

    parser.add_argument(
        "--ort-tensorrt-cache-dir",
        type=Path,
        default=DEFAULT_ORT_TRT_CACHE_DIR,
        help="Engine-cache directory planned for ONNX Runtime TensorRT EP.",
    )

    parser.add_argument(
        "--basename",
        type=str,
        default=DEFAULT_BASENAME,
        help="Base filename used for exported artifacts.",
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        default=["onnx_fp32"],
        choices=(
            "onnx_fp32",
            "onnx_fp16",
            "onnx_int8",
            "engine_fp32",
            "engine_fp16",
            "engine_int8",
            "onnx_all",
            "engine_all",
            "all",
        ),
        help=(
            "Artifacts to export. Default is onnx_fp32. "
            "Use all for ONNX + native TensorRT engine variants."
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export batch size. Use 1 for deployment.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Export device, for example '0' or 'cpu'. TensorRT export requires NVIDIA GPU support.",
    )

    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export with dynamic input shapes when supported.",
    )

    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simplify ONNX graph when supported.",
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset. If omitted, Ultralytics chooses the default.",
    )

    parser.add_argument(
        "--nms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Embed NMS into exported model when supported. Default is false "
            "because src/postprocess.cpp owns confidence filtering and NMS."
        ),
    )

    parser.add_argument(
        "--end2end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Export YOLO26 end-to-end final-detection graph when supported. "
            "Default is false because this project wants raw outputs for C++ postprocess.cpp."
        ),
    )

    parser.add_argument(
        "--workspace",
        type=float,
        default=None,
        help="TensorRT workspace size in GiB for engine export.",
    )

    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset used for INT8/PTQ calibration.",
    )

    parser.add_argument(
        "--calibration-max-images",
        type=int,
        default=0,
        help=(
            "Maximum calibration images for custom ONNX INT8 fallback. "
            "Use 0 for no cap. This does not limit Ultralytics native engine INT8 calibration."
        ),
    )

    parser.add_argument(
        "--engine-int8-calibration-split",
        choices=("train", "val"),
        default="train",
        help=(
            "Dataset split used for native TensorRT INT8 calibration. "
            "Ultralytics calibrates TensorRT INT8 from the YAML 'val' field, so this script "
            "writes a temporary calibration YAML with val=<this split>. Default 'train' "
            "keeps native TensorRT INT8 calibration consistent with the ONNX INT8 fallback."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported artifacts and metadata.",
    )

    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue exporting remaining variants if one variant fails.",
    )

    parser.add_argument(
        "--skip-onnx-check",
        action="store_true",
        help="Skip ONNX checker and ONNX Runtime inspection.",
    )

    parser.add_argument(
        "--check-ort-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check ONNX Runtime CPUExecutionProvider session for ONNX exports.",
    )

    parser.add_argument(
        "--check-ort-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Check ONNX Runtime CUDAExecutionProvider session for ONNX exports.",
    )

    parser.add_argument(
        "--check-ort-tensorrt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Check ONNX Runtime TensorRTExecutionProvider session for ONNX exports.",
    )

    parser.add_argument(
        "--ort-run-dummy",
        action="store_true",
        help=(
            "Run one dummy ONNX Runtime inference after ONNX export. Disabled by "
            "default because FP16/INT8 CPU dummy inference may be provider-limited."
        ),
    )

    parser.add_argument(
        "--strict-checks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail the export if ONNX/ORT/TensorRT smoke checks fail.",
    )

    parser.add_argument(
        "--engine-smoke-load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Smoke-load TensorRT engine through Ultralytics after export.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without exporting.",
    )

    return parser.parse_args()


def args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Convert parsed arguments into JSON-serializable values."""

    output: dict[str, Any] = {}

    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value

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
    """Reject Kaggle-only YAML paths before export/calibration starts."""

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
                    "Run data/dataset.py and export with "
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


def validate_dataset_yaml(data_yaml: Path) -> DatasetInfo:
    """Validate the prepared dataset YAML used for INT8 calibration."""

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

    return DatasetInfo(
        yaml_path=str(data_yaml),
        dataset_root=str(dataset_root),
        class_count=len(class_names),
        class_names=class_names,
        train=yaml_payload.get("train"),
        val=yaml_payload.get("val"),
        test=yaml_payload.get("test"),
    )


def expand_variants(raw_variants: list[str]) -> list[str]:
    """Expand variant aliases into a stable ordered list."""

    expanded: list[str] = []

    for item in raw_variants:
        if item == "all":
            expanded.extend(ALL_VARIANTS)
        elif item == "onnx_all":
            expanded.extend(ONNX_VARIANTS)
        elif item == "engine_all":
            expanded.extend(ENGINE_VARIANTS)
        else:
            expanded.append(item)

    output: list[str] = []

    for variant in ALL_VARIANTS:
        if variant in expanded and variant not in output:
            output.append(variant)

    return output


def variant_format(variant: str) -> str:
    """Return Ultralytics export format for a variant."""

    if variant.startswith("onnx_"):
        return "onnx"

    if variant.startswith("engine_"):
        return "engine"

    raise ValueError(f"Unsupported variant: {variant}")


def variant_precision(variant: str) -> str:
    """Return human-readable precision for a variant."""

    if variant.endswith("_fp32"):
        return "FP32"

    if variant.endswith("_fp16"):
        return "FP16"

    if variant.endswith("_int8"):
        return "INT8/PTQ"

    raise ValueError(f"Unsupported variant precision: {variant}")


def variant_quantize(variant: str) -> int | None:
    """Return Ultralytics quantize value for a variant."""

    if variant.endswith("_fp32"):
        return None

    if variant.endswith("_fp16"):
        return 16

    if variant.endswith("_int8"):
        return 8

    raise ValueError(f"Unsupported variant quantization: {variant}")


def variant_output_path(args: argparse.Namespace, variant: str) -> Path:
    """Return final artifact path for one variant."""

    if variant.startswith("onnx_"):
        suffix = variant.replace("onnx_", "")
        return (args.onnx_dir / f"{args.basename}_{suffix}.onnx").resolve()

    if variant.startswith("engine_"):
        suffix = variant.replace("engine_", "")
        return (args.engine_dir / f"{args.basename}_{suffix}.engine").resolve()

    raise ValueError(f"Unsupported variant: {variant}")


def variant_metadata_path(args: argparse.Namespace, variant: str) -> Path:
    """Return metadata path for one variant."""

    if variant.startswith("onnx_"):
        return args.onnx_dir / f"{args.basename}_{variant}_metadata.json"

    if variant.startswith("engine_"):
        return args.engine_dir / f"{args.basename}_{variant}_metadata.json"

    raise ValueError(f"Unsupported variant: {variant}")


def validate_export_args(args: argparse.Namespace, variants: list[str]) -> None:
    """Validate export options."""

    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive.")

    if args.batch <= 0:
        raise ValueError("--batch must be positive.")

    if not (0.0 < args.calibration_fraction <= 1.0):
        raise ValueError("--calibration-fraction must be in the range (0, 1].")

    if any(variant.startswith("engine_") for variant in variants):
        if args.device.lower() == "cpu":
            raise ValueError("TensorRT engine export requires an NVIDIA GPU device, not CPU.")

    if any(variant.endswith("_int8") for variant in variants):
        if args.data is None:
            raise ValueError("INT8 export requires --data for calibration.")

        if not args.data.is_file():
            raise FileNotFoundError(f"Calibration dataset YAML not found: {args.data}")

    if args.end2end and not args.nms:
        LOGGER.warning(
            "end2end=True with nms=False may produce model outputs that differ from "
            "the raw-output C++ postprocessing path."
        )

    if args.calibration_fraction < 0.25 and any(
        variant.endswith("_int8") for variant in variants
    ):
        LOGGER.warning(
            "INT8 calibration fraction is below 0.25. This is useful for quick tests "
            "but may hurt quantized accuracy."
        )


def validate_required_paths(args: argparse.Namespace, variants: list[str]) -> DatasetInfo | None:
    """Validate checkpoint and dataset paths."""

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    dataset_info: DatasetInfo | None = None

    requires_data = any(variant.endswith("_int8") for variant in variants)
    optional_data_exists = args.data is not None and args.data.is_file()

    if requires_data or optional_data_exists:
        dataset_info = validate_dataset_yaml(args.data)

    return dataset_info


def ensure_targets_available(args: argparse.Namespace, variants: list[str]) -> None:
    """Fail early if output targets already exist and overwrite is disabled."""

    if args.overwrite:
        return

    existing_paths: list[Path] = []

    for variant in variants:
        output_path = variant_output_path(args, variant)
        metadata_path = variant_metadata_path(args, variant)

        if output_path.exists():
            existing_paths.append(output_path)

        if metadata_path.exists():
            existing_paths.append(metadata_path)

    summary_path = export_summary_path(args)

    if summary_path.exists():
        existing_paths.append(summary_path)

    if existing_paths:
        formatted = "\n".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Export outputs already exist:\n"
            f"{formatted}\n\n"
            "Use --overwrite to replace existing exported artifacts."
        )


def make_export_kwargs(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    """Build Ultralytics model.export keyword arguments for one variant."""

    export_format = variant_format(variant)
    quantize = variant_quantize(variant)

    export_kwargs: dict[str, Any] = {
        "format": export_format,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "dynamic": args.dynamic,
    }

    if export_format == "onnx":
        export_kwargs["simplify"] = args.simplify

        if args.opset is not None:
            export_kwargs["opset"] = args.opset

    if export_format == "engine" and args.workspace is not None:
        export_kwargs["workspace"] = args.workspace

    if args.nms:
        export_kwargs["nms"] = True

    if args.end2end:
        export_kwargs["end2end"] = True

    # Ultralytics 8.x export paths generally use half=True / int8=True.
    # Some YOLO26 examples mention quantize=16/8, but that is not accepted by
    # the installed Ultralytics CLI/parser on the Yotta instances we tested.
    if quantize == 16:
        export_kwargs["half"] = True

    if quantize == 8:
        export_kwargs["int8"] = True
        export_kwargs["data"] = str(args.data.resolve())
        export_kwargs["fraction"] = args.calibration_fraction

    return export_kwargs


def fallback_export_kwargs(export_kwargs: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    """
    Build fallback export kwargs for older Ultralytics versions.

    This keeps compatibility with small Ultralytics API differences. The
    primary path uses half=True / int8=True because quantize=16/8 was rejected
    by the tested Ultralytics stack. The fallback only runs if the first export
    attempt fails.
    """

    fallbacks: list[dict[str, Any]] = []

    if variant.endswith("_fp16"):
        fallback = dict(export_kwargs)
        fallback.pop("quantize", None)
        fallback["half"] = True
        fallbacks.append(fallback)

    if variant.endswith("_int8"):
        fallback = dict(export_kwargs)
        fallback.pop("quantize", None)
        fallback["int8"] = True
        fallbacks.append(fallback)

    stripped_end2end = dict(export_kwargs)
    stripped_end2end.pop("end2end", None)

    if stripped_end2end != export_kwargs:
        fallbacks.append(stripped_end2end)

    stripped_nms_end2end = dict(export_kwargs)
    stripped_nms_end2end.pop("nms", None)
    stripped_nms_end2end.pop("end2end", None)

    if stripped_nms_end2end != export_kwargs:
        fallbacks.append(stripped_nms_end2end)

    return fallbacks


def export_with_fallbacks(model: Any, export_kwargs: dict[str, Any], variant: str) -> Any:
    """Run model.export with compatibility fallbacks."""

    errors: list[str] = []

    try:
        return model.export(**export_kwargs)
    except Exception as error:
        errors.append(f"primary kwargs={export_kwargs}: {error}")

    for fallback_kwargs in fallback_export_kwargs(export_kwargs, variant):
        LOGGER.warning("Retrying export for %s with fallback kwargs: %s", variant, fallback_kwargs)

        try:
            return model.export(**fallback_kwargs)
        except Exception as error:
            errors.append(f"fallback kwargs={fallback_kwargs}: {error}")

    raise RuntimeError(
        f"Export failed for variant '{variant}'. Errors:\n" + "\n".join(errors)
    )


def find_recent_artifact(
    *,
    start_dirs: list[Path],
    suffix: str,
    after_unix_time: float,
) -> Path | None:
    """Find the most recently modified artifact created after a timestamp."""

    candidates: list[Path] = []

    for directory in start_dirs:
        if not directory.exists():
            continue

        for path in directory.rglob(f"*{suffix}"):
            if path.is_file() and path.stat().st_mtime >= after_unix_time:
                candidates.append(path.resolve())

    if not candidates:
        return None

    return max(candidates, key=lambda item: item.stat().st_mtime)


def path_from_export_result(export_result: Any, expected_suffix: str) -> Path | None:
    """Try to get an artifact path directly from Ultralytics export output."""

    possible_values: list[Any]

    if isinstance(export_result, (list, tuple)):
        possible_values = list(export_result)
    else:
        possible_values = [export_result]

    for value in possible_values:
        if value is None:
            continue

        candidate = Path(str(value)).resolve()

        if candidate.is_file() and candidate.suffix.lower() == expected_suffix:
            return candidate

    return None


def resolve_exported_path(
    *,
    export_result: Any,
    weights: Path,
    expected_suffix: str,
    export_started_at: float,
) -> Path:
    """Resolve the artifact path returned by Ultralytics export."""

    direct_path = path_from_export_result(export_result, expected_suffix)

    if direct_path is not None:
        return direct_path

    search_dirs = [
        weights.parent.resolve(),
        weights.parent.parent.resolve(),
        Path.cwd().resolve(),
    ]

    found_path = find_recent_artifact(
        start_dirs=search_dirs,
        suffix=expected_suffix,
        after_unix_time=export_started_at,
    )

    if found_path is None:
        raise FileNotFoundError(
            f"Ultralytics export completed, but no {expected_suffix} artifact could be located."
        )

    return found_path


def copy_export_to_target(source_path: Path, target_path: Path, overwrite: bool) -> None:
    """Copy exported artifact to the final project artifact path."""

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Export target already exists: {target_path}\n"
            "Use --overwrite to replace it."
        )

    if source_path.resolve() == target_path.resolve():
        return

    shutil.copy2(source_path, target_path)


def onnx_dim_to_json(dim: Any) -> int | str | None:
    """Convert an ONNX dimension object to a JSON-safe value."""

    if getattr(dim, "dim_value", 0):
        return int(dim.dim_value)

    if getattr(dim, "dim_param", ""):
        return str(dim.dim_param)

    return None


def tensor_value_info_to_dict(value_info: Any) -> dict[str, Any]:
    """Convert ONNX ValueInfoProto input/output metadata to a dictionary."""

    tensor_type = value_info.type.tensor_type
    shape = tensor_type.shape

    return {
        "name": value_info.name,
        "elem_type": int(tensor_type.elem_type),
        "shape": [onnx_dim_to_json(dim) for dim in shape.dim],
    }


def inspect_onnx_graph(onnx_path: Path) -> dict[str, Any]:
    """Run ONNX checker and inspect graph input/output metadata."""

    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    opsets = [
        {
            "domain": item.domain,
            "version": int(item.version),
        }
        for item in model.opset_import
    ]

    return {
        "ir_version": int(model.ir_version),
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "opsets": opsets,
        "inputs": [tensor_value_info_to_dict(item) for item in model.graph.input],
        "outputs": [tensor_value_info_to_dict(item) for item in model.graph.output],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
    }


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


def build_provider_request(
    *,
    provider_name: str,
    cache_dir: Path | None = None,
) -> list[Any]:
    """Build an ONNX Runtime provider list for one backend profile."""

    if provider_name == "CPUExecutionProvider":
        return ["CPUExecutionProvider"]

    if provider_name == "CUDAExecutionProvider":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if provider_name == "TensorrtExecutionProvider":
        provider_options = {
            "trt_engine_cache_enable": "True",
        }

        if cache_dir is not None:
            provider_options["trt_engine_cache_path"] = str(cache_dir)

        return [
            ("TensorrtExecutionProvider", provider_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    raise ValueError(f"Unsupported ONNX Runtime provider: {provider_name}")


def is_int8_onnx_artifact(onnx_path: Path) -> bool:
    """Return True when an ONNX artifact name represents the INT8 QDQ export."""

    return "int8" in onnx_path.stem.lower()


def inspect_ort_session(
    *,
    onnx_path: Path,
    provider_request: list[Any],
    run_dummy: bool,
    batch: int,
    imgsz: int,
) -> dict[str, Any]:
    """Inspect one ONNX Runtime session and optionally run dummy inference."""

    import numpy as np
    import onnxruntime as ort

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

    dummy_result: dict[str, Any] | None = None

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

        dummy_result = {
            "input_name": first_input.name,
            "input_shape": dummy_shape,
            "input_dtype": str(dummy_input.dtype),
            "output_shapes": [list(array.shape) for array in output_arrays],
            "output_dtypes": [str(array.dtype) for array in output_arrays],
        }

    return {
        "actual_providers": session.get_providers(),
        "inputs": inputs,
        "outputs": outputs,
        "dummy_inference": dummy_result,
    }


def build_backend_profiles(
    *,
    onnx_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Build backend metadata for ONNX Runtime deployment paths.

    The same FP32/FP16 ONNX model can be loaded by:
        - CPUExecutionProvider
        - CUDAExecutionProvider
        - TensorrtExecutionProvider

    ONNX INT8 QDQ artifacts are intentionally not checked with ORT TensorRT EP
    in this project. Native TensorRT INT8 .engine artifacts are the supported
    TensorRT INT8 path.
    """

    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    is_int8_qdq = is_int8_onnx_artifact(onnx_path)

    profiles: dict[str, Any] = {
        "ort_cpu": {
            "artifact": str(onnx_path),
            "provider": "CPUExecutionProvider",
            "available": "CPUExecutionProvider" in available_providers,
            "checked": False,
            "check_passed": None,
            "check_error": None,
        },
        "ort_cuda": {
            "artifact": str(onnx_path),
            "provider": "CUDAExecutionProvider",
            "available": "CUDAExecutionProvider" in available_providers,
            "checked": False,
            "check_passed": None,
            "check_error": None,
        },
        "ort_tensorrt": {
            "artifact": str(onnx_path),
            "provider": "TensorrtExecutionProvider",
            "available": "TensorrtExecutionProvider" in available_providers,
            "checked": False,
            "check_passed": None,
            "check_error": None,
            "engine_cache_enable": True,
            "engine_cache_path": str(args.ort_tensorrt_cache_dir),
            "skipped": False,
            "skip_reason": None,
            "note": (
                "ONNX Runtime TensorRT EP consumes an ONNX file and may build/cache "
                "TensorRT engines internally at runtime. This is separate from native "
                ".engine export."
            ),
        },
    }

    requested_checks = {
        "ort_cpu": args.check_ort_cpu,
        "ort_cuda": args.check_ort_cuda,
        "ort_tensorrt": args.check_ort_tensorrt,
    }

    provider_names = {
        "ort_cpu": "CPUExecutionProvider",
        "ort_cuda": "CUDAExecutionProvider",
        "ort_tensorrt": "TensorrtExecutionProvider",
    }

    for profile_name, should_check in requested_checks.items():
        if not should_check:
            continue

        if profile_name == "ort_tensorrt" and is_int8_qdq:
            profiles[profile_name]["checked"] = True
            profiles[profile_name]["check_passed"] = True
            profiles[profile_name]["skipped"] = True
            profiles[profile_name]["skip_reason"] = ORT_TRT_INT8_QDQ_SKIP_REASON
            profiles[profile_name]["note"] = (
                "Skipped by design. Use native TensorRT INT8 .engine artifacts for "
                "TensorRT INT8 deployment."
            )
            continue

        provider_name = provider_names[profile_name]
        profiles[profile_name]["checked"] = True

        if provider_name not in available_providers:
            profiles[profile_name]["check_passed"] = False
            profiles[profile_name]["check_error"] = (
                f"{provider_name} is not available in this ONNX Runtime installation."
            )
            continue

        try:
            cache_dir = (
                args.ort_tensorrt_cache_dir
                if provider_name == "TensorrtExecutionProvider"
                else None
            )

            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)

            provider_request = build_provider_request(
                provider_name=provider_name,
                cache_dir=cache_dir,
            )

            session_info = inspect_ort_session(
                onnx_path=onnx_path,
                provider_request=provider_request,
                run_dummy=args.ort_run_dummy,
                batch=args.batch,
                imgsz=args.imgsz,
            )

            actual_providers = session_info.get("actual_providers", [])
            if provider_name not in actual_providers:
                raise RuntimeError(
                    f"{provider_name} was requested but ONNX Runtime fell back to "
                    f"{actual_providers}."
                )

            profiles[profile_name]["check_passed"] = True
            profiles[profile_name]["session"] = session_info
        except Exception as error:
            profiles[profile_name]["check_passed"] = False
            profiles[profile_name]["check_error"] = str(error)

            if args.strict_checks:
                raise

    return profiles


def inspect_onnx_artifact(
    *,
    onnx_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Inspect ONNX graph and planned ONNX Runtime backend profiles."""

    if args.skip_onnx_check:
        return {"skipped": True}, None, None

    try:
        onnx_info = inspect_onnx_graph(onnx_path)
        backend_profiles = build_backend_profiles(
            onnx_path=onnx_path,
            args=args,
        )

        onnxruntime_info = {
            "available_profiles": backend_profiles,
            "note": (
                "Use FP32/FP16 ONNX artifacts with ORT CPU, ORT CUDA, or ORT TensorRT EP. "
                "Use ONNX INT8 QDQ artifacts with ORT CPU/CUDA. For TensorRT INT8, use "
                "the native .engine artifacts exported by this script."
            ),
        }

        return onnx_info, onnxruntime_info, backend_profiles
    except Exception as error:
        if args.strict_checks:
            raise

        return (
            {"inspection_failed": True, "error": str(error)},
            {"inspection_failed": True, "error": str(error)},
            None,
        )



def split_entry_to_list(split_entry: Any) -> list[str]:
    """Normalize one YOLO split entry to a list of path strings."""

    if split_entry is None:
        return []

    if isinstance(split_entry, str):
        return [split_entry]

    if isinstance(split_entry, list):
        return [str(item) for item in split_entry]

    return [str(split_entry)]


def resolve_split_paths_from_yaml(data_yaml: Path, split: str = "train") -> list[Path]:
    """Resolve image roots for one YOLO dataset split."""

    payload = read_yaml(data_yaml)
    dataset_root = resolve_dataset_root(data_yaml.resolve(), payload)
    paths: list[Path] = []

    for item in split_entry_to_list(payload.get(split)):
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        if candidate.exists():
            paths.append(candidate.resolve())

    return paths


def collect_calibration_images(
    *,
    data_yaml: Path,
    fraction: float,
    max_images: int,
) -> list[Path]:
    """Collect deterministic calibration images for ONNX static INT8 quantization."""

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    roots = resolve_split_paths_from_yaml(data_yaml, split="train")

    if not roots:
        roots = resolve_split_paths_from_yaml(data_yaml, split="val")

    images: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in image_extensions:
            images.append(root.resolve())
            continue
        if root.is_dir():
            images.extend(
                path.resolve()
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in image_extensions
            )

    images = sorted(set(images))

    if not images:
        raise FileNotFoundError(
            f"No calibration images found from dataset YAML: {data_yaml}"
        )

    keep_count = max(1, int(len(images) * fraction))
    images = images[:keep_count]

    if max_images > 0:
        images = images[:max_images]

    return images


class YoloCalibrationDataReader:
    """ONNX Runtime static quantization calibration reader for YOLO image tensors."""

    def __init__(self, image_paths: list[Path], input_name: str, imgsz: int):
        self.image_paths = image_paths
        self.input_name = input_name
        self.imgsz = imgsz
        self.index = 0

    def get_next(self) -> dict[str, Any] | None:
        """Return the next calibration sample."""

        if self.index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        import numpy as np
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.imgsz, self.imgsz))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))[None, ...]

        return {self.input_name: array.astype(np.float32)}


def get_onnx_first_input_name(onnx_path: Path) -> str:
    """Return first ONNX graph input name."""

    import onnx

    model = onnx.load(str(onnx_path))
    if not model.graph.input:
        raise RuntimeError(f"ONNX model has no graph inputs: {onnx_path}")
    return model.graph.input[0].name


def export_onnx_int8_via_ort(
    *,
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
    args: argparse.Namespace,
) -> None:
    """
    Create an all-Conv ONNX INT8 QDQ artifact with ONNX Runtime static quantization.

    This mode quantizes every Conv node that ONNX Runtime can quantize.

    Still intentionally not quantized:
    - non-Conv ops such as TopK/Gather/Reshape/Concat/etc.
    - final detection/output formatting logic if it is implemented through non-Conv ops.
    - bias tensors, via QuantizeBias=False.

    This is not full-graph INT8. It is all-Conv INT8.
    """

    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    LOGGER.info("Creating ALL-CONV ONNX INT8 using ONNX Runtime static QDQ quantization:")
    LOGGER.info("  source: %s", fp32_onnx_path)
    LOGGER.info("  target: %s", int8_onnx_path)
    LOGGER.info("  mode: all Conv nodes, no Conv exclusion list")
    LOGGER.info("  op_types_to_quantize: Conv")
    LOGGER.info("  nodes_to_exclude: []")
    LOGGER.info("  activation_type: QInt8")
    LOGGER.info("  weight_type: QInt8")
    LOGGER.info("  per_channel: True")
    LOGGER.info("  QuantizeBias: False")

    image_paths = collect_calibration_images(
        data_yaml=args.data,
        fraction=args.calibration_fraction,
        max_images=args.calibration_max_images,
    )
    LOGGER.info("  calibration images: %d", len(image_paths))

    input_name = get_onnx_first_input_name(fp32_onnx_path)
    reader = YoloCalibrationDataReader(
        image_paths=image_paths,
        input_name=input_name,
        imgsz=args.imgsz,
    )

    int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv"],
        nodes_to_exclude=[],
        per_channel=True,
        extra_options={
            "QuantizeBias": False,
        },
    )


def ensure_onnx_source_for_variant(
    *,
    weights: Path,
    args: argparse.Namespace,
    variant: str,
) -> Path:
    """Ensure an ONNX source exists for TensorRT engine building."""

    if variant.endswith("_fp16"):
        onnx_variant = "onnx_fp16"
    elif variant.endswith("_int8"):
        onnx_variant = "onnx_int8"
    else:
        onnx_variant = "onnx_fp32"

    onnx_path = variant_output_path(args, onnx_variant)
    if onnx_path.is_file():
        return onnx_path

    LOGGER.info("Required ONNX source missing for %s, exporting %s first.", variant, onnx_variant)
    artifact = export_one_variant(weights=weights, args=args, variant=onnx_variant)
    if not artifact.success or artifact.output_path is None:
        raise RuntimeError(f"Failed to create ONNX source for {variant}: {onnx_variant}")

    return Path(artifact.output_path).resolve()


def count_images_in_roots(roots: list[Path]) -> int:
    """Count image files inside one or more resolved roots."""

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    count = 0

    for root in roots:
        if root.is_file() and root.suffix.lower() in image_extensions:
            count += 1
            continue

        if root.is_dir():
            count += sum(
                1
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in image_extensions
            )

    return count


def create_engine_int8_calibration_yaml(
    *,
    args: argparse.Namespace,
) -> tuple[Path, int]:
    """
    Create a temporary YAML for native TensorRT INT8 calibration.

    Ultralytics TensorRT INT8 export reads calibration images through the YAML
    'val' split. To make calibration explicit and reproducible, this function
    writes a small project-local YAML where 'val' points to the requested
    calibration split. The default requested split is 'train', matching the
    ONNX Runtime static INT8 calibration fallback in this script.
    """

    if args.data is None:
        raise ValueError("Native TensorRT INT8 export requires --data.")

    source_yaml = args.data.resolve()
    payload = read_yaml(source_yaml)

    split = args.engine_int8_calibration_split
    split_entry = payload.get(split)

    if split_entry is None:
        raise ValueError(
            f"Dataset YAML does not define split '{split}', required for native TensorRT INT8 calibration."
        )

    dataset_root = resolve_dataset_root(source_yaml, payload)

    # Make the generated YAML location-independent. If the original YAML used a
    # relative path, copying it under models/engine would otherwise break path
    # resolution.
    payload["path"] = str(dataset_root)
    payload["val"] = split_entry

    args.engine_dir.mkdir(parents=True, exist_ok=True)
    calibration_yaml = (
        args.engine_dir
        / f"{args.basename}_engine_int8_calib_{split}_as_val.yaml"
    ).resolve()

    calibration_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    val_roots = resolve_split_paths_from_yaml(calibration_yaml, split="val")
    calibration_image_count = count_images_in_roots(val_roots)

    if calibration_image_count <= 0:
        raise FileNotFoundError(
            "No images were found for native TensorRT INT8 calibration using "
            f"split '{split}' through generated YAML: {calibration_yaml}"
        )

    LOGGER.info("Prepared native TensorRT INT8 calibration YAML:")
    LOGGER.info("  source YAML: %s", source_yaml)
    LOGGER.info("  generated YAML: %s", calibration_yaml)
    LOGGER.info("  calibration split: %s", split)
    LOGGER.info("  calibration images: %d", calibration_image_count)
    LOGGER.info(
        "  note: Ultralytics reads calibration images from YAML val; generated YAML has val=%s.",
        split,
    )

    return calibration_yaml, calibration_image_count


def export_engine_int8_via_ultralytics(
    *,
    weights: Path,
    engine_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Build native TensorRT INT8 directly through Ultralytics.

    Do not build native TensorRT INT8 from the ONNX Runtime QDQ INT8 model.
    TensorRT can reject asymmetric QDQ zero-points with errors such as
    'TensorRT only supports symmetric quantization'. Native Ultralytics TensorRT
    INT8 export calibrates TensorRT directly from images and produces the
    correct .engine artifact.
    """

    from ultralytics import YOLO

    calibration_yaml, calibration_image_count = create_engine_int8_calibration_yaml(args=args)

    export_kwargs: dict[str, Any] = {
        "format": "engine",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "dynamic": args.dynamic,
        "int8": True,
        "data": str(calibration_yaml),
        "fraction": args.calibration_fraction,
    }

    if args.workspace is not None:
        export_kwargs["workspace"] = args.workspace

    if args.simplify:
        export_kwargs["simplify"] = True

    if args.nms:
        export_kwargs["nms"] = True

    if args.end2end:
        export_kwargs["end2end"] = True

    LOGGER.info("Building native TensorRT INT8 engine through Ultralytics:")
    LOGGER.info("  weights: %s", weights)
    LOGGER.info("  target: %s", engine_path)
    LOGGER.info("  calibration YAML: %s", calibration_yaml)
    LOGGER.info("  calibration images available: %d", calibration_image_count)
    LOGGER.info("  kwargs: %s", export_kwargs)

    started_at = time.time()
    model = YOLO(str(weights))
    export_result = export_with_fallbacks(model, export_kwargs, "engine_int8")

    source_path = resolve_exported_path(
        export_result=export_result,
        weights=weights,
        expected_suffix=".engine",
        export_started_at=started_at,
    )

    copy_export_to_target(
        source_path=source_path,
        target_path=engine_path,
        overwrite=True,
    )

    return {
        "backend": "ultralytics_native_tensorrt_int8",
        "source_output": str(source_path),
        "calibration_yaml": str(calibration_yaml),
        "calibration_split": args.engine_int8_calibration_split,
        "calibration_images_available": calibration_image_count,
        "calibration_fraction": args.calibration_fraction,
        "export_kwargs": export_kwargs,
        "note": (
            "Native TensorRT INT8 is exported directly from PyTorch weights through Ultralytics. "
            "This intentionally avoids parsing the ONNX Runtime QDQ INT8 graph as TensorRT."
        ),
    }


def trtexec_command_exists() -> bool:
    """Return True when trtexec is on PATH."""

    return shutil.which("trtexec") is not None


def export_engine_via_trtexec(
    *,
    onnx_path: Path,
    engine_path: Path,
    variant: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build TensorRT engine using NVIDIA trtexec when available."""

    if not trtexec_command_exists():
        raise FileNotFoundError("trtexec not found on PATH")

    command = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--shapes=images:{args.batch}x3x{args.imgsz}x{args.imgsz}",
    ]

    if args.workspace is not None:
        command.append(f"--memPoolSize=workspace:{int(args.workspace * 1024)}")

    if variant.endswith("_fp16"):
        command.append("--fp16")
    elif variant.endswith("_int8"):
        # If the source ONNX is already QDQ INT8, TensorRT can consume the quantized graph.
        # --int8 is included for explicit INT8 engine intent when trtexec supports it.
        command.append("--int8")

    LOGGER.info("Building TensorRT engine with trtexec:")
    LOGGER.info("  %s", " ".join(command))

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if result.returncode != 0 or not engine_path.is_file():
        raise RuntimeError(
            "trtexec failed with exit code "
            f"{result.returncode}. Output:\n{result.stdout[-4000:]}"
        )

    return {
        "backend": "trtexec",
        "command": command,
        "returncode": result.returncode,
        "output_tail": result.stdout[-4000:],
    }


def export_engine_via_python_tensorrt(
    *,
    onnx_path: Path,
    engine_path: Path,
    variant: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Build TensorRT engine directly through the optional TensorRT Python API.

    TensorRT Python bindings are intentionally optional because the main
    requirements.txt must remain reliable on local Windows machines. Native
    TensorRT export should be run only in an environment where TensorRT is
    already installed and compatible with the GPU, CUDA, driver, and OS.
    """

    try:
        import tensorrt as trt  # type: ignore  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError(
            "TensorRT Python bindings are not installed in this environment. "
            "This is expected for the default local development setup. Native "
            "TensorRT engine export requires a properly configured NVIDIA TensorRT environment "
            "Either install TensorRT separately for your platform or use trtexec "
            "if it is available on PATH."
        ) from error

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    network_flags = 0
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit_batch is not None:
        network_flags |= 1 << int(explicit_batch)

    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    LOGGER.info("Building TensorRT engine with Python TensorRT API:")
    LOGGER.info("  TensorRT: %s", getattr(trt, "__version__", "unknown"))
    LOGGER.info("  source: %s", onnx_path)
    LOGGER.info("  target: %s", engine_path)
    LOGGER.info("  flags: %s", network_flags)

    parsed = parser.parse(onnx_path.read_bytes())
    if not parsed:
        errors = []
        for index in range(parser.num_errors):
            errors.append(str(parser.get_error(index)))
        raise RuntimeError("TensorRT ONNX parser failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()

    if args.workspace is not None:
        workspace_bytes = int(args.workspace * (1024**3))
        if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        elif hasattr(config, "max_workspace_size"):
            config.max_workspace_size = workspace_bytes

    requested_flags: list[str] = []

    if variant.endswith("_fp16") and hasattr(trt, "BuilderFlag") and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
        requested_flags.append("FP16")

    if variant.endswith("_int8") and hasattr(trt, "BuilderFlag") and hasattr(trt.BuilderFlag, "INT8"):
        config.set_flag(trt.BuilderFlag.INT8)
        requested_flags.append("INT8")

    serialized_engine = None
    if hasattr(builder, "build_serialized_network"):
        serialized_engine = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        if engine is not None:
            serialized_engine = engine.serialize()

    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build serialized engine.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized_engine))

    return {
        "backend": "python_tensorrt",
        "tensorrt_version": getattr(trt, "__version__", "unknown"),
        "network_flags": network_flags,
        "requested_flags": requested_flags,
    }


def export_engine_from_onnx(
    *,
    onnx_path: Path,
    engine_path: Path,
    variant: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build native TensorRT engine from ONNX with multiple backends."""

    errors: list[str] = []

    try:
        return export_engine_via_trtexec(
            onnx_path=onnx_path,
            engine_path=engine_path,
            variant=variant,
            args=args,
        )
    except Exception as error:
        errors.append(f"trtexec: {error}")
        LOGGER.warning("trtexec engine build failed for %s: %s", variant, error)

    try:
        return export_engine_via_python_tensorrt(
            onnx_path=onnx_path,
            engine_path=engine_path,
            variant=variant,
            args=args,
        )
    except Exception as error:
        errors.append(f"python_tensorrt: {error}")
        LOGGER.warning("Python TensorRT engine build failed for %s: %s", variant, error)

    raise RuntimeError("All TensorRT engine builders failed:\n" + "\n".join(errors))


def inspect_engine_artifact(engine_path: Path, smoke_load: bool, strict: bool) -> dict[str, Any]:
    """Inspect TensorRT engine artifact metadata and optionally smoke-load it."""

    info: dict[str, Any] = {
        "path": str(engine_path),
        "file_size_bytes": engine_path.stat().st_size,
        "native_tensorrt_backend": True,
        "smoke_load_requested": smoke_load,
        "smoke_load_passed": None,
        "smoke_load_error": None,
        "note": (
            "This .engine file is for the native TensorRT runtime path. It is separate "
            "from ONNX Runtime TensorRT EP, which consumes .onnx files."
        ),
    }

    if not smoke_load:
        return info

    try:
        from ultralytics import YOLO

        YOLO(str(engine_path))
        info["smoke_load_passed"] = True
    except Exception as error:
        info["smoke_load_passed"] = False
        info["smoke_load_error"] = str(error)

        if strict:
            raise

    return info


def build_failed_artifact(
    *,
    args: argparse.Namespace,
    variant: str,
    error: Exception,
) -> ExportArtifact:
    """Build metadata for a failed export attempt."""

    export_kwargs: dict[str, Any]

    try:
        export_kwargs = make_export_kwargs(args, variant)
    except Exception:
        export_kwargs = {}

    output_path = variant_output_path(args, variant)

    return ExportArtifact(
        variant=variant,
        export_format=variant_format(variant),
        precision=variant_precision(variant),
        quantize=variant_quantize(variant),
        success=False,
        output_path=str(output_path),
        file_size_bytes=None,
        export_seconds=None,
        export_kwargs=export_kwargs,
        error=str(error),
        onnx_info=None,
        onnxruntime_info=None,
        backend_profiles=None,
        engine_info=None,
    )


def export_one_variant(
    *,
    weights: Path,
    args: argparse.Namespace,
    variant: str,
) -> ExportArtifact:
    """Export one model artifact variant."""

    from ultralytics import YOLO

    export_format = variant_format(variant)
    precision = variant_precision(variant)
    quantize = variant_quantize(variant)
    expected_suffix = ".onnx" if export_format == "onnx" else ".engine"
    output_path = variant_output_path(args, variant)
    export_kwargs = make_export_kwargs(args, variant)

    LOGGER.info("Exporting %s:", variant)
    LOGGER.info("  format: %s", export_format)
    LOGGER.info("  precision: %s", precision)
    LOGGER.info("  weights: %s", weights)
    LOGGER.info("  target: %s", output_path)
    LOGGER.info("  kwargs: %s", export_kwargs)

    export_started_at = time.time()
    engine_build_info: dict[str, Any] | None = None

    if variant == "onnx_int8":
        # Ultralytics currently rejects int8=True for format='onnx' in this stack.
        # Create a real ONNX Runtime QDQ INT8 artifact from the FP32 ONNX instead.
        # This ONNX INT8 file is checked with ORT CPU/CUDA; TensorRT INT8 uses
        # native .engine export.
        fp32_onnx_path = variant_output_path(args, "onnx_fp32")
        if not fp32_onnx_path.is_file():
            LOGGER.info("ONNX FP32 source missing; exporting onnx_fp32 before ONNX INT8.")
            source_artifact = export_one_variant(weights=weights, args=args, variant="onnx_fp32")
            if not source_artifact.success or source_artifact.output_path is None:
                raise RuntimeError("Failed to create ONNX FP32 source for ONNX INT8 export.")
            fp32_onnx_path = Path(source_artifact.output_path).resolve()

        if output_path.exists() and args.overwrite:
            output_path.unlink()

        export_onnx_int8_via_ort(
            fp32_onnx_path=fp32_onnx_path,
            int8_onnx_path=output_path,
            args=args,
        )
        source_path = output_path
        export_seconds = time.time() - export_started_at

    elif export_format == "engine":
        if output_path.exists() and args.overwrite:
            output_path.unlink()

        if variant == "engine_int8":
            # Native TensorRT INT8 must be calibrated directly from images. Building
            # an engine from the ONNX Runtime QDQ INT8 file can fail because TensorRT
            # requires symmetric QDQ zero-points for those nodes.
            engine_build_info = export_engine_int8_via_ultralytics(
                weights=weights,
                engine_path=output_path,
                args=args,
            )
        else:
            # FP32/FP16 engines are built from the corresponding ONNX graph using
            # trtexec when available and the TensorRT Python API as a fallback.
            source_onnx_path = ensure_onnx_source_for_variant(
                weights=weights,
                args=args,
                variant=variant,
            )

            engine_build_info = export_engine_from_onnx(
                onnx_path=source_onnx_path,
                engine_path=output_path,
                variant=variant,
                args=args,
            )

        source_path = output_path
        export_seconds = time.time() - export_started_at

    else:
        model = YOLO(str(weights))
        export_result = export_with_fallbacks(
            model=model,
            export_kwargs=export_kwargs,
            variant=variant,
        )
        export_seconds = time.time() - export_started_at

        source_path = resolve_exported_path(
            export_result=export_result,
            weights=weights,
            expected_suffix=expected_suffix,
            export_started_at=export_started_at,
        )

        copy_export_to_target(
            source_path=source_path,
            target_path=output_path,
            overwrite=args.overwrite,
        )

    onnx_info: dict[str, Any] | None = None
    onnxruntime_info: dict[str, Any] | None = None
    backend_profiles: dict[str, Any] | None = None
    engine_info: dict[str, Any] | None = None

    if export_format == "onnx":
        onnx_info, onnxruntime_info, backend_profiles = inspect_onnx_artifact(
            onnx_path=output_path,
            args=args,
        )

    if export_format == "engine":
        engine_info = inspect_engine_artifact(
            engine_path=output_path,
            smoke_load=args.engine_smoke_load,
            strict=args.strict_checks,
        )
        if engine_build_info is not None:
            engine_info["build_backend"] = engine_build_info

    return ExportArtifact(
        variant=variant,
        export_format=export_format,
        precision=precision,
        quantize=quantize,
        success=True,
        output_path=str(output_path),
        file_size_bytes=output_path.stat().st_size,
        export_seconds=export_seconds,
        export_kwargs=export_kwargs,
        error=None,
        onnx_info=onnx_info,
        onnxruntime_info=onnxruntime_info,
        backend_profiles=backend_profiles,
        engine_info=engine_info,
    )


def log_export_artifact(artifact: ExportArtifact) -> None:
    """Log one exported artifact summary."""

    if not artifact.success:
        LOGGER.error("Export failed for %s: %s", artifact.variant, artifact.error)
        return

    LOGGER.info("Exported %s:", artifact.variant)
    LOGGER.info("  path: %s", artifact.output_path)
    LOGGER.info("  size: %.2f MB", artifact.file_size_bytes / (1024 * 1024))
    LOGGER.info("  seconds: %.2f", artifact.export_seconds)

    if artifact.onnx_info:
        LOGGER.info("  ONNX inputs: %s", artifact.onnx_info.get("inputs"))
        LOGGER.info("  ONNX outputs: %s", artifact.onnx_info.get("outputs"))

    if artifact.backend_profiles:
        LOGGER.info("  ONNX Runtime backend profiles recorded: ort_cpu, ort_cuda, ort_tensorrt")

    if artifact.engine_info:
        LOGGER.info("  TensorRT engine info: %s", artifact.engine_info)


def export_summary_path(args: argparse.Namespace) -> Path:
    """Return global export summary path."""

    models_root = args.onnx_dir.parent
    return models_root / f"{args.basename}_export_summary.json"


def run(args: argparse.Namespace) -> int:
    """Run model export."""

    start_time = time.time()

    variants = expand_variants(args.variants)

    args.weights = args.weights.resolve()
    args.onnx_dir = args.onnx_dir.resolve()
    args.engine_dir = args.engine_dir.resolve()
    args.ort_tensorrt_cache_dir = args.ort_tensorrt_cache_dir.resolve()

    if args.data is not None:
        args.data = args.data.resolve()

    validate_export_args(args, variants)
    dataset_info = validate_required_paths(args, variants)
    ensure_targets_available(args, variants)

    LOGGER.info("Model export configuration:")
    LOGGER.info("  weights: %s", args.weights)
    LOGGER.info("  data: %s", args.data)
    LOGGER.info("  ONNX dir: %s", args.onnx_dir)
    LOGGER.info("  native TensorRT engine dir: %s", args.engine_dir)
    LOGGER.info("  ORT TensorRT cache dir: %s", args.ort_tensorrt_cache_dir)
    LOGGER.info("  basename: %s", args.basename)
    LOGGER.info("  variants: %s", variants)
    LOGGER.info("  imgsz: %d", args.imgsz)
    LOGGER.info("  batch: %d", args.batch)
    LOGGER.info("  device: %s", args.device)
    LOGGER.info("  dynamic: %s", args.dynamic)
    LOGGER.info("  simplify: %s", args.simplify)
    LOGGER.info("  nms: %s", args.nms)
    LOGGER.info("  end2end: %s", args.end2end)
    LOGGER.info("  calibration fraction: %s", args.calibration_fraction)
    LOGGER.info("  continue on error: %s", args.continue_on_error)

    if dataset_info is not None:
        LOGGER.info("  dataset root: %s", dataset_info.dataset_root)
        LOGGER.info("  classes: %d", dataset_info.class_count)

    if args.dry_run:
        LOGGER.info("Dry run passed. No export started.")
        return 0

    try:
        import ultralytics  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Install dependencies with:\n\n"
            "    pip install -r requirements.txt"
        ) from error

    if not args.skip_onnx_check and any(variant.startswith("onnx_") for variant in variants):
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "ONNX export checking requires onnx and onnxruntime. "
                "Install dependencies with:\n\n"
                "    pip install -r requirements.txt"
            ) from error

    args.onnx_dir.mkdir(parents=True, exist_ok=True)
    args.engine_dir.mkdir(parents=True, exist_ok=True)
    args.ort_tensorrt_cache_dir.mkdir(parents=True, exist_ok=True)

    exports: list[ExportArtifact] = []

    for variant in variants:
        try:
            artifact = export_one_variant(
                weights=args.weights,
                args=args,
                variant=variant,
            )
        except Exception as error:
            artifact = build_failed_artifact(
                args=args,
                variant=variant,
                error=error,
            )

            LOGGER.error("Export failed for %s: %s", variant, error)

            if not args.continue_on_error:
                raise

        exports.append(artifact)
        log_export_artifact(artifact)

        metadata_path = variant_metadata_path(args, variant)
        write_json(metadata_path, asdict(artifact))

    successful_variants = [artifact.variant for artifact in exports if artifact.success]
    failed_variants = [artifact.variant for artifact in exports if not artifact.success]

    summary = ExportSummary(
        created_unix_time=time.time(),
        duration_seconds=time.time() - start_time,
        weights=str(args.weights),
        data_yaml=str(args.data) if args.data is not None else None,
        onnx_dir=str(args.onnx_dir),
        engine_dir=str(args.engine_dir),
        ort_tensorrt_cache_dir=str(args.ort_tensorrt_cache_dir),
        basename=args.basename,
        requested_variants=variants,
        successful_variants=successful_variants,
        failed_variants=failed_variants,
        exports=exports,
        dataset=dataset_info,
        args=args_to_dict(args),
    )

    summary_path = export_summary_path(args)
    write_json(summary_path, asdict(summary))

    LOGGER.info("Model export completed.")
    LOGGER.info("Export summary: %s", summary_path)
    LOGGER.info("Successful variants: %s", successful_variants)
    LOGGER.info("Failed variants: %s", failed_variants)

    for artifact in exports:
        if artifact.success:
            LOGGER.info("%s: %s", artifact.variant, artifact.output_path)

    if failed_variants:
        LOGGER.warning(
            "Some variants failed. Check per-variant metadata JSON files and export summary."
        )
        return 2

    return 0


def main() -> None:
    """Command-line entrypoint."""

    setup_logging()
    args = parse_args()

    try:
        exit_code = run(args)
    except Exception as error:
        LOGGER.error("export_model.py failed: %s", error)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
