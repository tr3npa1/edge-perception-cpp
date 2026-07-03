"""
Train or fine-tune YOLO26M on the prepared BDD100K YOLO dataset.

This script is the training entrypoint for edge-perception-cpp.

Expected dataset input:
    data/processed/bdd100k_yolo/bdd100k.yaml

Run data/dataset.py before this script.

Responsibilities:
- validate the prepared BDD100K YAML
- train or resume YOLO26M training
- save checkpoints through Ultralytics
- save local training metadata
- integrate MLflow experiment tracking
- log epoch-level training metrics to MLflow
- replay Ultralytics results.csv metrics to MLflow after training

This script does not export ONNX.
This script does not run C++ inference.
This script does not benchmark deployment latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger("train_yolo26m_bdd100k")

DEFAULT_MODEL = "yolo26m.pt"
DEFAULT_DATA_YAML = Path("data/processed/bdd100k_yolo/bdd100k.yaml")
DEFAULT_PROJECT_DIR = Path("runs/train")
DEFAULT_RUN_NAME = "yolo26m_bdd100k"

DEFAULT_MLFLOW_EXPERIMENT = "edge-perception-cpp"
DEFAULT_MLFLOW_RUN_NAME = "yolo26m_bdd100k"

KAGGLE_PATH_MARKER = "/kaggle/input"


@dataclass(frozen=True)
class DatasetInfo:
    """Resolved dataset information used for training."""

    yaml_path: str
    dataset_root: str
    train_images: str
    val_images: str
    test_images: str | None
    class_count: int
    class_names: list[str]


@dataclass(frozen=True)
class TrainingMetadata:
    """Metadata saved after a training run."""

    created_unix_time: float
    duration_seconds: float
    status: str
    model: str
    data_yaml: str
    save_dir: str
    best_checkpoint: str | None
    last_checkpoint: str | None
    args: dict[str, Any]
    dataset: DatasetInfo
    train_kwargs: dict[str, Any]
    environment: dict[str, Any]
    mlflow: dict[str, Any]


def setup_logging() -> None:
    """Configure readable console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train YOLO26M on the prepared BDD100K YOLO dataset."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Ultralytics model name or checkpoint path.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Prepared local BDD100K dataset YAML.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Training device, for example '0', '0,1', or 'cpu'.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Ultralytics project directory.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Run name inside the project directory.",
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing into an existing run directory.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from --resume-model.",
    )

    parser.add_argument(
        "--resume-model",
        type=Path,
        default=None,
        help="Checkpoint path used for resume or checkpoint fine-tuning.",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        help="Ultralytics optimizer setting.",
    )

    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate.",
    )

    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning-rate fraction.",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.937,
        help="SGD momentum or Adam beta1 depending on optimizer.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Optimizer weight decay.",
    )

    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        help="Warmup epochs.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early-stopping patience.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training where supported.",
    )

    parser.add_argument(
        "--rect",
        action="store_true",
        help="Use rectangular training batches.",
    )

    parser.add_argument(
        "--cos-lr",
        action="store_true",
        help="Use cosine learning-rate schedule.",
    )

    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic augmentation for the final N epochs.",
    )

    parser.add_argument(
        "--cache",
        choices=("none", "ram", "disk"),
        default="none",
        help="Dataset caching mode.",
    )

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable automatic mixed precision training.",
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use. Keep 1.0 for real training.",
    )

    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every N epochs. Use -1 to disable periodic saves.",
    )

    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable training plots.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose Ultralytics logging.",
    )

    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable MLflow tracking.",
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help=(
            "MLflow tracking URI. Example: http://127.0.0.1:5000. "
            "If omitted, MLflow uses its default local tracking location."
        ),
    )

    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=DEFAULT_MLFLOW_EXPERIMENT,
        help="MLflow experiment name.",
    )

    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=DEFAULT_MLFLOW_RUN_NAME,
        help="MLflow run name.",
    )

    parser.add_argument(
        "--mlflow-log-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Log best.pt and last.pt as MLflow artifacts. Disabled by default "
            "because checkpoint files are large."
        ),
    )

    parser.add_argument(
        "--mlflow-log-run-dir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Log the full Ultralytics run directory as MLflow artifacts. Disabled "
            "by default because it can be large."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without training.",
    )

    return parser.parse_args()


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
    """Reject Kaggle-only YAML paths before training starts."""

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
                    "Run data/dataset.py and train with "
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


def resolve_split_path(split_entry: Any, dataset_root: Path) -> Path | None:
    """Resolve one split path from a YOLO dataset YAML."""

    if split_entry is None:
        return None

    if isinstance(split_entry, list):
        if len(split_entry) != 1:
            raise ValueError(
                "This training script expects one image root per split. "
                f"Got {len(split_entry)} entries."
            )

        split_entry = split_entry[0]

    if not isinstance(split_entry, str):
        raise ValueError(
            "Dataset split entries must be strings or single-item lists. "
            f"Got {type(split_entry).__name__}."
        )

    split_path = Path(split_entry)

    if split_path.is_absolute():
        return split_path.resolve()

    return (dataset_root / split_path).resolve()


def validate_existing_dir(path: Path | None, split_name: str, required: bool) -> str | None:
    """Validate that a split directory exists when required."""

    if path is None:
        if required:
            raise ValueError(f"Dataset YAML is missing required split: {split_name}")

        return None

    if not path.is_dir():
        raise FileNotFoundError(f"{split_name} image directory not found: {path}")

    return str(path)


def validate_dataset_yaml(data_yaml: Path) -> DatasetInfo:
    """Validate the prepared local BDD100K YAML before training."""

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

    train_path = resolve_split_path(yaml_payload.get("train"), dataset_root)
    val_path = resolve_split_path(yaml_payload.get("val"), dataset_root)
    test_path = resolve_split_path(yaml_payload.get("test"), dataset_root)

    train_images = validate_existing_dir(train_path, "train", required=True)
    val_images = validate_existing_dir(val_path, "val", required=True)
    test_images = validate_existing_dir(test_path, "test", required=False)

    return DatasetInfo(
        yaml_path=str(data_yaml),
        dataset_root=str(dataset_root),
        train_images=train_images if train_images is not None else "",
        val_images=val_images if val_images is not None else "",
        test_images=test_images,
        class_count=len(class_names),
        class_names=class_names,
    )


def args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Convert parsed arguments into JSON-serializable values."""

    output: dict[str, Any] = {}

    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value

    return output


def build_train_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build keyword arguments passed to Ultralytics training."""

    cache_value: bool | str = False if args.cache == "none" else args.cache

    return {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "resume": args.resume,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "rect": args.rect,
        "cos_lr": args.cos_lr,
        "close_mosaic": args.close_mosaic,
        "cache": cache_value,
        "amp": args.amp,
        "fraction": args.fraction,
        "save_period": args.save_period,
        "plots": args.plots,
        "verbose": args.verbose,
    }


def get_ultralytics_version() -> str | None:
    """Return the installed Ultralytics version if available."""

    try:
        import ultralytics

        return str(ultralytics.__version__)
    except Exception:
        return None


def get_mlflow_version() -> str | None:
    """Return the installed MLflow version if available."""

    try:
        import mlflow

        return str(mlflow.__version__)
    except Exception:
        return None


def resolve_model_reference(args: argparse.Namespace) -> str:
    """Resolve the model reference used to initialize YOLO."""

    if args.resume:
        if args.resume_model is None:
            raise ValueError("--resume requires --resume-model pointing to last.pt.")

        if not args.resume_model.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_model}")

        return str(args.resume_model)

    if args.resume_model is not None:
        if not args.resume_model.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_model}")

        return str(args.resume_model)

    return args.model


def resolve_save_dir(model: Any, results: Any, args: argparse.Namespace) -> Path:
    """Resolve the Ultralytics run directory."""

    candidates = [
        getattr(results, "save_dir", None),
        getattr(getattr(model, "trainer", None), "save_dir", None),
    ]

    for candidate in candidates:
        if candidate is not None:
            return Path(candidate).resolve()

    return (args.project / args.name).resolve()


def checkpoint_path(save_dir: Path, filename: str) -> str | None:
    """Return a checkpoint path if it exists."""

    path = save_dir / "weights" / filename

    if path.is_file():
        return str(path)

    return None


def write_training_metadata(save_dir: Path, metadata: TrainingMetadata) -> Path:
    """Write training metadata inside the run directory."""

    save_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = save_dir / "training_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(metadata), file, indent=2)

    return metadata_path


def sanitize_metric_name(metric_name: str) -> str:
    """Sanitize a results.csv column name for MLflow metric logging."""

    cleaned = metric_name.strip()
    cleaned = cleaned.replace(" ", "_")
    cleaned = cleaned.replace("(", "")
    cleaned = cleaned.replace(")", "")
    cleaned = cleaned.replace(":", "_")
    cleaned = cleaned.replace(",", "_")

    return cleaned


def log_results_csv_metrics_to_mlflow(mlflow: Any, results_csv: Path) -> None:
    """
    Replay Ultralytics results.csv metrics to MLflow.

    Ultralytics should already log epoch metrics through its MLflow callback.
    This replay is an explicit backup so that train/val losses, precision,
    recall, mAP, and learning-rate columns appear as MLflow metric curves.
    """

    if not results_csv.is_file():
        LOGGER.warning("results.csv not found, skipping metric replay: %s", results_csv)
        return

    with results_csv.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row_index, row in enumerate(reader):
            raw_epoch = row.get("epoch")

            try:
                step = int(float(raw_epoch)) if raw_epoch is not None else row_index
            except ValueError:
                step = row_index

            for raw_key, raw_value in row.items():
                if raw_key is None or raw_value is None:
                    continue

                key = raw_key.strip()
                value_text = raw_value.strip()

                if not key or not value_text:
                    continue

                if key.lower() == "epoch":
                    continue

                try:
                    value = float(value_text)
                except ValueError:
                    continue

                metric_name = f"csv/{sanitize_metric_name(key)}"
                mlflow.log_metric(metric_name, value, step=step)


def configure_mlflow(args: argparse.Namespace) -> Any | None:
    """
    Configure MLflow and Ultralytics MLflow integration.

    The script starts the MLflow run itself. Ultralytics sees the active run and
    logs its epoch-level training metrics into the same run.
    """

    if not args.mlflow:
        try:
            from ultralytics import settings

            settings.update({"mlflow": False})
        except Exception:
            pass

        LOGGER.info("MLflow tracking disabled.")
        return None

    try:
        import mlflow
        from ultralytics import settings
    except ImportError as error:
        raise RuntimeError(
            "MLflow or Ultralytics is not installed. Install dependencies with:\n\n"
            "    pip install -r requirements.txt"
        ) from error

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri

    mlflow.set_experiment(args.mlflow_experiment)

    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment
    os.environ["MLFLOW_RUN"] = args.mlflow_run_name
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "true"

    settings.update({"mlflow": True})

    LOGGER.info("MLflow tracking enabled.")
    LOGGER.info("  experiment: %s", args.mlflow_experiment)
    LOGGER.info("  run name: %s", args.mlflow_run_name)
    LOGGER.info("  tracking uri: %s", mlflow.get_tracking_uri())

    return mlflow


def log_mlflow_inputs(
    mlflow: Any,
    args: argparse.Namespace,
    dataset_info: DatasetInfo,
    train_kwargs: dict[str, Any],
    model_reference: str,
) -> None:
    """Log pre-training metadata to MLflow without conflicting with Ultralytics params."""

    mlflow.set_tags(
        {
            "project": "edge-perception-cpp",
            "stage": "training",
            "task": "object-detection",
            "dataset": "BDD100K",
            "model_family": "YOLO26",
            "model_size": "M",
        }
    )

    mlflow.log_params(
        {
            "script_model_reference": model_reference,
            "script_data_yaml": str(args.data.resolve()),
            "script_dataset_root": dataset_info.dataset_root,
            "script_class_count": dataset_info.class_count,
            "script_tracking_source": "train_detector.py",
        }
    )

    mlflow.log_dict(asdict(dataset_info), "dataset_info.json")
    mlflow.log_dict(train_kwargs, "train_kwargs.json")
    mlflow.log_dict(args_to_dict(args), "script_args.json")

    if args.data.is_file():
        mlflow.log_artifact(str(args.data), artifact_path="dataset")


def log_mlflow_outputs(
    mlflow: Any,
    args: argparse.Namespace,
    save_dir: Path,
    metadata_path: Path,
    best_checkpoint: str | None,
    last_checkpoint: str | None,
    duration_seconds: float,
) -> None:
    """Log post-training outputs to MLflow."""

    mlflow.log_metric("script/duration_seconds", duration_seconds)
    mlflow.log_artifact(str(metadata_path), artifact_path="metadata")

    results_csv = save_dir / "results.csv"
    args_yaml = save_dir / "args.yaml"

    if results_csv.is_file():
        log_results_csv_metrics_to_mlflow(mlflow, results_csv)
        mlflow.log_artifact(str(results_csv), artifact_path="ultralytics")

    if args_yaml.is_file():
        mlflow.log_artifact(str(args_yaml), artifact_path="ultralytics")

    if args.mlflow_log_checkpoints:
        if best_checkpoint is not None:
            mlflow.log_artifact(best_checkpoint, artifact_path="checkpoints")

        if last_checkpoint is not None:
            mlflow.log_artifact(last_checkpoint, artifact_path="checkpoints")

    if args.mlflow_log_run_dir:
        mlflow.log_artifacts(str(save_dir), artifact_path="ultralytics_run_dir")


def build_metadata(
    args: argparse.Namespace,
    status: str,
    start_time: float,
    model_reference: str,
    dataset_info: DatasetInfo,
    train_kwargs: dict[str, Any],
    save_dir: Path,
    mlflow_run_id: str | None,
    mlflow_tracking_uri: str | None,
) -> TrainingMetadata:
    """Build local training metadata."""

    duration_seconds = time.time() - start_time
    best_checkpoint = checkpoint_path(save_dir, "best.pt")
    last_checkpoint = checkpoint_path(save_dir, "last.pt")

    return TrainingMetadata(
        created_unix_time=time.time(),
        duration_seconds=duration_seconds,
        status=status,
        model=model_reference,
        data_yaml=str(args.data.resolve()),
        save_dir=str(save_dir),
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        args=args_to_dict(args),
        dataset=dataset_info,
        train_kwargs=train_kwargs,
        environment={
            "python_version": sys.version,
            "platform": platform.platform(),
            "ultralytics_version": get_ultralytics_version(),
            "mlflow_version": get_mlflow_version(),
        },
        mlflow={
            "enabled": args.mlflow,
            "experiment": args.mlflow_experiment,
            "run_name": args.mlflow_run_name,
            "run_id": mlflow_run_id,
            "tracking_uri": mlflow_tracking_uri,
        },
    )


def log_training_summary(metadata: TrainingMetadata, metadata_path: Path) -> None:
    """Log final training paths."""

    LOGGER.info("Training completed.")
    LOGGER.info("Run directory: %s", metadata.save_dir)
    LOGGER.info("Best checkpoint: %s", metadata.best_checkpoint)
    LOGGER.info("Last checkpoint: %s", metadata.last_checkpoint)
    LOGGER.info("Training metadata: %s", metadata_path)

    if metadata.mlflow.get("enabled"):
        LOGGER.info("MLflow run ID: %s", metadata.mlflow.get("run_id"))
        LOGGER.info("MLflow tracking URI: %s", metadata.mlflow.get("tracking_uri"))


def run_training_without_mlflow(
    args: argparse.Namespace,
    start_time: float,
    dataset_info: DatasetInfo,
    model_reference: str,
    train_kwargs: dict[str, Any],
) -> None:
    """Run training without MLflow."""

    from ultralytics import YOLO

    model = YOLO(model_reference)
    results = model.train(**train_kwargs)

    save_dir = resolve_save_dir(model=model, results=results, args=args)

    metadata = build_metadata(
        args=args,
        status="completed",
        start_time=start_time,
        model_reference=model_reference,
        dataset_info=dataset_info,
        train_kwargs=train_kwargs,
        save_dir=save_dir,
        mlflow_run_id=None,
        mlflow_tracking_uri=None,
    )

    metadata_path = write_training_metadata(save_dir=save_dir, metadata=metadata)
    log_training_summary(metadata, metadata_path)


def run_training_with_mlflow(
    args: argparse.Namespace,
    start_time: float,
    dataset_info: DatasetInfo,
    model_reference: str,
    train_kwargs: dict[str, Any],
    mlflow: Any,
) -> None:
    """Run training with MLflow tracking."""

    from ultralytics import YOLO

    with mlflow.start_run(run_name=args.mlflow_run_name) as active_run:
        mlflow_run_id = active_run.info.run_id
        mlflow_tracking_uri = mlflow.get_tracking_uri()

        log_mlflow_inputs(
            mlflow=mlflow,
            args=args,
            dataset_info=dataset_info,
            train_kwargs=train_kwargs,
            model_reference=model_reference,
        )

        model = YOLO(model_reference)
        results = model.train(**train_kwargs)

        save_dir = resolve_save_dir(model=model, results=results, args=args)

        metadata = build_metadata(
            args=args,
            status="completed",
            start_time=start_time,
            model_reference=model_reference,
            dataset_info=dataset_info,
            train_kwargs=train_kwargs,
            save_dir=save_dir,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

        metadata_path = write_training_metadata(save_dir=save_dir, metadata=metadata)

        log_mlflow_outputs(
            mlflow=mlflow,
            args=args,
            save_dir=save_dir,
            metadata_path=metadata_path,
            best_checkpoint=metadata.best_checkpoint,
            last_checkpoint=metadata.last_checkpoint,
            duration_seconds=metadata.duration_seconds,
        )

        log_training_summary(metadata, metadata_path)


def run(args: argparse.Namespace) -> None:
    """Validate configuration and run training."""

    start_time = time.time()

    dataset_info = validate_dataset_yaml(args.data)
    model_reference = resolve_model_reference(args)
    train_kwargs = build_train_kwargs(args)

    LOGGER.info("Training configuration:")
    LOGGER.info("  model: %s", model_reference)
    LOGGER.info("  data: %s", args.data.resolve())
    LOGGER.info("  dataset root: %s", dataset_info.dataset_root)
    LOGGER.info("  classes: %d", dataset_info.class_count)
    LOGGER.info("  train images: %s", dataset_info.train_images)
    LOGGER.info("  val images: %s", dataset_info.val_images)
    LOGGER.info("  test images: %s", dataset_info.test_images)
    LOGGER.info("  epochs: %d", args.epochs)
    LOGGER.info("  image size: %d", args.imgsz)
    LOGGER.info("  batch size: %d", args.batch)
    LOGGER.info("  device: %s", args.device)
    LOGGER.info("  MLflow enabled: %s", args.mlflow)

    if args.dry_run:
        LOGGER.info("Dry run passed. No training started.")
        return

    try:
        import ultralytics  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Install dependencies with:\n\n"
            "    pip install -r requirements.txt"
        ) from error

    mlflow = configure_mlflow(args)

    if mlflow is None:
        run_training_without_mlflow(
            args=args,
            start_time=start_time,
            dataset_info=dataset_info,
            model_reference=model_reference,
            train_kwargs=train_kwargs,
        )
        return

    run_training_with_mlflow(
        args=args,
        start_time=start_time,
        dataset_info=dataset_info,
        model_reference=model_reference,
        train_kwargs=train_kwargs,
        mlflow=mlflow,
    )


def main() -> None:
    """Command-line entrypoint."""

    setup_logging()
    args = parse_args()

    try:
        run(args)
    except Exception as error:
        LOGGER.error("train_detector.py failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()