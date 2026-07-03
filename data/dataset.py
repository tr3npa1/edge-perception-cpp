"""
Validate a YOLO-format BDD100K dataset and prepare a clean training YAML.

The BDD100K Kaggle dataset used by this project is expected to already be in
YOLO format. This file does not convert BDD100K JSON annotations.

Responsibilities:
- find the source dataset YAML
- resolve train/val/test image paths
- validate image files
- validate matching YOLO label files
- optionally validate YOLO label contents
- write a clean project-local YAML for training

Output:
    data/processed/bdd100k_yolo/bdd100k.yaml
    data/processed/bdd100k_yolo/dataset_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger("prepare_bdd100k_dataset")

DEFAULT_DATASET_DIR = Path("data/raw/bdd100k/bdd100k")
DEFAULT_OUTPUT_YAML = Path("data/processed/bdd100k_yolo/bdd100k.yaml")
DEFAULT_SUMMARY_PATH = Path("data/processed/bdd100k_yolo/dataset_summary.json")

SOURCE_YAML_NAMES = ("data.yaml", "dataset.yaml", "bdd100k.yaml")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REQUIRED_SPLITS = ("train", "val")
OPTIONAL_SPLITS = ("test",)

REPORT_SAMPLE_LIMIT = 10


@dataclass(frozen=True)
class SplitReport:
    """Validation summary for one dataset split."""

    split: str
    image_roots: list[str]
    image_count: int
    label_count: int
    missing_label_count: int
    invalid_label_file_count: int
    sample_images: list[str]
    sample_missing_labels: list[str]
    sample_invalid_labels: list[str]
    valid: bool
    notes: list[str]


@dataclass(frozen=True)
class DatasetReport:
    """Validation summary for the full YOLO dataset."""

    created_unix_time: float
    dataset_dir: str
    source_yaml: str
    output_yaml: str
    class_count: int
    class_names: list[str]
    splits: list[SplitReport]
    valid: bool
    notes: list[str]


def setup_logging() -> None:
    """Configure readable console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Validate YOLO-format BDD100K and write a clean training YAML."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Root directory of the downloaded YOLO-format BDD100K dataset.",
    )

    parser.add_argument(
        "--source-yaml",
        type=Path,
        default=None,
        help=(
            "Explicit source YAML path. If omitted, the script searches for "
            "data.yaml, dataset.yaml, or bdd100k.yaml under dataset-dir."
        ),
    )

    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=DEFAULT_OUTPUT_YAML,
        help="Clean project-local YAML path written for training.",
    )

    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="JSON dataset summary path.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate only. Do not write the clean output YAML.",
    )

    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help=(
            "Allow images without matching label files. Keep disabled for strict "
            "dataset validation."
        ),
    )

    parser.add_argument(
        "--skip-label-content-check",
        action="store_true",
        help="Check label file existence only. Do not parse YOLO label lines.",
    )

    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write dataset_summary.json.",
    )

    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return a dictionary"""

    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}" )
    
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}, got {type(payload).__name__}")
    
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a YAML file."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            payload,
            file,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def find_source_yaml(dataset_dir: Path, explicit_yaml: Path | None) -> Path:
    """Find the source dataset YAML."""

    if explicit_yaml is not None:
        if not explicit_yaml.is_file():
            raise FileNotFoundError(f"Explicit source YAML not found: {explicit_yaml}")

        return explicit_yaml

    for yaml_name in SOURCE_YAML_NAMES:
        candidate = dataset_dir / yaml_name

        if candidate.is_file():
            return candidate

    yaml_files = sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.name.lower() in SOURCE_YAML_NAMES
    )

    if yaml_files:
        return yaml_files[0]

    raise FileNotFoundError(
        f"No dataset YAML found under {dataset_dir}. "
        f"Expected one of: {', '.join(SOURCE_YAML_NAMES)}"
    )



def normalize_class_names(names_payload: Any) -> list[str]:
    """
    Normalize YOLO class names.

    Supported formats:
        names: [car, person]

    and:

        names:
          0: car
          1: person
    """

    if isinstance(names_payload, list):
        names = [str(name) for name in names_payload]

        if not names:
            raise ValueError("The YAML names list is empty")
        
        return names
    
    if isinstance(names_payload, dict):
        class_names_by_id: dict[int, str] = {}

        for key, value in names_payload.items():
            try:
                class_id = int(key)
            except ValueError as error:
                raise ValueError(f"Class key is not an integer: {key}") from error
            
            class_names_by_id[class_id] = str(value)

        if not class_names_by_id:
            raise ValueError("The YAML 'names' dictionary is empty.")
        
        actual_ids = sorted(class_names_by_id)
        expected_ids = list(range(len(class_names_by_id)))

        if actual_ids != expected_ids:
            raise ValueError(
                "Class IDs must be contiguous from 0. "
                f"Expected {expected_ids}, got {actual_ids}."
            )

        return [class_names_by_id[index] for index in expected_ids]

    raise ValueError(
        "Dataset YAML must contain 'names' as either a list or dictionary."
    )


def validate_nc(yaml_payload: dict[str, Any], class_names: list[str]) -> None:
    """Validate optional nc value against the class-name count."""

    nc_value = yaml_payload.get("nc")

    if nc_value is None:
        return

    try:
        nc = int(nc_value)
    except ValueError as error:
        raise ValueError(f"YAML 'nc' must be an integer, got: {nc_value}") from error

    if nc != len(class_names):
        raise ValueError(
            f"YAML 'nc' does not match number of class names. "
            f"nc={nc}, len(names)={len(class_names)}"
        )


def resolve_yaml_base_dir(source_yaml: Path, yaml_payload: dict[str, Any]) -> Path:
    """
    Resolve the base directory for train/val/test paths.

    YOLO dataset YAML usually uses:
        path: /dataset/root
        train: images/train
        val: images/val

    If path is missing, split paths are resolved relative to the YAML file.
    """

    raw_base = yaml_payload.get("path")

    if raw_base is None:
        return source_yaml.parent.resolve()
    
    base_path = Path(str(raw_base))

    if base_path.is_absolute():
        return base_path.resolve()
    
    return (source_yaml.parent / base_path).resolve()


def resolve_path_entry(entry: Any, base_dir: Path) -> list[Path]:
    """Resolve a YAML split entry into absolute paths."""
    
    if entry is None:
        return []
    
    if isinstance(entry, str):
        path = Path(entry)
        return [path.resolve() if path.is_absolute() else (base_dir / path).resolve()]
    
    if isinstance(entry, list):
        resolved_paths: list[Path] = []

        for item in entry:
            if not isinstance(item, str):
                raise ValueError(
                    "Split path lists must contain strings only. "
                    f"Got {type(item).__name__}."
                )
            
            path = Path(item)
            resolved_paths.append(
                path.resolve() if path.is_absolute() else (base_dir / path).resolve()
            )

        return resolved_paths
    
    raise ValueError(
        "Dataset YAML split entries must be strings or lists of strings. "
        f"Got {type(entry).__name__}."
    )


def infer_split_paths(dataset_dir: Path, split: str) -> list[Path]:
    """Infer common YOLO image split folders if YAML does not define them."""

    split_names = [split]

    if split == "val":
        split_names.append("valid")

    candidates: list[Path] = []

    for split_name in split_names:
        candidates.extend(
            [
                dataset_dir / "images" / split_name,
                dataset_dir / split_name / "images",
                dataset_dir / split_name,
            ]
        )
    
    for candidate in candidates:
        if candidate.is_dir():
            return [candidate.resolve()]
        
    return []


def read_image_list_file(list_path: Path) -> list[Path]:
    """Read image paths from a YOLO list file."""

    image_paths: list[Path] = []

    with list_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()

            if not stripped:
                continue

            path = Path(stripped)

            if path.is_absolute():
                image_paths.append(path.resolve())
            else:
                image_paths.append((list_path.parent / path).resolve())

    return image_paths


def collect_images(paths: list[Path]) -> list[Path]:
    """Collect image files from directories, image paths, or list files."""

    images: list[Path] = []

    for path in paths:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path.resolve())
            continue

        if path.is_file() and path.suffix.lower() == ".txt":
            images.extend(read_image_list_file(path))
            continue

        if path.is_dir():
            images.extend(
                image_path.resolve()
                for image_path in path.rglob("*")
                if image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            )

    return sorted(set(images))


def get_split_roots(
    split: str,
    dataset_dir: Path,
    source_yaml: Path,
    yaml_payload: dict[str, Any],
) -> list[Path]:
    """
    Get usable local image roots for a split.

    The source Kaggle YAML may contain absolute paths such as:
        /kaggle/input/bdd100k-yolo/train/images

    Those paths do not exist locally after downloading the dataset into this repo.
    Therefore, this function first tries the YAML paths, but if they do not
    point to real local images, it falls back to common YOLO folders under
    dataset_dir.
    """

    base_dir = resolve_yaml_base_dir(source_yaml, yaml_payload)
    
    if split == "val":
        entry = yaml_payload.get("val", yaml_payload.get("valid"))
    else:
        entry = yaml_payload.get(split)

    yaml_roots = resolve_path_entry(entry, base_dir)

    if yaml_roots and collect_images(yaml_roots):
        return yaml_roots
    
    inferred_roots = infer_split_paths(dataset_dir, split)

    if inferred_roots and collect_images(inferred_roots):
        return inferred_roots
    
    return yaml_roots or inferred_roots


def replace_first_path_part(path: Path, old: str, new: str) -> Path | None:
    """Replace the first matching path component."""
    
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part.lower() == old.lower():
            parts[index] = new
            return Path(*parts)
    
    return None


def candidate_label_paths_for_image(image_path: Path) -> list[Path]:
    """
    Return possible YOLO label paths for one image.

    Main convention:
        images/train/example.jpg
        labels/train/example.txt

    Also handles val/valid naming differences.
    """

    label_path = replace_first_path_part(image_path, old="images", new="labels")

    if label_path is None:
        return []
    
    label_path = label_path.with_suffix(".txt")
    candidates = [label_path]

    parts = list(label_path.parts)

    for index, part in enumerate(parts):
        lower = part.lower()

        if lower == "valid":
            alt_parts = parts.copy()
            alt_parts[index] = "val"
            candidates.append(Path(*alt_parts))

        if lower == "val":
            alt_parts = parts.copy()
            alt_parts[index] = "valid"
            candidates.append(Path(*alt_parts))

    unique_candidates: list[Path] = []

    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    return unique_candidates


def resolve_label_path_for_image(image_path: Path) -> Path | None:
    """Find the existing label path for an image, or return the primary expected path."""

    candidates = candidate_label_paths_for_image(image_path)

    if not candidates:
        return None
    
    for candidate in candidates:
        if candidate.is_file():
            return candidate
        
    return candidates[0]


def parse_label_line(
        line: str,
        label_path: Path,
        line_number: int,
        class_count: int,
) -> str | None:
    """
    Validate one YOLO detection label line.

    Expected format:
        class_id x_center y_center width height

    Coordinates must be normalized to [0, 1].
    Empty lines are ignored.
    """

    stripped = line.strip()

    if not stripped:
        return None
    
    parts = stripped.split()

    if len(parts) != 5:
        return f"{label_path}:{line_number} expected 5 values got {len(parts)}."
    
    try:
        class_id = int(float(parts[0]))
    except ValueError:
        return f"{label_path}:{line_number} invalid class_id: {parts[0]}"
    
    if class_id < 0 or class_id >= class_count:
        return(
            f"{label_path}:{line_number} class id {class_id} outside "
            f"valid range [0,{class_count - 1}]."
        )
    
    for token in parts[1:]:
        try:
            value = float(token)
        except ValueError:
            return f"{label_path}:{line_number} invalid box value: {token}"
        
        if value < -1e-6 or value > 1.0 + 1e-6:
            return (
                f"{label_path}:{line_number} normalized box value out of range: "
                f"{value}"
            )

    return None


def validate_label_file(label_path: Path, class_count: int) -> list[str]:
    """Validate YOLO label file contents."""

    errors: list[str] = []

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            error = parse_label_line(
                line=line,
                label_path=label_path,
                line_number=line_number,
                class_count=class_count,
            )

            if error is not None:
                errors.append(error)

    return errors


def validate_split(
        split: str,
        image_roots: list[Path],
        class_count: int,
        allow_missing_labels: bool,
        check_label_contents: bool,
) -> SplitReport:
    """Validate one dataset split."""

    images = collect_images(image_roots)

    notes: list[str] = []
    valid = True

    if not images:
        valid = False
        notes.append(f"No images found for split '{split}'.")
    
    label_count = 0
    missing_labels: list[str] = []
    invalid_labels: list[str] = []

    for image_path in images:
        label_path = resolve_label_path_for_image(image_path)

        if label_path is None:
            missing_labels.append(str(image_path))
            continue

        if not label_path.is_file():
            missing_labels.append(str(label_path))
            continue

        label_count += 1

        if check_label_contents:
            errors = validate_label_file(label_path, class_count=class_count)

            if errors:
                invalid_labels.append(str(label_path))

                for error in errors[:REPORT_SAMPLE_LIMIT]:
                    notes.append(error)

    if missing_labels and not allow_missing_labels:
        valid = False
        notes.append(
            f"Split '{split}' has {len(missing_labels)} image(s) without "
            "matching label files."
        )

    if invalid_labels:
        valid = False
        notes.append(
            f"Split '{split}' has {len(invalid_labels)} invalid label file(s)."
        )

    if valid:
        notes.append(f"Split '{split}' is valid.")

    return SplitReport(
        split=split,
        image_roots=[str(path) for path in image_roots],
        image_count=len(images),
        label_count=label_count,
        missing_label_count=len(missing_labels),
        invalid_label_file_count=len(invalid_labels),
        sample_images=[str(path) for path in images[:REPORT_SAMPLE_LIMIT]],
        sample_missing_labels=missing_labels[:REPORT_SAMPLE_LIMIT],
        sample_invalid_labels=invalid_labels[:REPORT_SAMPLE_LIMIT],
        valid=valid,
        notes=notes,
    )


def make_yaml_path_entry(paths: list[Path], dataset_dir: Path) -> str | list[str]:
    """
    Convert split roots into YAML path entries.

    Paths inside dataset_dir are written relative to dataset_dir.
    Paths outside dataset_dir are written as absolute paths.
    """

    entries: list[str] = []

    for path in paths:
        resolved = path.resolve()

        try:
            relative = resolved.relative_to(dataset_dir)
            entries.append(relative.as_posix())
        except ValueError:
            entries.append(resolved.as_posix())

    if len(entries) == 1:
        return entries[0]
    
    return entries


def build_clean_yaml(
    dataset_dir: Path,
    class_names: list[str],
    split_roots: dict[str, list[Path]],
) -> dict[str, Any]:
    """Build the clean YOLO dataset YAML used by training scripts."""

    payload: dict[str, Any] = {
        "path": dataset_dir.resolve().as_posix(),
        "train": make_yaml_path_entry(split_roots["train"], dataset_dir),
        "val": make_yaml_path_entry(split_roots["val"], dataset_dir),
        "nc": len(class_names),
        "names": {index: name for index, name in enumerate(class_names)},
    }

    if "test" in split_roots:
        payload["test"] = make_yaml_path_entry(split_roots["test"], dataset_dir)

    return payload


def log_split_report(report: SplitReport) -> None:
    """Log one split report."""

    LOGGER.info("Split: %s", report.split)
    LOGGER.info("  image roots: %s", ", ".join(report.image_roots))
    LOGGER.info("  images: %d", report.image_count)
    LOGGER.info("  labels: %d", report.label_count)
    LOGGER.info("  missing labels: %d", report.missing_label_count)
    LOGGER.info("  invalid label files: %d", report.invalid_label_file_count)

    for note in report.notes:
        if report.valid:
            LOGGER.info("  %s", note)
        else:
            LOGGER.warning("  %s", note)


def log_dataset_report(report: DatasetReport) -> None:
    """Log dataset validation summary."""

    LOGGER.info("BDD100K YOLO dataset summary:")
    LOGGER.info("  dataset_dir: %s", report.dataset_dir)
    LOGGER.info("  source_yaml: %s", report.source_yaml)
    LOGGER.info("  output_yaml: %s", report.output_yaml)
    LOGGER.info("  classes: %d", report.class_count)
    LOGGER.info("  class names: %s", ", ".join(report.class_names))

    for split_report in report.splits:
        log_split_report(split_report)

    for note in report.notes:
        if report.valid:
            LOGGER.info("  %s", note)
        else:
            LOGGER.warning("  %s", note)


def run(args: argparse.Namespace) -> None:
    """Run dataset validation and clean YAML generation."""

    dataset_dir = args.dataset_dir.resolve()

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    source_yaml = find_source_yaml(
        dataset_dir=dataset_dir,
        explicit_yaml=args.source_yaml,
    ).resolve()

    source_payload = read_yaml(source_yaml)

    class_names = normalize_class_names(source_payload.get("names"))
    validate_nc(source_payload, class_names)

    split_roots: dict[str, list[Path]] = {}
    split_reports: list[SplitReport] = []

    for split in (*REQUIRED_SPLITS, *OPTIONAL_SPLITS):
        roots = get_split_roots(
            split=split,
            dataset_dir=dataset_dir,
            source_yaml=source_yaml,
            yaml_payload=source_payload,
        )

        if split in OPTIONAL_SPLITS and not roots:
            continue

        split_roots[split] = roots

        split_report = validate_split(
            split=split,
            image_roots=roots,
            class_count=len(class_names),
            allow_missing_labels=args.allow_missing_labels,
            check_label_contents=not args.skip_label_content_check,
        )

        split_reports.append(split_report)

    notes: list[str] = []
    valid = True

    available_splits = {split_report.split for split_report in split_reports}

    for required_split in REQUIRED_SPLITS:
        if required_split not in available_splits:
            valid = False
            notes.append(f"Required split missing: {required_split}")

    for split_report in split_reports:
        if not split_report.valid:
            valid = False

    if valid:
        notes.append("BDD100K YOLO dataset is valid for training.")

    report = DatasetReport(
        created_unix_time=time.time(),
        dataset_dir=str(dataset_dir),
        source_yaml=str(source_yaml),
        output_yaml=str(args.output_yaml.resolve()),
        class_count=len(class_names),
        class_names=class_names,
        splits=split_reports,
        valid=valid,
        notes=notes,
    )

    log_dataset_report(report)

    if not args.no_summary:
        write_json(args.summary_path, asdict(report))
        LOGGER.info("Wrote dataset summary: %s", args.summary_path)

    if not valid:
        raise RuntimeError("BDD100K YOLO dataset validation failed.")

    if not args.validate_only:
        clean_yaml = build_clean_yaml(
            dataset_dir=dataset_dir,
            class_names=class_names,
            split_roots=split_roots,
        )

        write_yaml(args.output_yaml, clean_yaml)
        LOGGER.info("Wrote clean training YAML: %s", args.output_yaml)

    LOGGER.info("Dataset preparation completed successfully.")


def main() -> None:
    """Command-line entrypoint."""

    setup_logging()
    args = parse_args()

    try:
        run(args)
    except Exception as error:
        LOGGER.error("dataset.py failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()