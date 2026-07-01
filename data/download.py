"""
Download and validate BDD100K data from Kaggle.

This script prepares raw BDD100K data for the project.

Default target:
    Kaggle BDD100K YOLO-format dataset

Expected output:
    data/raw/bdd100k/
    ├── kaggle_download/
    ├── bdd100k/
    └── download_manifest.json

The script does not train a model.
The script does not run inference.
The script only downloads, extracts, locates, and validates dataset files.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


LOGGER = logging.getLogger("download_bdd100k")

DEFAULT_KAGGLE_DATASET = "a7madmostafa/bdd100k-yolo"

DEFAULT_RAW_DIR = Path("data/raw/bdd100k")
DEFAULT_DOWNLOAD_DIRNAME = "kaggle_download"
DEFAULT_DATASET_DIRNAME = "bdd100k"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABEL_EXTENSION = ".txt"
YAML_EXTENSIONS = {".yaml", ".yml"}


@dataclass(frozen=True)
class DatasetLocation:
    """Resolved dataset paths after download and extraction."""

    raw_dir: str
    download_dir: str
    dataset_dir: str
    images_dir: str | None
    labels_dir: str | None
    yaml_path: str | None


@dataclass(frozen=True)
class ValidationReport:
    """Validation summary for the downloaded dataset."""

    image_count: int
    label_count: int
    yaml_found: bool
    valid: bool
    notes: list[str]


@dataclass(frozen=True)
class DownloadManifest:
    """Manifest written after dataset preparation."""

    created_unix_time: float
    kaggle_dataset: str
    location: DatasetLocation
    validation: ValidationReport


def setup_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Download and validate BDD100K data from Kaggle."
    )

    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default=DEFAULT_KAGGLE_DATASET,
        help=(
            "Kaggle dataset slug. Default is the BDD100K YOLO-format dataset: "
            f"{DEFAULT_KAGGLE_DATASET}"
        ),
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Raw dataset directory.",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete raw-dir before downloading.",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Run Kaggle download even if files already exist.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing data. Do not download.",
    )

    parser.add_argument(
        "--allow-missing-yaml",
        action="store_true",
        help="Allow validation to pass even if no dataset YAML is found.",
    )

    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def clean_dir(path: Path) -> None:
    """Delete a directory if it exists."""

    if path.exists():
        LOGGER.warning("Deleting directory: %s", path)
        shutil.rmtree(path)


def run_command(command: list[str]) -> None:
    """Run a shell command and fail clearly if it fails."""

    LOGGER.info("Running command: %s", " ".join(command))

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "The Kaggle CLI was not found. Install it with:\n\n"
            "    pip install kaggle\n\n"
            "Then configure your Kaggle API token."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Kaggle download failed. Make sure your Kaggle API token is configured "
            "and that you accepted any dataset/competition terms on Kaggle."
        ) from error


def download_from_kaggle(
    kaggle_dataset: str,
    download_dir: Path,
    force_download: bool,
) -> None:
    """Download a Kaggle dataset into download_dir."""

    ensure_dir(download_dir)

    existing_files = [path for path in download_dir.iterdir()]

    if existing_files and not force_download:
        LOGGER.info(
            "Download directory already contains files. Skipping Kaggle download: %s",
            download_dir,
        )
        return

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        kaggle_dataset,
        "-p",
        str(download_dir),
    ]

    run_command(command)


def extract_zip_files(download_dir: Path) -> list[Path]:
    """Extract all zip files in download_dir."""

    zip_files = sorted(download_dir.glob("*.zip"))
    extracted: list[Path] = []

    for zip_path in zip_files:
        LOGGER.info("Extracting: %s", zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(download_dir)

        extracted.append(zip_path)

    return extracted


def count_files_with_extensions(root: Path, extensions: set[str]) -> int:
    """Count files under root matching a set of suffixes."""

    if not root.is_dir():
        return 0

    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def find_first_dir_named(root: Path, names: set[str]) -> Path | None:
    """Find the first directory whose name matches one of names."""

    if not root.exists():
        return None

    candidates = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.lower() in names
    ]

    if not candidates:
        return None

    return sorted(candidates, key=lambda path: len(path.parts))[0]


def find_dataset_yaml(root: Path) -> Path | None:
    """Find a dataset YAML file."""

    if not root.exists():
        return None

    yaml_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in YAML_EXTENSIONS
    )

    if not yaml_files:
        return None

    preferred_names = {"data.yaml", "dataset.yaml", "bdd100k.yaml"}

    for path in yaml_files:
        if path.name.lower() in preferred_names:
            return path

    return yaml_files[0]


def find_images_dir(dataset_dir: Path) -> Path | None:
    """Find the most likely YOLO images directory."""

    direct = dataset_dir / "images"

    if direct.is_dir():
        return direct

    return find_first_dir_named(dataset_dir, {"images", "image"})


def find_labels_dir(dataset_dir: Path) -> Path | None:
    """Find the most likely YOLO labels directory."""

    direct = dataset_dir / "labels"

    if direct.is_dir():
        return direct

    return find_first_dir_named(dataset_dir, {"labels", "label"})


def prepare_dataset_dir(download_dir: Path, dataset_dir: Path) -> None:
    """
    Create a stable dataset directory.

    Kaggle datasets often extract with their own folder names. To keep the repo
    stable, this function copies the extracted content into data/raw/bdd100k/bdd100k.
    """

    if dataset_dir.exists():
        LOGGER.info("Dataset directory already exists: %s", dataset_dir)
        return

    ensure_dir(dataset_dir)

    extracted_items = [
        path
        for path in download_dir.iterdir()
        if path.name != dataset_dir.name and path.suffix.lower() != ".zip"
    ]

    if not extracted_items:
        LOGGER.warning("No extracted dataset files found in %s", download_dir)
        return

    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        source_root = extracted_items[0]

        LOGGER.info("Copying extracted dataset directory into: %s", dataset_dir)

        for item in source_root.iterdir():
            destination = dataset_dir / item.name

            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)

        return

    LOGGER.info("Copying extracted dataset contents into: %s", dataset_dir)

    for item in extracted_items:
        destination = dataset_dir / item.name

        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def resolve_location(raw_dir: Path) -> DatasetLocation:
    """Resolve important dataset paths."""

    download_dir = raw_dir / DEFAULT_DOWNLOAD_DIRNAME
    dataset_dir = raw_dir / DEFAULT_DATASET_DIRNAME

    images_dir = find_images_dir(dataset_dir)
    labels_dir = find_labels_dir(dataset_dir)
    yaml_path = find_dataset_yaml(dataset_dir)

    return DatasetLocation(
        raw_dir=str(raw_dir.resolve()),
        download_dir=str(download_dir.resolve()),
        dataset_dir=str(dataset_dir.resolve()),
        images_dir=str(images_dir.resolve()) if images_dir else None,
        labels_dir=str(labels_dir.resolve()) if labels_dir else None,
        yaml_path=str(yaml_path.resolve()) if yaml_path else None,
    )


def validate_dataset(
    location: DatasetLocation,
    allow_missing_yaml: bool,
) -> ValidationReport:
    """Validate that downloaded BDD100K data looks usable."""

    images_dir = Path(location.images_dir) if location.images_dir else None
    labels_dir = Path(location.labels_dir) if location.labels_dir else None

    image_count = (
        count_files_with_extensions(images_dir, IMAGE_EXTENSIONS)
        if images_dir is not None
        else 0
    )

    label_count = (
        count_files_with_extensions(labels_dir, {LABEL_EXTENSION})
        if labels_dir is not None
        else 0
    )

    yaml_found = location.yaml_path is not None

    notes: list[str] = []
    valid = True

    if image_count == 0:
        valid = False
        notes.append("No images found.")

    if label_count == 0:
        valid = False
        notes.append("No YOLO .txt label files found.")

    if not yaml_found and not allow_missing_yaml:
        valid = False
        notes.append("No dataset YAML file found.")

    if valid:
        notes.append("BDD100K dataset is valid for YOLO training.")

    return ValidationReport(
        image_count=image_count,
        label_count=label_count,
        yaml_found=yaml_found,
        valid=valid,
        notes=notes,
    )


def write_manifest(
    raw_dir: Path,
    kaggle_dataset: str,
    location: DatasetLocation,
    validation: ValidationReport,
) -> Path:
    """Write download manifest."""

    manifest = DownloadManifest(
        created_unix_time=time.time(),
        kaggle_dataset=kaggle_dataset,
        location=location,
        validation=validation,
    )

    manifest_path = raw_dir / "download_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(manifest), file, indent=2)

    return manifest_path


def log_validation(validation: ValidationReport) -> None:
    """Log validation results."""

    LOGGER.info("Validation summary:")
    LOGGER.info("  images: %d", validation.image_count)
    LOGGER.info("  labels: %d", validation.label_count)
    LOGGER.info("  yaml found: %s", validation.yaml_found)

    for note in validation.notes:
        if validation.valid:
            LOGGER.info("  %s", note)
        else:
            LOGGER.warning("  %s", note)


def run(args: argparse.Namespace) -> None:
    """Run the full workflow."""

    raw_dir: Path = args.raw_dir
    download_dir = raw_dir / DEFAULT_DOWNLOAD_DIRNAME
    dataset_dir = raw_dir / DEFAULT_DATASET_DIRNAME

    if args.clean:
        clean_dir(raw_dir)

    ensure_dir(raw_dir)
    ensure_dir(download_dir)

    if args.validate_only:
        LOGGER.info("Validation-only mode enabled.")
    else:
        download_from_kaggle(
            kaggle_dataset=args.kaggle_dataset,
            download_dir=download_dir,
            force_download=args.force_download,
        )
        extract_zip_files(download_dir)
        prepare_dataset_dir(download_dir, dataset_dir)

    location = resolve_location(raw_dir)

    validation = validate_dataset(
        location=location,
        allow_missing_yaml=args.allow_missing_yaml,
    )

    log_validation(validation)

    manifest_path = write_manifest(
        raw_dir=raw_dir,
        kaggle_dataset=args.kaggle_dataset,
        location=location,
        validation=validation,
    )

    LOGGER.info("Wrote manifest: %s", manifest_path)

    if not validation.valid:
        raise RuntimeError("Downloaded BDD100K data is not valid yet.")

    LOGGER.info("BDD100K data is ready.")


def main() -> None:
    """CLI entrypoint."""

    setup_logging()
    args = parse_args()

    try:
        run(args)
    except Exception as error:
        LOGGER.error("download.py failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()