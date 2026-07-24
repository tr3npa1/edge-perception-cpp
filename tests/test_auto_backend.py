#!/usr/bin/env python3
"""Exercise the public --backend auto CLI with the generated ONNX fixture."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ORT_BACKENDS = {"ort_tensorrt", "ort_cuda", "ort_cpu"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate automatic ONNX Runtime backend selection."
    )
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_ppm(path: Path) -> None:
    """Write a dependency-free 4x4 RGB test image accepted by OpenCV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = bytes(
        channel
        for row in range(4)
        for column in range(4)
        for channel in (
            32 + row * 20,
            64 + column * 20,
            96 + (row + column) * 10,
        )
    )
    path.write_bytes(b"P6\n4 4\n255\n" + pixels)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    args = parse_args()
    executable = args.executable.resolve()
    model = args.model.resolve()
    output_dir = args.output_dir.resolve()

    if not executable.is_file():
        raise FileNotFoundError(f"Executable not found: {executable}")
    if not model.is_file():
        raise FileNotFoundError(f"Generated ONNX model not found: {model}")

    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = output_dir / "input.ppm"
    write_ppm(image)

    automatic = run(
        [
            str(executable),
            "--mode",
            "infer",
            "--source",
            str(image),
            "--backend",
            "auto",
            "--model",
            str(model),
            "--precision",
            "fp32",
            "--output-dir",
            str(output_dir),
        ]
    )
    if automatic.returncode != 0:
        raise RuntimeError(
            "Automatic backend inference failed.\n" + automatic.stdout
        )

    metadata_path = output_dir / "backend_selection.json"
    detections_path = output_dir / "detections.jsonl"
    if not metadata_path.is_file() or not detections_path.is_file():
        raise RuntimeError(
            "Automatic inference did not produce backend metadata and detections."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("requested_backend") != "auto":
        raise AssertionError("Metadata did not preserve the automatic request.")

    selected = metadata.get("selected_backend")
    if selected not in ORT_BACKENDS:
        raise AssertionError(f"Unexpected selected backend: {selected!r}")

    attempts = metadata.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise AssertionError("Backend attempt history is missing.")
    if attempts[-1].get("succeeded") is not True:
        raise AssertionError("The final backend attempt was not successful.")
    if attempts[-1].get("backend") != selected:
        raise AssertionError("Selected backend and final successful attempt disagree.")

    providers = set(metadata.get("available_onnxruntime_providers", []))
    if (
        "TensorrtExecutionProvider" not in providers
        and "CUDAExecutionProvider" not in providers
        and selected != "ort_cpu"
    ):
        raise AssertionError("A CPU-only ORT package did not fall back to ORT CPU.")

    # Explicit requests remain strict. A CPU-only ORT package must reject an
    # explicit CUDA request instead of silently changing it to CPU.
    if "CUDAExecutionProvider" not in providers:
        strict_output = output_dir / "strict_cuda"
        explicit_cuda = run(
            [
                str(executable),
                "--mode",
                "infer",
                "--source",
                str(image),
                "--backend",
                "ort-cuda",
                "--model",
                str(model),
                "--precision",
                "fp32",
                "--output-dir",
                str(strict_output),
            ]
        )
        if explicit_cuda.returncode == 0:
            raise AssertionError(
                "Explicit ORT CUDA unexpectedly fell back on a CPU-only runtime."
            )

    print(f"Automatic backend selected: {selected}")
    print("Automatic fallback and strict explicit-backend behavior passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
