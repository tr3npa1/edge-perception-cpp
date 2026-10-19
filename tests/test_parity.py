"""Deterministic parity regression tests for Python and native C++ outputs."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training.evaluate import (
    Detection,
    compare_detection_sets,
    parity_thresholds,
    predict_one_image,
)


DEFAULT_THRESHOLDS = {
    "match_iou_threshold": 0.90,
    "min_mean_iou": 0.98,
    "max_box_abs_diff": 2.0,
    "max_conf_abs_diff": 0.02,
    "max_count_diff": 0,
}


def detection(
    class_id: int,
    confidence: float,
    box: list[float],
) -> Detection:
    return Detection(
        class_id=class_id,
        confidence=confidence,
        xyxy=box,
    )


def compare(
    reference: list[Detection],
    artifact: list[Detection],
    **overrides: float | int,
):
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(overrides)

    return compare_detection_sets(
        image_path=Path("synthetic.jpg"),
        reference_detections=reference,
        artifact_detections=artifact,
        **thresholds,
    )


def test_exact_detections_pass() -> None:
    reference = [
        detection(2, 0.95, [10.0, 20.0, 100.0, 120.0]),
        detection(0, 0.80, [150.0, 40.0, 210.0, 180.0]),
    ]

    result = compare(reference, list(reference))

    assert result.passed
    assert result.matched_detection_count == 2
    assert result.unmatched_reference_count == 0
    assert result.unmatched_artifact_count == 0
    assert result.max_box_abs_diff == pytest.approx(0.0)
    assert result.max_conf_abs_diff == pytest.approx(0.0)


def test_small_numeric_drift_passes() -> None:
    reference = [
        detection(2, 0.95, [10.0, 20.0, 100.0, 120.0]),
    ]
    artifact = [
        detection(2, 0.94, [10.25, 19.75, 100.25, 120.25]),
    ]

    result = compare(reference, artifact)

    assert result.passed
    assert result.max_box_abs_diff == pytest.approx(0.25)
    assert result.max_conf_abs_diff == pytest.approx(0.01)


def test_class_mismatch_fails() -> None:
    reference = [
        detection(2, 0.95, [10.0, 20.0, 100.0, 120.0]),
    ]
    artifact = [
        detection(3, 0.95, [10.0, 20.0, 100.0, 120.0]),
    ]

    result = compare(reference, artifact)

    assert not result.passed
    assert result.matched_detection_count == 0
    assert result.unmatched_reference_count == 1
    assert result.unmatched_artifact_count == 1


def test_excessive_coordinate_drift_fails() -> None:
    reference = [
        detection(2, 0.95, [10.0, 20.0, 100.0, 120.0]),
    ]
    artifact = [
        detection(2, 0.95, [15.0, 25.0, 105.0, 125.0]),
    ]

    result = compare(
        reference,
        artifact,
        match_iou_threshold=0.50,
        min_mean_iou=0.50,
    )

    assert not result.passed
    assert result.max_box_abs_diff == pytest.approx(5.0)


def test_excessive_confidence_drift_fails() -> None:
    reference = [
        detection(2, 0.95, [10.0, 20.0, 100.0, 120.0]),
    ]
    artifact = [
        detection(2, 0.80, [10.0, 20.0, 100.0, 120.0]),
    ]

    result = compare(reference, artifact)

    assert not result.passed
    assert result.max_conf_abs_diff == pytest.approx(0.15)


def test_missing_detection_fails() -> None:
    reference = [
        detection(2, 0.95, [10.0, 20.0, 100.0, 120.0]),
        detection(0, 0.80, [150.0, 40.0, 210.0, 180.0]),
    ]

    result = compare(reference, reference[:1])

    assert not result.passed
    assert result.unmatched_reference_count == 1


def test_both_empty_pass() -> None:
    result = compare([], [])

    assert result.passed
    assert result.matched_detection_count == 0
    assert result.mean_matched_iou is None
    assert result.failure_reasons == []
    assert result.matched_detections == []


def test_prediction_uses_same_square_input_for_all_backends() -> None:
    # Model.predict normally injects rect=True. For a 720x1280 image that gives
    # PyTorch 384x640, while a static exported model still consumes 640x640.
    # Do not let backend-specific defaults invalidate an equivalence check.
    for backend in ("pytorch", "onnx", "engine"):
        model = SimpleNamespace(predict=Mock(return_value=[]))
        assert predict_one_image(
            model=model,
            image_path=Path(f"{backend}.jpg"),
            imgsz=640,
            conf=0.25,
            iou=0.7,
            max_det=300,
            device="cpu",
        ) == []
        model.predict.assert_called_once_with(
            source=f"{backend}.jpg",
            imgsz=640,
            conf=0.25,
            iou=0.7,
            max_det=300,
            device="cpu",
            rect=False,
            verbose=False,
        )


def test_parity_metadata_records_square_preprocessing() -> None:
    args = SimpleNamespace(
        imgsz=640,
        parity_conf=0.25,
        parity_iou=0.7,
        max_det=300,
        parity_match_iou=0.90,
        parity_min_mean_iou=0.98,
        parity_max_box_abs_diff=2.0,
        parity_max_conf_abs_diff=0.02,
        parity_max_count_diff=1,
    )
    assert parity_thresholds(args)["preprocessing"] == {"imgsz": 640, "rect": False}


def test_failure_diagnostics_retain_exact_detection_pairs() -> None:
    reference = [
        detection(2, 0.95, [10.0, 20.0, 100.0, 120.0]),
        detection(0, 0.80, [150.0, 40.0, 210.0, 180.0]),
    ]
    artifact = [
        detection(3, 0.80, [150.0, 40.0, 210.0, 180.0]),
        detection(2, 0.70, [10.5, 20.0, 100.0, 120.0]),
    ]

    result = compare(reference, artifact)

    assert not result.passed
    assert result.failure_reasons == ["unmatched_detections", "max_conf_abs_diff"]
    assert result.unmatched_reference_detections == reference[1:]
    assert result.unmatched_artifact_detections == artifact[:1]
    pair = result.matched_detections[0]
    assert (pair.reference_index, pair.artifact_index) == (0, 1)
    assert pair.reference == reference[0]
    assert pair.artifact == artifact[1]
    assert pair.box_abs_diffs == [0.5, 0.0, 0.0, 0.0]
    assert pair.conf_abs_diff == pytest.approx(0.25)
    assert json.loads(json.dumps(asdict(result)))["matched_detections"][0]["reference"] == asdict(reference[0])


def test_count_tolerance_does_not_hide_no_matching_detections() -> None:
    reference = [detection(2, 0.95, [10.0, 20.0, 100.0, 120.0])]

    result = compare(reference, [], max_count_diff=1)

    assert not result.passed
    assert result.matched_detection_count == 0
    assert result.unmatched_reference_detections == reference
    assert result.failure_reasons == [
        "mean_matched_iou", "max_box_abs_diff", "max_conf_abs_diff"
    ]


def test_count_tolerance_does_not_ignore_confidence_drift() -> None:
    reference = [detection(2, 0.95, [10.0, 20.0, 100.0, 120.0])]
    artifact = [detection(2, 0.70, [10.0, 20.0, 100.0, 120.0])]

    result = compare(reference, artifact, max_count_diff=1)

    assert not result.passed
    assert result.failure_reasons == ["max_conf_abs_diff"]


def read_cpp_detections(path: Path) -> list[Detection]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1, "Single-image inference must write one JSONL row."

    payload = json.loads(lines[0])
    return [
        Detection(
            class_id=int(item["class_id"]),
            confidence=float(item["confidence"]),
            xyxy=[float(value) for value in item["box"]],
        )
        for item in payload["detections"]
    ]


def required_integration_paths() -> tuple[Path, Path, Path, Path]:
    variables = {
        "EDGE_TEST_WEIGHTS": os.environ.get("EDGE_TEST_WEIGHTS"),
        "EDGE_TEST_ONNX_MODEL": os.environ.get("EDGE_TEST_ONNX_MODEL"),
        "EDGE_TEST_IMAGE": os.environ.get("EDGE_TEST_IMAGE"),
        "EDGE_TEST_EXECUTABLE": os.environ.get("EDGE_TEST_EXECUTABLE"),
    }

    missing = [name for name, value in variables.items() if not value]
    if missing:
        pytest.skip(
            "Real-model integration artifacts are not configured: "
            + ", ".join(missing)
        )

    paths = tuple(Path(value).resolve() for value in variables.values() if value)
    for path in paths:
        if not path.is_file():
            pytest.fail(f"Configured integration artifact does not exist: {path}")

    return paths  # type: ignore[return-value]


@pytest.mark.integration
def test_pytorch_onnx_cpp_real_model_parity(tmp_path: Path) -> None:
    weights, onnx_model, image, executable = required_integration_paths()

    confidence = float(os.environ.get("EDGE_TEST_CONFIDENCE", "0.25"))
    iou = float(os.environ.get("EDGE_TEST_IOU", "0.7"))
    maximum_detections = int(os.environ.get("EDGE_TEST_MAX_DETECTIONS", "300"))
    device = os.environ.get("EDGE_TEST_DEVICE", "cpu")

    from ultralytics import YOLO

    pytorch_model = YOLO(str(weights))
    onnx_runtime_model = YOLO(str(onnx_model))

    reference = predict_one_image(
        model=pytorch_model,
        image_path=image,
        imgsz=640,
        conf=confidence,
        iou=iou,
        max_det=maximum_detections,
        device=device,
    )
    onnx_detections = predict_one_image(
        model=onnx_runtime_model,
        image_path=image,
        imgsz=640,
        conf=confidence,
        iou=iou,
        max_det=maximum_detections,
        device=device,
    )

    python_result = compare_detection_sets(
        image_path=image,
        reference_detections=reference,
        artifact_detections=onnx_detections,
        match_iou_threshold=0.90,
        min_mean_iou=0.98,
        max_box_abs_diff=2.0,
        max_conf_abs_diff=0.02,
        max_count_diff=1,
    )
    assert python_result.passed, python_result

    output_dir = tmp_path / "cpp"
    command = [
        str(executable),
        "--mode",
        "infer",
        "--source",
        str(image),
        "--backend",
        "auto",
        "--model",
        str(onnx_model),
        "--precision",
        "fp32",
        "--confidence",
        str(confidence),
        "--max-detections",
        str(maximum_detections),
        "--output-dir",
        str(output_dir),
    ]

    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout

    cpp_detections = read_cpp_detections(output_dir / "detections.jsonl")
    cpp_result = compare_detection_sets(
        image_path=image,
        reference_detections=onnx_detections,
        artifact_detections=cpp_detections,
        match_iou_threshold=0.95,
        min_mean_iou=0.99,
        max_box_abs_diff=1.5,
        max_conf_abs_diff=0.01,
        max_count_diff=0,
    )
    assert cpp_result.passed, cpp_result

    selection = json.loads(
        (output_dir / "backend_selection.json").read_text(encoding="utf-8")
    )
    assert selection["requested_backend"] == "auto"
    assert selection["selected_backend"] in {
        "ort_tensorrt",
        "ort_cuda",
        "ort_cpu",
    }
