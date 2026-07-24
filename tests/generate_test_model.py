#!/usr/bin/env python3
"""Generate a tiny deterministic ONNX model for C++ ORT integration tests."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


INPUT_SHAPE = [1, 3, 640, 640]
OUTPUT_SHAPE = [1, 300, 6]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic edge-perception test model."
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_model() -> onnx.ModelProto:
    output_values = np.zeros(OUTPUT_SHAPE, dtype=np.float32)

    # Two final, confidence-sorted detections in model-space coordinates.
    output_values[0, 0] = [100.0, 100.0, 200.0, 200.0, 0.90, 2.0]
    output_values[0, 1] = [300.0, 250.0, 500.0, 450.0, 0.80, 0.0]

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "ReduceSum",
                inputs=["images", "reduce_axes"],
                outputs=["input_sum"],
                keepdims=0,
            ),
            helper.make_node(
                "Mul",
                inputs=["input_sum", "zero"],
                outputs=["zero_from_input"],
            ),
            helper.make_node(
                "Add",
                inputs=["base_output", "zero_from_input"],
                outputs=["output0"],
            ),
        ],
        name="edge_perception_deterministic_test_model",
        inputs=[
            helper.make_tensor_value_info(
                "images",
                TensorProto.FLOAT,
                INPUT_SHAPE,
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output0",
                TensorProto.FLOAT,
                OUTPUT_SHAPE,
            )
        ],
        initializer=[
            numpy_helper.from_array(
                np.asarray([0, 1, 2, 3], dtype=np.int64),
                name="reduce_axes",
            ),
            numpy_helper.from_array(
                np.asarray(0.0, dtype=np.float32),
                name="zero",
            ),
            numpy_helper.from_array(output_values, name="base_output"),
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="edge-perception-cpp-tests",
        producer_version="1.1.0",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    model = build_model()
    onnx.save(model, output)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Generated: {output}")
    print(f"SHA-256:   {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
