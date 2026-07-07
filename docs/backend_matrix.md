# Backend Matrix

This document defines the deployment-backend truth table for `edge-perception-cpp`.

The project does **not** try to force every artifact into every runtime. Instead, it evaluates the backend combinations that are technically meaningful and documents the one known unsupported combination.

## Artifact families

| Artifact family | File pattern | Purpose |
|---|---|---|
| PyTorch checkpoint | `runs/train/yolo26m_bdd100k/weights/best.pt` | Training baseline and quality reference |
| ONNX FP32 | `models/onnx/yolo26m_bdd100k_fp32.onnx` | Portable full-precision ONNX deployment |
| ONNX FP16 | `models/onnx/yolo26m_bdd100k_fp16.onnx` | Mixed-precision ONNX deployment |
| ONNX INT8 QDQ | `models/onnx/yolo26m_bdd100k_int8.onnx` | ONNX Runtime quantized deployment experiments |
| Native TensorRT engine FP32 | `models/engine/yolo26m_bdd100k_fp32.engine` | Direct TensorRT FP32 baseline |
| Native TensorRT engine FP16 | `models/engine/yolo26m_bdd100k_fp16.engine` | Direct TensorRT FP16 optimized baseline |
| Native TensorRT engine INT8 | `models/engine/yolo26m_bdd100k_int8.engine` | Direct TensorRT INT8 optimized baseline |
| Native TensorRT engine INT8 traincalib | `models/engine/yolo26m_bdd100k_int8_traincalib.engine` | Direct TensorRT INT8 with train-split calibration |

## Supported backend checks

| Artifact | ORT CPU | ORT CUDA | ORT TensorRT EP | Native TensorRT runtime |
|---|---:|---:|---:|---:|
| PyTorch `best.pt` | N/A | N/A | N/A | N/A |
| ONNX FP32 | Required | Required | Required | N/A |
| ONNX FP16 | Measured | Required | Required | N/A |
| ONNX INT8 QDQ | Required | Required | Skipped by design | N/A |
| TensorRT engine FP32 | N/A | N/A | N/A | Required |
| TensorRT engine FP16 | N/A | N/A | N/A | Required |
| TensorRT engine INT8 | N/A | N/A | N/A | Required |
| TensorRT engine INT8 traincalib | N/A | N/A | N/A | Optional/Recommended |

## Known unsupported combination

```text
ONNX INT8 QDQ + ONNX Runtime TensorRT Execution Provider
```

This combination is skipped for the current YOLO26M graph.

During local debugging, the saved ONNX graph could pass graph-level checks, but ORT TensorRT EP still emitted TensorRT parser failures involving generated `Conv_bias_dq` nodes. That indicates a backend-conversion compatibility issue rather than a training or dataset issue.

The correct TensorRT INT8 path for this project is:

```text
PyTorch best.pt
→ native TensorRT INT8 engine export/calibration
→ native TensorRT runtime / Ultralytics engine validation
```

## Why this is still production-quality

Production deployment work is not about making every theoretical runtime combination work. It is about:

1. choosing valid runtime paths,
2. checking provider fallback explicitly,
3. documenting unsupported combinations,
4. validating accuracy per artifact,
5. benchmarking end-to-end latency fairly.

This matrix is the contract used by `training/evaluate.py`, the README, and the future C++ benchmark suite.
