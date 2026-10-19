# Backend compatibility and model contract

[Results](results/README.md) · [Setup](USAGE.md)

## Runtime routes

| Artifact | ORT CPU | ORT CUDA | ORT TensorRT EP | Native TensorRT | Fused CUDA/TensorRT |
|---|---|---|---|---|---|
| ONNX FP32 | Measured | Measured | Measured | — | — |
| ONNX FP16 | Measured | Measured | Measured | — | — |
| Canonical ONNX INT8 QDQ | Measured | Measured | Rejected by project policy | — | — |
| Legacy all-convolution ONNX INT8 | Measured | Measured | Rejected by project policy | — | — |
| TensorRT FP32 | — | — | — | Measured | Measured |
| TensorRT FP16 | — | — | — | Measured | Measured |
| TensorRT INT8 | — | — | — | Measured | Measured |

“Measured” means successful execution plus full-test quality and timing evidence,
not universal equivalence. The PyTorch checkpoint is a Python reference, not a
C++ CLI backend. ORT TensorRT can partition work across TensorRT/CUDA/CPU;
provider initialization alone does not establish node placement.

`auto` selects an ONNX execution provider with session-creation fallback.
Explicit backend requests fail visibly. Neither policy makes a TensorRT engine
portable or implements arbitrary mid-inference GPU-to-CPU recovery.

## Artifacts

| Role | Repository-relative path |
|---|---|
| Trained reference | `runs/train/yolo26m_bdd100k/weights/best.pt` |
| ONNX reference/deployment | `models/onnx/yolo26m_bdd100k_{fp32,fp16,int8}.onnx` |
| Native engines | `models/engine/yolo26m_bdd100k_{fp32,fp16,int8}.engine` |
| Legacy quantization comparison | `models/onnx/yolo26m_bdd100k_allconv_int8.onnx` |
| Accepted INT8 registry | `models/int8_active.json` |

Braces above denote separate filenames, not literal names. Artifact hashes are
in [provenance.json](results/provenance.json). FP32 allconv/convonly aliases are
byte-identical in the measured inventory and are counted once. Superseded
artifacts are archived outside active discovery under `models/archive/`.

Weights/engines are not distributed with the repository. Re-exported artifacts
need their own hashes and measurements.

## Detector contract

The deployed model uses batch-1, square 640×640, planar NCHW input and NMS-free
`[1,300,6]` output rows containing boxes, confidence and class ID. Preprocessing
letterboxes with padding 114, converts BGR to RGB and normalizes input. Both
CPU and CUDA paths restore boxes to original-image coordinates.

The pipeline validates tensor shapes/types and buffer sizes; there is no extra
NMS pass applied to this already-decoded output. A generic raw YOLO head or
arbitrary model is not a drop-in replacement.

Native engine bindings may be FP32 or FP16. INT8 describes internal mixed
precision, not necessarily integer input/output tensors. Raw plans and validated
Ultralytics metadata-prefixed engines are accepted; model bytes are not modified.

## Measured same-artifact parity

| C++ path | FP32 | FP16 | Canonical INT8 | Legacy INT8 |
|---|---:|---:|---:|---:|
| ORT CPU | 64/64 | 57/64 | 1/64 | 1/64 |
| ORT CUDA | 64/64 | 64/64 | 64/64 | 64/64 |
| ORT TensorRT EP | 64/64 | 48/64 | Unsupported | Unsupported |
| Native TensorRT | 64/64 | 64/64 | 64/64 | — |
| Fused TensorRT | 64/64 | 64/64 | 64/64 | — |

These compare each path with the same artifact through Python on 64 images.
FP16/INT8 comparison against the original PyTorch checkpoint is a separate,
stricter equivalence question with retained failures.
[Gate definitions](results/fp16-correctness.md#comparison-gates).

Full-test C++ native/fused mAP50–95 is 34.0161 (FP32), 34.0120 (FP16) and
33.2609 (INT8). Full quality results—not empty-output parity alone—establish
that the active INT8 artifacts produce useful detections on this dataset.

## Platform boundaries

The published evidence is native Windows on the recorded RTX 2060 stack.
Engines must be built for their target GPU/OS/runtime. Linux CPU CI is configured;
[container build and execution checks](DOCKER.md#validation-boundary) are
documented separately. Full-dataset Linux GPU quality/performance is not
established by the Windows evidence.
