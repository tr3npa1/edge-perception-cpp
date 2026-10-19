# Correctness and parity

[All results](README.md) · [FP16 performance](fp16-matrix.md)

## Evidence scope

The final all-model run uses 64 fixed validation images for strict comparisons
and 20,000 prepared test images for quality. These are independent checks.
All current numbers below come from that run; earlier diagnostic samples are
not mixed into it.

| Candidate | Reference | Images passing |
|---|---|---:|
| Native TensorRT FP16 | Python, same engine | 64/64 |
| Fused TensorRT FP16 | Python, same engine | 64/64 |
| Fused TensorRT FP16 | Native, same engine | 64/64 |
| Native/fused TensorRT FP16 | PyTorch | 61/64 |
| Python ONNX FP32 | PyTorch | 64/64 |
| Python ONNX FP16 | PyTorch | 46/64 |
| ONNX FP16 / ORT CPU | Python, same artifact | 57/64 |
| ONNX FP16 / ORT CUDA | Python, same artifact | 64/64 |
| ONNX FP16 / ORT TensorRT | Python, same artifact | 48/64 |

FP32 and INT8 native/fused/Python TensorRT same-engine comparisons also pass
64/64. Across all 42 comparisons, 22 pass and 20 fail. Every result remains in
[parity.json](parity.json); a failed gate is not waived because mAP is close.

## Root causes and corrections

The fused resize originally retained fractional interpolated pixels while
OpenCV resized into uint8 before normalization. CUDA packed BGR/RGB
preprocessing now reproduces fixed-point interpolation and uint8 rounding,
including the exact 2× downsampling case, then follows the CPU normalization
operation order. Color order, letterbox geometry, layout and FP16 conversion
are covered by focused tests.

The original PyTorch comparator could use rectangular input while static exports
required square 640×640. It now explicitly requests `rect=False`. This fixes a
comparison-methodology error; it does not make lower precision identical to
PyTorch. Detection decoding and original-image coordinate restoration use the
same intended semantics, without an additional NMS pass.

Residual FP16/PyTorch score/box differences remain. Per-image diagnostics are
retained in the raw comparison reports; aggregate mAP does not establish strict
equivalence.

## Comparison gates

| Gate | Match IoU | Minimum image mean IoU | Max box difference | Max confidence difference | Max count difference |
|---|---:|---:|---:|---:|---:|
| Same artifact | 0.95 | 0.99 | 1.5 px | 0.01 | 0 |
| PyTorch / cross-precision | 0.90 | 0.98 | 2.0 px | 0.02 | 1 |

Predictions use confidence 0.25. Passing requires every image to satisfy the
configured gate. Empty predictions can be valid, but agreement on empty output
alone cannot establish a useful detector. Quality evaluation uses confidence
0.001 and ground truth separately.

## Full-pipeline quality

| FP16 path / reference | mAP50 | mAP50–95 |
|---|---:|---:|
| PyTorch | 60.0549 | 34.0161 |
| Python TensorRT | 60.0480 | 34.0144 |
| Native C++ TensorRT | 60.0445 | 34.0120 |
| Fused C++ TensorRT | 60.0445 | 34.0120 |

Scores use a 0–100 scale. The small Python/C++ metric difference is retained;
this report does not claim an isolated explanation for it. Native/fused quality
is measured directly on the entire test split, not inferred from artifact mAP.

## Regression coverage

The benchmark's NVIDIA preparation passed all five registered CTest targets:
GPU FP16 correctness, native unit tests, ORT CPU integration, automatic-backend
CLI and Python parity units. INT8 GPU regression work also completed.

Focused fixtures cover resize/rounding, color and tensor layout, custom
normalization/padding, non-contiguous inputs, decoding and coordinate restoration,
graph-on/off determinism, output reuse, native worker concurrency, cancellation,
input ownership, out-of-order collection, backpressure and stale-ticket rejection.

Portable CTest does not automatically run real GPU fixtures. Configure
`EDGE_GPU_TEST_ENGINE` and `EDGE_GPU_TEST_IMAGE` to register that target.
Tests validate their named fixtures; they do not prove universal equivalence
for every image, input API or runtime version.
