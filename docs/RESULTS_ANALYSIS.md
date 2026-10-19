# Results and engineering trade-offs

[Complete evidence](results/README.md) · [Protocol](BENCHMARKING.md)

## Scope

The measured configuration is YOLO26M, batch 1, 640×640, on an RTX 2060 6 GB
and Ryzen 5 3600 under Windows. The completed suite evaluates 24 full-test
quality paths, 42 strict comparisons and 202 performance executions.
All 16 supported C++ paths have complete timing repetitions.

The public results come from one source/artifact-identified run,
`all_20261018_151027_308`, resumed after interruptions. Earlier benchmarks
remain local historical evidence and are not mixed into the current tables.

## Precision changes

| Same-path comparison | Observed result | Accuracy cost |
|---|---|---|
| Native FP32 → FP16, depth 3 | 64.49 → 254.62 FPS; 3.95× throughput | 0.0041 mAP50–95 points |
| Fused FP16 → INT8, sequential | 5.233 → 4.929 ms; 5.82% lower mean | 0.7511 mAP50–95 points |
| Fused FP16 → INT8, depth 3 | 245.91 → 278.81 FPS; 13.38% higher throughput | Same accuracy trade-off |

FP16 retains essentially the baseline aggregate quality on this test split.
INT8 offers a measured latency/throughput benefit through TensorRT but is not
lossless. These are configuration comparisons, not proof that one source-code
optimization caused the whole improvement.

INT8 here is explicit-QDQ mixed precision: backbone/neck convolutions are
quantized, while sensitive head/output operations remain floating point.
External input/output bindings are floating point; it is not an all-integer pipeline.

## Native versus fused execution

| Precision | Native / fused sequential mean | Native / fused depth-3 FPS |
|---|---:|---:|
| FP32 | 17.107 / 16.884 ms | 64.49 / 66.08 |
| FP16 | 5.271 / 5.233 ms | 254.62 / 245.91 |
| INT8 | 5.185 / 4.929 ms | 278.79 / 278.81 |

Fused FP16 reduces mean latency by only 0.038 ms and has a slightly higher
p95 (5.407 versus 5.339 ms). Its median depth-3 throughput is 3.42% lower.
Fused INT8 has 4.93% lower sequential mean latency, but depth-3 throughput is
effectively tied. Fusion therefore does not establish a universal speedup.

Depth-3 FPS ranges across the three repetitions:

| Path | Minimum–maximum FPS |
|---|---:|
| Native FP16 | 236.54–256.18 |
| Fused FP16 | 240.17–247.92 |
| Native INT8 | 256.31–287.82 |
| Fused INT8 | 267.98–289.17 |

Run ranges overlap for several comparisons. Clocks were not locked and the
machine was not a controlled laboratory system. No statistical significance
claim is made.

## ONNX Runtime and INT8

ORT CUDA FP16 measures 12.404 ms sequential mean latency; FP32 measures
29.191 ms. The canonical INT8 ONNX measures 42.921 ms and 32.9093 mAP50–95.
On CPU it measures 432.156 ms, versus 253.110 ms for FP32. Quantized artifact
size/type alone does not establish efficient execution on a particular provider.

The legacy all-convolution INT8 model remains an explicit comparison:
ORT CUDA measures 46.015 ms and 28.6131 mAP50–95. The canonical head-excluded
PTQ model improves quality, but neither ONNX INT8 path is the performance winner
on this workstation.

ORT TensorRT FP16 measures 5.231 ms sequential mean, but it is a provider-managed
ONNX session, not the native engine path. Its full-pipeline mAP50–95 is 33.9276
and same-artifact parity is 48/64. Provider lists do not prove every node executes
on TensorRT; CPU/CUDA partitioning is permitted.

Two INT8 ONNX / ORT TensorRT smoke checks are rejected by project policy.
These are expected unsupported combinations, not failed native INT8 engines.

## Accuracy, strict parity and execution are different

1. **Execution** establishes that a path loads and returns valid output.
2. **Same-artifact parity** compares deployment behavior against that exact
   artifact in Python.
3. **PyTorch parity** measures changes relative to the original checkpoint.
4. **Dataset mAP** measures detection quality against ground-truth labels.

Native/fused/Python TensorRT pass 64/64 same-engine comparisons for all three
precisions. Native/fused full-test quality agrees at the reported precision.
TensorRT FP16 nevertheless passes only 61/64 against PyTorch, and INT8 1/64.
Small aggregate mAP changes can coexist with per-image confidence/count/box
differences; the strict failures are not waived.

Of 42 comparisons, 22 pass and 20 fail. Same-artifact failures remain for ONNX
FP16 on CPU (57/64) and ORT TensorRT (48/64), and both INT8 ONNX variants on CPU
(1/64). These are not evidence of universal equivalence across providers.
[Full gates and results](results/fp16-correctness.md).

## CUDA graph ablation

| Fused FP16 | Graphs off | Graphs on |
|---|---:|---:|
| Sequential mean | 5.418 ms | 5.233 ms |
| Depth-3 streaming FPS | 247.06 | 245.91 |

Graph capture reduces the measured sequential mean but does not improve every
metric. The small throughput difference should be read alongside run ranges.
Streaming frame residence is reported separately from sequential latency.

## Defects addressed and remaining boundaries

The CUDA resize path now reproduces the CPU's uint8 interpolation rounding
before normalization; Python parity uses square rather than rectangular input.
Both TensorRT loaders validate and remove metadata envelopes before raw plan
deserialization. INT8 calibration matches deployment letterboxing and avoids
quantizing the mixed-unit detection output that previously zeroed confidence
values. These are correctness fixes, not reasons to hide residual failures.

Outstanding boundaries:

- Same-artifact ONNX/provider differences and strict cross-precision failures
  remain documented rather than weakened into passes.
- Tiny Python/C++ mAP differences are retained; no unverified exact-cause claim
  is made for them.
- Docker image execution, Linux GPU results and fresh remote CI remain outside
  this native Windows evidence.
- Device-resident input APIs are implemented but not separately benchmarked here.
- No camera-to-display latency, multi-GPU scaling, long-duration service test,
  QAT training or automotive safety certification is established.
