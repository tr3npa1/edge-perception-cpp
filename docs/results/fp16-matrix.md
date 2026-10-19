# FP16 native/fused comparison

[Full evidence](README.md) · [Correctness](fp16-correctness.md) · [Protocol](../BENCHMARKING.md)

All rows use the same TensorRT FP16 engine, batch 1, 640×640, 128 fixed decoded
images, 50 warmups and three repetitions on the RTX 2060. Sequential tests
measure 300 frames; streaming tests measure 1,000. Accuracy is measured through
each C++ pipeline on 20,000 images: mAP50 60.0445, mAP50–95 34.0120.

## Sequential latency

| Path | Mean | p50 | p95 |
|---|---:|---:|---:|
| Native | 5.271 ms | 5.264 ms | 5.339 ms |
| Fused | 5.233 ms | 5.155 ms | 5.407 ms |

Fused mean latency is 0.71% lower, but its p95 is higher. This small difference
does not establish a general latency advantage.

## Streaming throughput

| Path | Depth | Median FPS | Run minimum–maximum | FPS CV |
|---|---:|---:|---:|---:|
| Native | 1 | 189.008 | 176.797–190.031 | 3.97% |
| Fused | 1 | 192.160 | 191.119–193.462 | 0.61% |
| Native | 3 | 254.615 | 236.540–256.182 | 4.38% |
| Fused | 3 | 245.906 | 240.167–247.921 | 1.64% |
| Fused | 2 | 229.891 | 224.981–230.833 | 1.37% |

Both equal-depth comparisons are complete. Native has higher median depth-3
throughput in this run. Depth is a concurrency limit, not a promised speedup.
Native depth 3 recorded three active workers in each repetition; GPU kernel
overlap is not independently established by that host-side counter.

## CUDA graph ablation

| Fused path | Sequential mean | Depth-3 FPS |
|---|---:|---:|
| Graphs enabled | 5.233 ms | 245.906 |
| Graphs disabled | 5.418 ms | 247.057 |

Graphs improve the measured sequential mean, not every streaming metric.
Streaming per-frame residence is distinct from sequential latency; the complete
residence distributions are in [performance.json](performance.json).

All values are medians of per-run statistics, not pooled distributions.
Runs resumed across interruptions, and clocks were not locked. No statistical
significance claim is made. CPU preprocessing remains in the native timing;
CUDA preprocessing remains in the fused timing. Neither includes decode,
drawing, model creation or disk output.
