# edge-perception-cpp

C++17 object detection with ONNX Runtime and TensorRT, including a CUDA
preprocessing/postprocessing pipeline and bounded asynchronous execution.
The repository covers BDD100K dataset preparation, YOLO26M training and export,
FP16/INT8 deployment, correctness tests, and reproducible performance evaluation.

[Setup and usage](docs/USAGE.md) · [Results](docs/results/README.md) ·
[Benchmark protocol](docs/BENCHMARKING.md) · [Architecture](docs/OPTIMIZATION_AUDIT.md) ·
[Backend compatibility](docs/backend_matrix.md)

Source license: [AGPL-3.0](LICENSE), with a limited
[NVIDIA linking permission](LICENSE-EXCEPTION) for the original C++ runtime.
Dependencies and example imagery retain their own [terms](THIRD_PARTY_NOTICES.md).

## Measured results

YOLO26M on an **RTX 2060 6 GB**, Ryzen 5 3600, native Windows; batch 1,
640×640 input. Accuracy uses the full prepared **20,000-image test split**.
Latency and throughput are separate measurements, summarized across three runs.

| C++ TensorRT path | mAP50–95 ↑ | Sequential mean ↓ | Sequential p95 ↓ | Streaming FPS, depth 1 ↑ | Streaming FPS, depth 3 ↑ |
|---|---:|---:|---:|---:|---:|
| Native FP32 | 34.0161 | 17.107 ms | 17.781 ms | 57.99 | 64.49 |
| Fused FP32 | 34.0161 | 16.884 ms | 17.523 ms | 58.79 | 66.08 |
| Native FP16 | 34.0120 | 5.271 ms | 5.339 ms | 189.01 | 254.62 |
| Fused FP16 | 34.0120 | 5.233 ms | 5.407 ms | 192.16 | 245.91 |
| Native INT8 | 33.2609 | 5.185 ms | 5.246 ms | 193.30 | 278.79 |
| Fused INT8 | 33.2609 | 4.929 ms | 4.973 ms | 202.98 | 278.81 |

The PyTorch quality reference is **34.0161 mAP50–95**. Scores use a 0–100 scale.
The C++ accuracy rows measure their actual preprocessing, inference and
postprocessing paths—not quality inferred from Python exports.

What the measurements support:

- Native FP16 achieves **3.95×** native FP32 depth-3 throughput, with a
  **0.0041-point** mAP50–95 reduction.
- Fused INT8 achieves **13.38%** more depth-3 throughput than fused FP16, with a
  **0.7511-point** mAP50–95 reduction.
- Fusion is not a universal improvement: native FP16 has higher median
  depth-3 throughput than fused FP16; native/fused INT8 are effectively tied
  at that depth. Small timing differences overlap run-to-run variability.
- ONNX INT8 through ORT CPU/CUDA is slower than the corresponding FP32/FP16
  paths on this machine. INT8 speed benefits here require the TensorRT engines.

Timing starts with a decoded BGR image and ends with detection objects.
It excludes model loading, image decoding, drawing and disk output. Depth 3
allows multiple frames in flight; streaming FPS is not reciprocal single-frame
latency. Clocks were not locked and interrupted runs were resumed with identity
checks. These are workstation measurements, not guaranteed deployment rates.

[All 24 quality paths, 56 timing configurations, ranges and failures](docs/results/README.md) ·
[Interpretation and trade-offs](docs/RESULTS_ANALYSIS.md)

## Correctness and validation

Native, fused and Python TensorRT match **64/64 images using the same engine**
for FP32, FP16 and INT8. Direct native/fused comparisons also pass 64/64.
Full-test native/fused quality agrees at the precision reported above.

Strict PyTorch equivalence is a different gate: TensorRT FP16 passes **61/64**
and INT8 **1/64**. ONNX FP16 CPU/ORT-TensorRT and ONNX INT8 CPU also retain
same-artifact failures. Close dataset mAP does not waive these differences.

The completed suite contains 120 coordinator jobs and 202 performance
executions: 200 pass; two INT8 ONNX / ORT TensorRT combinations are intentionally
rejected by project routing policy. All 16 supported C++ paths have complete
measurements. [Exact parity gates and results](docs/results/fp16-correctness.md).

## Runtime design

```text
Image / video / camera
        │
        ▼
Decoded BGR frame
        ├─ CPU letterbox + tensor packing → ORT / native TensorRT → CPU decode
        └─ CUDA resize + letterbox + packing → TensorRT → CUDA decode
                                                               │
                                                               ▼
                                                        Detection objects
```

- **Explicit backends:** ORT CPU, ORT CUDA, ORT TensorRT EP, native TensorRT,
  and fused CUDA/TensorRT. Explicit requests fail visibly; `auto` provides
  ONNX session-creation fallback.
- **Fixed model contract:** batch-1 NCHW input and NMS-free `[1,300,6]` output;
  validated shapes/types, coordinate restoration and bounded detection storage.
- **Asynchronous ownership:** fused slots own contexts, buffers, streams/events
  and graph state. Native streaming workers use independent execution resources;
  depth-3 tests record three active workers, not simulated concurrency.
- **Transfer and allocation control:** persistent buffers, pinned host staging,
  compact detection transfers, and optional CUDA graph capture.
- **Input interfaces:** ordinary OpenCV frames plus pinned host and
  device-resident BGR/RGB/NV12 views. Published timing uses OpenCV frames;
  external decoder integration is not part of the benchmark.

[Execution and ownership details](docs/OPTIMIZATION_AUDIT.md) ·
[Fused pipeline API](docs/GPU_PIPELINE_USAGE.md) ·
[INT8 calibration and deployment](docs/INT8_DEPLOYMENT.md)

## Build and run

Model weights, datasets and SDK binaries are **not included**. Prepare an
exported model and configure dependency paths using [USAGE.md](docs/USAGE.md).
The portable build needs a C++17 compiler, CMake 3.24+, Ninja, OpenCV,
ONNX Runtime C++ and Python test dependencies.

From a configured Windows x64 developer shell:

```powershell
cmake --preset cpu-release
cmake --build --preset cpu-release
ctest --preset cpu-release --output-on-failure

.\build\cpu-release\bin\Release\edge_perception.exe `
    --mode infer `
    --source samples\frame.jpg `
    --backend ort-cpu `
    --model models\onnx\yolo26m_bdd100k_fp32.onnx `
    --precision fp32 `
    --output-dir outputs\image
```

The setup guide includes Linux CPU builds, NVIDIA SDK/runtime configuration,
dataset preparation, export, directory/video inference and tests.

For NVIDIA Linux containers, see [Docker deployment](docs/DOCKER.md): local
image build, target-specific engine generation, mounted inputs/outputs and
container validation. Weights and datasets are not embedded in the image.

Published container: `tr3npa1/edge-perception-cpp:1.1.0-20261019`
([Docker Hub](https://hub.docker.com/r/tr3npa1/edge-perception-cpp)).
The `latest` tag follows this release; use the versioned tag for reproducibility.

On the configured NVIDIA workstation, with all artifacts prepared:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_all_models.ps1
```

This runs the full evaluation and can take many hours. It does not train,
publish or invoke Docker. [Prerequisites, smoke mode and resume](docs/BENCHMARKING.md).

## Example output

![Fused TensorRT FP16 detections on a BDD100K image](docs/results/example-1.jpg)

First saved image from the final fused FP16 smoke run; not selected for visual
quality. The example illustrates output, not an accuracy guarantee.
[Native FP16 and INT8 comparisons](docs/results/README.md#annotated-outputs).
BDD100K imagery remains subject to its dataset terms.

## Repository map

| Location | Responsibility |
|---|---|
| `include/`, `src/` | C++ interfaces, inference backends, CUDA pipeline, CLI |
| `data/`, `training/` | Dataset validation, training, export and quality evaluation |
| `tests/` | Unit, CLI, parity, GPU lifecycle and workflow regressions |
| `tools/` | Calibration and resumable benchmark orchestration |
| `docs/results/` | Curated metrics, failures, examples and provenance |
| `docs/scripts/` | Evidence-to-documentation export; no inference or upload |
| `.github/workflows/` | Windows/Linux CPU portability CI configuration |

Raw predictions, model files, caches and benchmark runs stay local and ignored.
The public evidence bundle includes hashes and compact results, not model downloads.

## Scope and limitations

- TensorRT engines must be built for the target platform/runtime; the measured
  Windows engines are not Linux container artifacts.
- Docker build and targeted CPU/GPU execution are validated separately in the
  [container guide](docs/DOCKER.md#validation-boundary). Full-dataset Linux GPU
  accuracy and performance remain unmeasured; the tables above are Windows results.
- CPU CI configuration is not evidence of a fresh remote CI pass.
- No QAT training, camera-to-display benchmark, tracking, planning or
  automotive safety certification is provided.
