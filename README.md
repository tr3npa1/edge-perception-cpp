# edge-perception-cpp

[![CPU portability](https://github.com/tr3npa1/edge-perception-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/tr3npa1/edge-perception-cpp/actions/workflows/ci.yml)

A cross-platform C++17 object-detection runtime for YOLO26M, with portable
ONNX Runtime inference and optional NVIDIA acceleration through CUDA, the ONNX
Runtime TensorRT Execution Provider, native TensorRT, and a fused CUDA/TensorRT
pipeline.

The repository covers the deployment path from BDD100K preparation and model
training to export, parity testing, native inference, benchmarking, and Linux
GPU container packaging.

## Highlights

- Portable ONNX Runtime CPU execution on Windows and Linux.
- Strict explicit backends plus automatic ONNX fallback: TensorRT EP, CUDA,
  then CPU.
- Native TensorRT and fused CUDA/TensorRT execution for NVIDIA deployments.
- Image, directory, video, and camera inference.
- Deterministic C++/Python tests and a generated ONNX integration fixture.
- PyTorch → ONNX → C++ parity testing for real model artifacts.
- Reproducible latency and streaming-throughput reports.
- CPU CI on Windows and Ubuntu.
- Multi-stage NVIDIA Docker image with build-time tests and a non-root runtime
  user.

## Scope and safety boundary

This is a performance-oriented deployment project. It is not automotive
safety-certified software, an ISO 26262 implementation, a complete autonomous
vehicle stack, or a substitute for a production perception safety case.
TensorRT engines are deployment-target artifacts and are not portable across
arbitrary operating systems, GPU architectures, CUDA/TensorRT versions, or
build settings.

## Architecture

```text
BDD100K
  ↓
data/download.py → data/dataset.py
  ↓
training/train_detector.py
  ↓
training/evaluate.py
  ↓
training/export_model.py
  ├─ ONNX FP32 / FP16 / INT8
  └─ TensorRT FP32 / FP16 / INT8
  ↓
C++17 runtime
  ├─ ONNX Runtime CPU
  ├─ ONNX Runtime CUDA
  ├─ ONNX Runtime TensorRT EP
  ├─ native TensorRT
  └─ fused CUDA preprocessing → TensorRT → CUDA postprocessing
```

The native executable separates preprocessing, inference, postprocessing, and
benchmark reporting into reusable components under `include/` and `src/`.

## Backend matrix

| CLI backend | Artifact | Requirement | Behavior |
|---|---|---|---|
| `auto` | `.onnx` | ONNX Runtime | Tries TensorRT EP, CUDA, then CPU |
| `ort-cpu` | `.onnx` | ONNX Runtime CPU | Portable, strict CPU selection |
| `ort-cuda` | `.onnx` | ORT GPU package and CUDA stack | Strict CUDA selection |
| `ort-trt` | `.onnx` | ORT TensorRT EP and TensorRT stack | Strict TensorRT EP selection |
| `native-trt` | `.engine` | Native TensorRT build | Synchronous native TensorRT |
| `fused-trt` | `.engine` | Native TensorRT plus CUDA pipeline | Fused asynchronous GPU path |

Automatic fallback applies only while creating an ONNX Runtime session. It
does not hide invalid model contracts, inference failures, malformed outputs,
or incompatible TensorRT engines. Explicit backend requests always fail rather
than silently selecting another backend.

See [docs/backend_matrix.md](docs/backend_matrix.md) for the artifact/backend
validation contract and the documented ONNX INT8 QDQ limitation.

## Model contract

The native runtime expects the exported end-to-end detector contract:

```text
input:  [1, 3, 640, 640], float32 or float16
output: [1, 300, 6], float32
row:    [x1, y1, x2, y2, confidence, class_id]
```

The exported head already produces final, confidence-ranked detections. The C++
postprocessor restores original-image coordinates, clips boxes when enabled,
and preserves output order; it does not apply a second NMS pass.

BDD100K class IDs are fixed as follows:

```text
0 person          5 bike
1 rider           6 motor
2 car             7 traffic light
3 bus             8 traffic sign
4 truck           9 train
```

## Repository layout

```text
.github/workflows/ci.yml   CPU portability CI
CMakeLists.txt             Native targets, dependencies, tests, installation
CMakePresets.json          Portable CPU and NVIDIA release profiles
Dockerfile                 Linux NVIDIA builder/runtime image

data/                      Dataset download and validation
training/                  Training, evaluation, and export entrypoints
include/                   Public C++ interfaces
src/                       C++ and CUDA implementation
tests/                     C++ tests, Python parity tests, ONNX fixture generator
docs/                      Backend and optimization notes
```

Datasets, model weights, exported artifacts, build trees, virtual environments,
experiment tracking data, benchmark output, and rendered inference output are
intentionally excluded from Git. The Docker build context is deny-by-default
and includes only paths copied by the Dockerfile.

## Prerequisites

### All native builds

- CMake 3.24 or newer
- A C++17 compiler
- OpenCV with `core`, `imgproc`, `imgcodecs`, and `videoio`
- An extracted ONNX Runtime C/C++ release package
- Ninja, unless another CMake generator is selected manually
- Python 3.10 or newer for generated/test fixtures; Python 3.11 is recommended

### NVIDIA builds

- NVIDIA driver compatible with the selected CUDA runtime
- CUDA Toolkit
- TensorRT C++ SDK
- A CUDA-supported host compiler

The validated Windows NVIDIA configuration uses CUDA 12.8, TensorRT 10.9,
ONNX Runtime GPU 1.27, and MSVC 14.39 for an RTX 2060 (`sm_75`). Other targets
must use versions and a CUDA architecture appropriate to that machine.

## Python environments

Create environments from one of the purpose-specific files:

| File | Purpose |
|---|---|
| `requirements.txt` | Backward-compatible default; includes the GPU stack |
| `requirements-cpu.txt` | CPU training/export/evaluation environment |
| `requirements-gpu.txt` | CUDA training/export/evaluation environment |
| `requirements-test.txt` | Lightweight public test environment |
| `requirements-tensorrt.txt` | GPU environment plus target-provided TensorRT bindings |

Portable test setup on Windows:

```powershell
py -3.11 -m venv .venv-test
.\.venv-test\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
python -m pip check
```

GPU development setup:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip check
$env:YOLO_AUTOINSTALL = "false"
```

`YOLO_AUTOINSTALL=false` prevents Ultralytics from replacing controlled runtime
packages while loading an artifact. TensorRT Python bindings are optional for
the native C++ runtime and should come from the target SDK or NVIDIA container.

## Dataset preparation

Download the configured BDD100K YOLO-format dataset:

```powershell
python data\download.py
```

Validate it and write project-local metadata:

```powershell
python data\dataset.py
```

The default prepared dataset file is:

```text
data/processed/bdd100k_yolo/bdd100k.yaml
```

The data scripts do not modify the original Kaggle YAML. Use `--help` on either
entrypoint for validation-only and path-override options.

## Training, evaluation, and export

Train YOLO26M:

```powershell
python training\train_detector.py `
    --data data\processed\bdd100k_yolo\bdd100k.yaml `
    --device 0
```

The default run directory is `runs/train/yolo26m_bdd100k`. Use `best.pt` for
deployment export and quality comparisons.

Evaluate a checkpoint and selected deployment artifacts:

```powershell
python training\evaluate.py `
    --weights runs\train\yolo26m_bdd100k\weights\best.pt `
    --data data\processed\bdd100k_yolo\bdd100k.yaml `
    --onnx models\onnx\yolo26m_bdd100k_fp32.onnx `
    --check-ort-cpu
```

Export FP32 and FP16 ONNX artifacts:

```powershell
python training\export_model.py `
    --weights runs\train\yolo26m_bdd100k\weights\best.pt `
    --data data\processed\bdd100k_yolo\bdd100k.yaml `
    --variants onnx_fp32 onnx_fp16
```

Use `--dry-run` with isolated output directories before a production export.
Native TensorRT export and INT8 calibration require the target NVIDIA stack and
prepared calibration data. Every export records artifact and environment
metadata; TensorRT engines should be rebuilt on the deployment target.

## Native builds

List available profiles:

```powershell
cmake --list-presets
cmake --build --list-presets
ctest --list-presets
```

### Portable CPU profile

Set dependency locations before configuring:

```powershell
$env:EDGE_ONNXRUNTIME_ROOT = "E:\cpp-libs\onnxruntime-win-x64-1.27.0"
$env:EDGE_ONNXRUNTIME_RUNTIME_DIR = "$env:EDGE_ONNXRUNTIME_ROOT\lib"
$env:EDGE_OPENCV_DIR = "C:\opencv\build\x64\vc16\lib"
```

Then configure, build, and test:

```powershell
cmake --preset cpu-release
cmake --build --preset cpu-release
ctest --preset cpu-release --output-on-failure
```

The profile builds the application, core library, generated ONNX fixture, and
four registered tests.

### NVIDIA profile

Run from an MSVC developer shell whose toolset is supported by the installed
CUDA Toolkit. Configure the dependency roots:

```powershell
$env:EDGE_ONNXRUNTIME_ROOT = "E:\cpp-libs\onnxruntime-win-x64-gpu_cuda13-1.27.0"
$env:EDGE_ONNXRUNTIME_RUNTIME_DIR = "$env:EDGE_ONNXRUNTIME_ROOT\lib"
$env:EDGE_TENSORRT_ROOT = "E:\cpp-libs\TensorRT-10.9.0.34"
$env:EDGE_CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:EDGE_OPENCV_DIR = "C:\opencv\build\x64\vc16\lib"
```

The validated executable uses the CUDA 12.8 toolkit for native TensorRT and
the CUDA 13 runtime required by the ORT 1.27 GPU package. Before running a GPU
backend, put every provider dependency on `Path`:

```powershell
$cuda13Runtime = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
$cudnnRuntime = "E:\cpp-libs\cudnn9-runtime"
$runtimePaths = @(
    "$env:EDGE_ONNXRUNTIME_ROOT\lib"
    "$env:EDGE_TENSORRT_ROOT\lib"
    "$env:EDGE_TENSORRT_ROOT\bin"
    "$env:EDGE_CUDA_ROOT\bin"
    $cuda13Runtime
    $cudnnRuntime
)
$env:Path = ($runtimePaths -join ";") + ";" + $env:Path
```

Adjust those two standalone runtime paths for the installed CUDA 13 and cuDNN
packages. A provider can appear in ORT's provider list even when one of its
runtime DLL dependencies is still missing.

For the validated CUDA 12.8 Windows setup, initialize MSVC 14.39 before
configuring. Then build for the target GPU architecture:

```powershell
cmake --preset nvidia-release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build --preset nvidia-release
ctest --preset nvidia-release --output-on-failure
```

Do not reuse a CMake cache after changing compilers, CUDA versions, or major
SDK paths. See
[docs/BUILD_MAX_PERFORMANCE.md](docs/BUILD_MAX_PERFORMANCE.md) for the explicit
maximum-performance configuration and
[docs/GPU_PIPELINE_USAGE.md](docs/GPU_PIPELINE_USAGE.md) for the fused API.

### Installation

Install a completed build into an isolated prefix:

```powershell
cmake --install build\cpu-release `
    --config Release `
    --prefix install\cpu-release
```

## Command-line usage

Show the authoritative option list:

```powershell
build\cpu-release\bin\Release\edge_perception.exe --help
```

### Image inference with automatic fallback

```powershell
build\cpu-release\bin\Release\edge_perception.exe `
    --mode infer `
    --source samples\frame.jpg `
    --backend auto `
    --model models\onnx\yolo26m_bdd100k_fp32.onnx `
    --precision fp32 `
    --output-dir outputs\image
```

### Directory inference with ORT CUDA

```powershell
build\nvidia-release\bin\Release\edge_perception.exe `
    --mode infer `
    --source data\raw\bdd100k\bdd100k\val\images `
    --backend ort-cuda `
    --model models\onnx\yolo26m_bdd100k_fp16.onnx `
    --precision fp16 `
    --output-dir outputs\directory
```

### Video inference with the fused pipeline

```powershell
build\nvidia-release\bin\Release\edge_perception.exe `
    --mode infer `
    --source samples\video.mp4 `
    --backend fused-trt `
    --model models\engine\yolo26m_bdd100k_fp16.engine `
    --precision fp16 `
    --pipeline-depth 3 `
    --output-dir outputs\video
```

### Camera inference

Camera sources use the explicit `camera:<index>` form:

```powershell
build\nvidia-release\bin\Release\edge_perception.exe `
    --mode infer `
    --source camera:0 `
    --backend fused-trt `
    --model models\engine\yolo26m_bdd100k_fp16.engine `
    --precision fp16 `
    --pipeline-depth 2 `
    --output-dir outputs\camera
```

Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.ppm`,
`.tif`, `.tiff`, and `.webp`. PPM support is used by the deterministic CLI
integration test.

## Benchmarking

Latency mode completes one frame before starting the next:

```powershell
build\nvidia-release\bin\Release\edge_perception.exe `
    --mode benchmark `
    --source data\raw\bdd100k\bdd100k\val\images `
    --backend fused-trt `
    --model models\engine\yolo26m_bdd100k_fp16.engine `
    --precision fp16 `
    --benchmark-mode latency `
    --timing-scope end-to-end `
    --pipeline-depth 1 `
    --warmup 30 `
    --iterations 300 `
    --output-dir benchmarks\latency
```

Throughput mode keeps a bounded number of frames in flight:

```powershell
build\nvidia-release\bin\Release\edge_perception.exe `
    --mode benchmark `
    --source data\raw\bdd100k\bdd100k\val\images `
    --backend fused-trt `
    --model models\engine\yolo26m_bdd100k_fp16.engine `
    --precision fp16 `
    --benchmark-mode throughput `
    --timing-scope end-to-end `
    --pipeline-depth 3 `
    --warmup 50 `
    --iterations 1000 `
    --retain-samples `
    --output-dir benchmarks\throughput
```

Streaming FPS is derived from completed frames over measured wall-clock time,
not from `1000 / mean latency` when stages overlap. Record the exact hardware,
driver, CUDA, TensorRT, ONNX Runtime, compiler, model hash, precision, warm-up,
iteration count, input set, timing scope, and pipeline settings with published
results.

## Tests and validation

### Python

```powershell
python -m compileall -q data training tests
python -m pytest -q -m "not integration"
python -m pip check
```

The default Python suite contains seven parity unit tests and one opt-in real
artifact integration test.

### Native CTest

```powershell
ctest --preset cpu-release --output-on-failure
ctest --preset nvidia-release --output-on-failure
```

Each configured profile registers:

```text
edge_unit_tests
edge_ort_cpu_integration
edge_auto_backend_cli
edge_python_parity_unit
```

The tests cover preprocessing geometry and tensor layout, postprocessing,
benchmark statistics, provider discovery, ORT CPU inference, automatic CPU
fallback, CLI output, and Python parity rules.

### Real PyTorch → ONNX → C++ parity

Set all required paths before enabling the integration marker:

```powershell
$env:EDGE_TEST_WEIGHTS = "runs\train\yolo26m_bdd100k\weights\best.pt"
$env:EDGE_TEST_ONNX_MODEL = "models\onnx\yolo26m_bdd100k_fp32.onnx"
$env:EDGE_TEST_IMAGE = "samples\frame.jpg"
$env:EDGE_TEST_EXECUTABLE = "build\cpu-release\bin\Release\edge_perception.exe"
$env:EDGE_TEST_DEVICE = "cpu"

python -m pytest -q tests\test_parity.py -m integration -s
```

Optional threshold variables are documented in `tests/test_parity.py`.

## Docker

The Docker image is an NVIDIA deployment image based on
`nvcr.io/nvidia/tensorrt:25.03-py3`. It builds the full native CUDA/TensorRT
profile, generates the deterministic ONNX fixture, runs all four CPU-safe CTest
targets during the builder stage, installs into `/opt/edge-perception`, and
runs as UID 10001 in the final stage.

The Dockerfile pins ONNX Runtime GPU 1.26 for its CUDA 12-compatible Linux
package. ORT 1.27 GPU packages move to CUDA 13, so do not update that build
argument independently of the TensorRT base image and CUDA runtime.

The image intentionally excludes datasets, checkpoints, ONNX models, TensorRT
engines, caches, benchmark inputs, and generated output. Mount those at
runtime.

### Build

RTX 2060 uses compute capability 7.5:

```powershell
docker build `
    --build-arg CUDA_ARCHITECTURES=75 `
    --tag edge-perception-cpp:1.1.0 `
    .
```

Use a semicolon-separated architecture list such as `75;86` only when the
image must support multiple named GPU targets.

### Container smoke test

```powershell
docker run --rm edge-perception-cpp:1.1.0 --version
```

### ORT CUDA inference

```powershell
docker run `
    --rm `
    --gpus all `
    --volume "${PWD}\models\onnx:/workspace/models/onnx:ro" `
    --volume "${PWD}\samples:/workspace/inputs:ro" `
    --volume "${PWD}\outputs\docker:/workspace/outputs" `
    edge-perception-cpp:1.1.0 `
    --mode infer `
    --source /workspace/inputs/frame.jpg `
    --backend ort-cuda `
    --model /workspace/models/onnx/yolo26m_bdd100k_fp32.onnx `
    --precision fp32 `
    --output-dir /workspace/outputs
```

### Create a Linux TensorRT engine

A Windows `.engine` file must not be reused in the Linux image. Build the
engine inside the target container:

```powershell
docker run `
    --rm `
    --gpus all `
    --entrypoint /bin/bash `
    --volume "${PWD}\models\onnx:/workspace/models/onnx:ro" `
    --volume "${PWD}\models\docker-engine:/workspace/models/engine" `
    edge-perception-cpp:1.1.0 `
    -lc "trtexec --onnx=/workspace/models/onnx/yolo26m_bdd100k_fp16.onnx --saveEngine=/workspace/models/engine/yolo26m_bdd100k_fp16_linux_sm75.engine --fp16 --memPoolSize=workspace:4096 --skipInference"
```

Use the resulting engine with `native-trt` or `fused-trt` and the same mounted
paths. Verify GPU access independently with:

```powershell
docker run --rm --gpus all nvcr.io/nvidia/tensorrt:25.03-py3 nvidia-smi
```

## Continuous integration

`.github/workflows/ci.yml` runs the portable CPU profile on Windows Server 2022
and Ubuntu 24.04. It installs pinned test dependencies, checks Python syntax,
builds the native application, runs all CTest targets and direct Python parity
tests, and checks patch whitespace.

Public GitHub-hosted runners do not validate native TensorRT or fused CUDA
execution. Those checks require the target machine or a compatible self-hosted
GPU runner.

## Troubleshooting

### CTest reports no tests

Reconfigure from a fresh build directory with `EDGE_BUILD_TESTS=ON` (the
provided release presets already enable it), build, then run the matching test
preset.

### MSVC cannot find standard or Windows SDK headers

Run CMake from a Visual Studio developer shell. Having `cl.exe` on `PATH` alone
is insufficient; the developer environment must also provide `INCLUDE`, `LIB`,
and Windows SDK paths.

### CUDA rejects the host compiler

Select a CUDA-supported MSVC toolset before configuring and use a fresh CMake
cache. Avoid `--allow-unsupported-compiler` for final release validation when a
supported toolset is installed.

### ORT CUDA or TensorRT providers disappear

Disable Ultralytics automatic installation, run `python -m pip check`, and
inspect the active package:

```powershell
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

A provider appearing in the list does not guarantee successful session
creation; its CUDA, cuDNN, TensorRT, driver, and shared-library dependencies
must also be available.

### TensorRT engine deserialization fails

Rebuild the engine for the current operating system, GPU architecture,
TensorRT version, CUDA stack, driver/runtime combination, precision, and build
configuration.

### Docker sends a large build context

Confirm `.dockerignore` is at the repository root. It uses an allowlist; do not
replace it with a broad context unless the Dockerfile is intentionally changed
to consume additional files.

## Known limitations

- TensorRT engines are target-specific.
- The Docker image does not reuse Windows TensorRT engines.
- GPU CI requires a self-hosted or otherwise GPU-enabled runner.
- The runtime does not include NVDEC, DeepStream, GStreamer camera integration,
  or multi-camera synchronization.
- INT8 quality and performance must be validated per artifact and target.
- Fused GPU execution is validated only on compatible NVIDIA builds.

## Release checklist

Before publishing a tag or container image, confirm:

```text
[ ] git diff --check passes
[ ] Python syntax and unit tests pass
[ ] python -m pip check passes in the selected environment
[ ] CPU configure, build, and CTest pass
[ ] NVIDIA configure, build, and CTest pass on the target machine
[ ] Real model parity passes for the release artifacts
[ ] Explicit ORT CUDA and TensorRT provider smoke tests pass
[ ] Native and fused TensorRT smoke tests pass with a target-built engine
[ ] Docker static check, image build, and container smoke test pass
[ ] GitHub CPU CI passes on Windows and Ubuntu
[ ] Model, executable, environment, and benchmark provenance is recorded
```

Do not commit or publish local datasets, credentials, model weights, exported
engines, caches, build trees, or experiment output.

## Additional documentation

- [Backend matrix](docs/backend_matrix.md)
- [Maximum-performance Windows build](docs/BUILD_MAX_PERFORMANCE.md)
- [Native fused pipeline usage](docs/GPU_PIPELINE_USAGE.md)
- [Optimization architecture](docs/OPTIMIZATION_AUDIT.md)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [NVIDIA TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
- [BDD100K](https://bdd-data.berkeley.edu/)
