# edge-perception-cpp

[![CI](https://github.com/tr3npa1/edge-perception-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/tr3npa1/edge-perception-cpp/actions/workflows/ci.yml)

A production-oriented YOLO26M deployment and benchmarking system for automotive and industrial computer vision.

The repository covers the full machine-learning deployment path:

```text
BDD100K acquisition
→ dataset validation
→ YOLO26M training
→ MLflow experiment tracking
→ model evaluation
→ FP32 / FP16 / INT8 export
→ ONNX Runtime deployment
→ native TensorRT deployment
→ fused CUDA/TensorRT execution
→ Python/C++ parity validation
→ end-to-end latency and throughput benchmarking
```

The central engineering goal is not merely to train an object detector. It is to build a reproducible bridge between Python model development and a portable, performance-oriented C++17 inference runtime.

---

## Project status

### Completed

- BDD100K download and validation pipeline
- YOLO26M training
- MLflow experiment tracking
- model evaluation
- FP32 ONNX export
- FP16 ONNX export
- INT8 ONNX post-training quantization
- native FP32 TensorRT engine export
- native FP16 TensorRT engine export
- TensorRT engine provenance metadata
- ONNX Runtime CPU inference
- ONNX Runtime CUDA inference
- ONNX Runtime TensorRT Execution Provider inference
- native TensorRT inference
- fused TensorRT/CUDA pipeline
- image inference
- image-directory inference
- video inference
- camera inference
- asynchronous fused-pipeline execution
- CUDA Graph execution
- GPU postprocessing
- persistent ONNX Runtime device I/O binding
- reproducible benchmark output
- automatic ONNX backend selection
- universal ONNX Runtime CPU fallback
- CPU-only CMake build profile
- NVIDIA CMake build profile
- registered CTest suite
- Python parity unit tests
- real PyTorch → ONNX → C++ parity validation
- Windows CPU acceptance testing
- Windows NVIDIA acceptance testing
- Windows and Ubuntu CPU CI workflow
- Linux GPU Docker packaging

### Local acceptance result

The final local Windows validation completed successfully after the portability upgrade:

```text
CPU CTest:
    4/4 passed

NVIDIA CTest:
    4/4 passed

Python parity unit tests:
    7 passed
    1 integration test deselected

Real PyTorch → ONNX → C++ parity:
    1 passed

Automatic CPU fallback:
    requested backend: auto
    selected backend: ort_cpu

Python ONNX Runtime providers after environment repair:
    TensorrtExecutionProvider
    CUDAExecutionProvider
    CPUExecutionProvider
```

The registered CTest targets are:

```text
edge_unit_tests
edge_ort_cpu_integration
edge_auto_backend_cli
edge_python_parity_unit
```

The old state in which CTest reported:

```text
No tests were found!!!
```

has been eliminated.

---

## Important project boundary

This repository is a high-performance computer-vision deployment project.

It is not:

- automotive safety-certified software;
- an ISO 26262 implementation;
- a replacement for a production perception safety case;
- a complete autonomous-driving stack;
- a universal TensorRT binary distribution;
- a guarantee of identical performance on all hardware.

The software demonstrates deployment engineering, portability, backend selection, parity testing, TensorRT integration, CUDA optimization and reproducible benchmarking.

---

# Key measured results

The final Windows benchmark matrix completed:

```text
12 valid artifact/backend cells
12 successful smoke-test rows
50 successful benchmark rows
62/62 successful processes
0 recorded process failures
```

Representative RTX 2060 results:

| Configuration | Result |
|---|---:|
| Fused TensorRT FP16, pipeline depth 1 | approximately 5.03 ms mean end-to-end latency |
| Fused TensorRT FP16, pipeline depth 3 | approximately 254.1 FPS sustained throughput |
| Fused pipeline throughput improvement over synchronous native TensorRT | approximately 28.4% |

These measurements came from the saved benchmark package:

```text
full_matrix_20260723_204901
```

The benchmark distinguishes:

```text
backend execution latency
end-to-end frame latency
streaming frame latency
sustained throughput
```

Streaming throughput is measured from completed frames over wall-clock time. It is not inferred by taking `1000 / mean latency` when pipeline stages overlap.

---

# System architecture

```text
                           Python development side
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  BDD100K                                                            │
│      │                                                              │
│      ▼                                                              │
│  data/download.py                                                   │
│      │                                                              │
│      ▼                                                              │
│  data/dataset.py                                                    │
│      │                                                              │
│      ▼                                                              │
│  training/train_detector.py                                         │
│      │                                                              │
│      ├──────────────► MLflow                                        │
│      │                                                              │
│      ▼                                                              │
│  training/evaluate.py                                               │
│      │                                                              │
│      ▼                                                              │
│  training/export_model.py                                           │
│      │                                                              │
│      ├────────► FP32 ONNX                                           │
│      ├────────► FP16 ONNX                                           │
│      ├────────► INT8 ONNX                                           │
│      ├────────► FP32 TensorRT engine                                │
│      ├────────► FP16 TensorRT engine                                │
│      └────────► export and provenance metadata                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                            artifacts
                               │
                               ▼
                         Native C++17 runtime
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  image / directory / video / camera                                 │
│      │                                                              │
│      ▼                                                              │
│  image_processor.cpp                                                │
│      │                                                              │
│      ├──── resize / letterbox                                       │
│      ├──── BGR → RGB                                                │
│      ├──── normalization                                            │
│      └──── HWC → NCHW                                               │
│      │                                                              │
│      ▼                                                              │
│  inference_engine.cpp                                               │
│      │                                                              │
│      ├──── ORT CPU                                                  │
│      ├──── ORT CUDA                                                 │
│      ├──── ORT TensorRT EP                                          │
│      └──── native TensorRT                                          │
│      │                                                              │
│      ▼                                                              │
│  postprocess.cpp                                                    │
│      │                                                              │
│      ├──── confidence filtering                                     │
│      ├──── coordinate restoration                                   │
│      ├──── clipping                                                 │
│      └──── final detections                                         │
│      │                                                              │
│      ▼                                                              │
│  rendered output / JSONL / benchmark report                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

                         Fused NVIDIA pipeline
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  decoded frame                                                      │
│      │                                                              │
│      ▼                                                              │
│  CUDA preprocessing                                                 │
│      │                                                              │
│      ▼                                                              │
│  TensorRT inference                                                 │
│      │                                                              │
│      ▼                                                              │
│  CUDA postprocessing                                                │
│      │                                                              │
│      ▼                                                              │
│  asynchronous pipeline slots                                        │
│      │                                                              │
│      ▼                                                              │
│  ordered final detections                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

# Backend architecture

The native executable supports five explicit backends and one automatic backend:

```text
auto
ort-cpu
ort-cuda
ort-trt
native-trt
fused-trt
```

## Automatic backend selection

For ONNX artifacts:

```text
--backend auto
```

attempts:

```text
ONNX Runtime TensorRT Execution Provider
→ ONNX Runtime CUDA Execution Provider
→ ONNX Runtime CPU Execution Provider
```

Each failed initialization attempt is recorded.

The selected backend is written to:

```text
backend_selection.json
```

Example CPU-only selection:

```json
{
  "requested_backend": "auto",
  "selected_backend": "ort_cpu"
}
```

Automatic fallback applies only to ONNX Runtime session/provider initialization.

It does not silently hide:

- invalid model contracts;
- malformed model outputs;
- inference failures after successful initialization;
- corrupted artifacts;
- invalid preprocessing;
- invalid postprocessing;
- TensorRT engine deserialization failures.

Explicit backend requests remain strict.

For example:

```text
--backend ort-cuda
```

must initialize CUDA or fail.

Likewise:

```text
--backend fused-trt
```

must initialize the fused TensorRT/CUDA implementation or fail.

Native and fused TensorRT engines are not part of the automatic ONNX fallback chain because `.engine` and `.onnx` are different artifact types.

---

# Model contract

The deployed detection model uses:

```text
input name: model-dependent, discovered at runtime
input shape: [1, 3, 640, 640]
input type: float32 or float16

output shape: [1, 300, 6]
```

Each final output row is:

```text
[x1, y1, x2, y2, confidence, class_id]
```

The C++ runtime preserves model output order.

It does not perform a second NMS pass or reorder final detections after the model has already generated final detection rows.

---

# BDD100K classes

```text
0  person
1  rider
2  car
3  bus
4  truck
5  bike
6  motor
7  traffic light
8  traffic sign
9  train
```

---

# Repository structure

```text
edge-perception-cpp/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── download.py
│   ├── dataset.py
│   ├── processed/
│   └── raw/
│
├── docs/
│
├── include/
│   ├── benchmark.hpp
│   ├── gpu_pipeline.hpp
│   ├── image_processor.hpp
│   ├── inference_engine.hpp
│   └── postprocess.hpp
│
├── models/
│   ├── engine/
│   ├── onnx/
│   └── ort_trt_cache/
│
├── src/
│   ├── benchmark.cpp
│   ├── gpu_pipeline.cu
│   ├── gpu_pipeline_common.cpp
│   ├── gpu_pipeline_stub.cpp
│   ├── image_processor.cpp
│   ├── inference_engine.cpp
│   ├── main.cpp
│   └── postprocess.cpp
│
├── tests/
│   ├── generate_test_model.py
│   ├── test_auto_backend.py
│   ├── test_inference.cpp
│   └── test_parity.py
│
├── training/
│   ├── evaluate.py
│   ├── export_model.py
│   └── train_detector.py
│
├── .dockerignore
├── .gitignore
├── CMakeLists.txt
├── CMakePresets.json
├── Dockerfile
├── pytest.ini
├── requirements-common.txt
├── requirements-cpu.txt
├── requirements-gpu.txt
├── requirements-tensorrt.txt
├── requirements-test.txt
├── requirements.txt
└── README.md
```

Large generated artifacts are intentionally not committed:

```text
BDD100K data
training runs
MLflow data
PyTorch checkpoints
ONNX models
TensorRT engines
ORT TensorRT caches
benchmark output
rendered inference output
native build directories
Python virtual environments
```

---

# Python environments

Recommended Python:

```text
Python 3.11
```

The dependency files are separated by purpose.

## Main GPU development environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` is the main GPU training/export/evaluation environment.

## CPU testing environment

```powershell
py -3.11 -m venv .venv-cpu
.\.venv-cpu\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
```

## Test environment

```powershell
py -3.11 -m venv .venv-test
.\.venv-test\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
```

## GPU-specific environment

```powershell
python -m pip install -r requirements-gpu.txt
```

## Optional TensorRT Python environment

```powershell
python -m pip install -r requirements-tensorrt.txt
```

TensorRT Python bindings remain optional.

The native C++ TensorRT implementation uses the native NVIDIA TensorRT SDK and does not depend on the Python TensorRT package at runtime.

---

# Important dependency pins

The validated environment includes:

```text
numpy==2.2.6
protobuf==6.33.6
onnxruntime-gpu==1.27.0
```

These pins are intentional.

A forced reinstall of `onnxruntime-gpu` without constraints may upgrade NumPy or protobuf to versions that conflict with MLflow or Databricks dependencies.

When repairing ONNX Runtime GPU without changing dependencies, use:

```powershell
python -m pip install `
    --no-cache-dir `
    --force-reinstall `
    --no-deps `
    "onnxruntime-gpu==1.27.0"
```

Then validate:

```powershell
python -m pip check
```

Expected:

```text
No broken requirements found.
```

---

# Ultralytics automatic dependency installation

Ultralytics may attempt to install missing packages while a model is loading.

During parity validation, Ultralytics installed CPU-only:

```text
onnxruntime
```

which replaced the effective GPU runtime and temporarily reduced available providers to:

```text
AzureExecutionProvider
CPUExecutionProvider
```

Disable automatic package modification in controlled environments:

```powershell
$env:YOLO_AUTOINSTALL = "false"
```

Persist it for the current Windows user:

```powershell
[Environment]::SetEnvironmentVariable(
    "YOLO_AUTOINSTALL",
    "false",
    "User"
)
```

The expected GPU provider list is:

```text
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

Verify:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

---

# Dataset

The project uses BDD100K object-detection data in YOLO format.

## Download

```powershell
python data\download.py
```

## Validate and prepare local dataset metadata

```powershell
python data\dataset.py
```

Prepared dataset YAML:

```text
data/processed/bdd100k_yolo/bdd100k.yaml
```

The raw Kaggle YAML is not modified.

The preparation step creates normalized local paths and class metadata under:

```text
data/processed/
```

---

# Training

Training entrypoint:

```text
training/train_detector.py
```

Representative configuration:

```text
model: YOLO26M
dataset: BDD100K
batch size: 16
workers: 8
initial learning rate: 0.01
final learning-rate factor: 0.01
momentum: 0.937
weight decay: 0.0005
warmup epochs: 3
patience: 50
close mosaic: final 10 epochs
AMP: enabled
cache: disabled
```

Example:

```powershell
python training\train_detector.py `
    --data data\processed\bdd100k_yolo\bdd100k.yaml `
    --device 0
```

The final deployment workflow should use the best checkpoint rather than the last checkpoint.

Typical path:

```text
runs/detect/runs/train/yolo26m_bdd100k/weights/best.pt
```

---

# Evaluation

Evaluation entrypoint:

```text
training/evaluate.py
```

The evaluation layer supports:

- detector metrics;
- artifact validation;
- provider checks;
- final-detection parity;
- IoU matching;
- confidence-difference checks;
- coordinate-difference checks;
- missing and extra detection accounting;
- per-image parity results;
- aggregate parity summaries.

The Python parity implementation was reused by the automated regression suite rather than duplicated into a separate incompatible comparison system.

---

# Export

Export entrypoint:

```text
training/export_model.py
```

Supported artifact families include:

```text
onnx_fp32
onnx_fp16
onnx_int8
engine_fp32
engine_fp16
engine_int8
```

Canonical artifacts:

```text
models/onnx/yolo26m_bdd100k_fp32.onnx
models/onnx/yolo26m_bdd100k_fp16.onnx
models/onnx/yolo26m_bdd100k_int8.onnx

models/engine/yolo26m_bdd100k_fp32.engine
models/engine/yolo26m_bdd100k_fp16.engine
models/engine/yolo26m_bdd100k_int8.engine
```

## Dry run

Use an isolated output directory:

```powershell
python training\export_model.py `
    --weights runs\detect\runs\train\yolo26m_bdd100k\weights\best.pt `
    --data data\processed\bdd100k_yolo\bdd100k.yaml `
    --onnx-dir outputs\v11_export_dryrun\onnx `
    --engine-dir outputs\v11_export_dryrun\engine `
    --ort-tensorrt-cache-dir outputs\v11_export_dryrun\ort_cache `
    --basename v11_dryrun `
    --variants engine_fp16 `
    --device 0 `
    --dry-run `
    --overwrite
```

Using isolated output paths avoids collisions with canonical production artifacts.

## TensorRT engine provenance

TensorRT engine metadata records information such as:

```text
source artifact path
source artifact SHA-256
engine artifact path
engine artifact SHA-256
engine size
GPU name
compute capability
CUDA version
TensorRT version
driver information
build flags
build timestamp
calibration identity when applicable
```

TensorRT engines are target-specific.

A TensorRT engine may need rebuilding when any of these change:

```text
operating system
GPU architecture
TensorRT version
CUDA stack
driver/runtime compatibility
engine build configuration
precision or calibration configuration
```

ONNX is the portable deployment artifact.

TensorRT `.engine` files are target-specific acceleration artifacts.

---

# Native C++ build

The project requires:

```text
C++17
CMake 3.24 or newer
OpenCV
ONNX Runtime C/C++ package
Ninja or another supported CMake generator
```

Optional NVIDIA acceleration requires:

```text
NVIDIA CUDA Toolkit
NVIDIA TensorRT SDK
compatible host compiler
compatible NVIDIA driver
```

---

# CMake presets

Two primary build profiles are provided:

```text
cpu-release
nvidia-release
```

List them:

```powershell
cmake --list-presets
cmake --build --list-presets
ctest --list-presets
```

---

# Windows CPU-only build

Use the CPU-only ONNX Runtime C++ package.

Example dependency variables:

```powershell
$env:EDGE_ONNXRUNTIME_ROOT = `
    "E:\cpp-libs\onnxruntime-win-x64-1.27.0"

$env:EDGE_ONNXRUNTIME_RUNTIME_DIR = `
    "E:\cpp-libs\onnxruntime-win-x64-1.27.0\lib"

$env:EDGE_OPENCV_DIR = `
    "C:\opencv\build\x64\vc16\lib"
```

Configure:

```powershell
cmake --preset cpu-release
```

Build:

```powershell
cmake --build --preset cpu-release
```

Test:

```powershell
ctest `
    --preset cpu-release `
    --output-on-failure
```

Expected:

```text
100% tests passed, 0 tests failed out of 4
```

---

# Windows NVIDIA build

Example dependency variables:

```powershell
$env:EDGE_ONNXRUNTIME_ROOT = `
    "E:\cpp-libs\onnxruntime-win-x64-gpu_cuda13-1.27.0"

$env:EDGE_ONNXRUNTIME_RUNTIME_DIR = `
    "E:\cpp-libs\onnxruntime-win-x64-gpu_cuda13-1.27.0\lib"

$env:EDGE_TENSORRT_ROOT = `
    "E:\cpp-libs\TensorRT-10.9.0.34"

$env:EDGE_CUDA_ROOT = `
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

$env:EDGE_OPENCV_DIR = `
    "C:\opencv\build\x64\vc16\lib"
```

Configure:

```powershell
cmake --preset nvidia-release
```

Build:

```powershell
cmake --build --preset nvidia-release
```

Test:

```powershell
ctest `
    --preset nvidia-release `
    --output-on-failure
```

Expected:

```text
100% tests passed, 0 tests failed out of 4
```

---

# CUDA 12.8 and Windows host compiler

CUDA 12.8 rejected the default MSVC 14.51 toolset on the validated machine.

The compatible installed toolset used for the final NVIDIA build was:

```text
MSVC toolset: 14.39.33519
compiler: 19.39.33523
```

Load it explicitly:

```powershell
$VcVars = `
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

cmd /s /c `
    "`"$VcVars`" x64 -vcvars_ver=14.39 && set" |
    ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable(
                $Matches[1],
                $Matches[2],
                "Process"
            )
        }
    }

$env:EDGE_MSVC_COMPILER = (
    where.exe cl |
    Select-Object -First 1
)

$env:CUDAHOSTCXX = $env:EDGE_MSVC_COMPILER
```

Confirm:

```powershell
Write-Host $env:EDGE_MSVC_COMPILER

(Get-Item -LiteralPath $env:EDGE_MSVC_COMPILER).
    VersionInfo.
    FileVersion
```

The selected compiler path should contain:

```text
MSVC\14.39.33519\bin\Hostx64\x64\cl.exe
```

Avoid using:

```text
--allow-unsupported-compiler
```

for final release validation when a supported toolset is available.

---

# Command-line interface

Version:

```powershell
edge_perception.exe --version
```

Help:

```powershell
edge_perception.exe --help
```

## Image inference

```powershell
edge_perception.exe `
    --mode infer `
    --source samples\frame.jpg `
    --backend auto `
    --model models\onnx\yolo26m_bdd100k_fp32.onnx `
    --precision fp32 `
    --confidence 0.25 `
    --max-detections 300 `
    --output-dir outputs\image
```

## Directory inference

```powershell
edge_perception.exe `
    --mode infer `
    --source data\raw\bdd100k\bdd100k\val\images `
    --backend ort-cuda `
    --model models\onnx\yolo26m_bdd100k_fp16.onnx `
    --precision fp16 `
    --confidence 0.25 `
    --max-detections 300 `
    --output-dir outputs\directory
```

## Video inference

```powershell
edge_perception.exe `
    --mode infer `
    --source samples\video.mp4 `
    --backend fused-trt `
    --model models\engine\yolo26m_bdd100k_fp16.engine `
    --precision fp16 `
    --pipeline-depth 3 `
    --output-dir outputs\video
```

## Camera inference

```powershell
edge_perception.exe `
    --mode infer `
    --source 0 `
    --backend fused-trt `
    --model models\engine\yolo26m_bdd100k_fp16.engine `
    --precision fp16 `
    --pipeline-depth 2 `
    --output-dir outputs\camera
```

---

# Supported image extensions

The CLI accepts common image formats including:

```text
.jpg
.jpeg
.png
.bmp
.ppm
.tif
.tiff
.webp
```

PPM support is intentional.

The automatic-backend CLI integration test creates a deterministic PPM fixture because PPM is simple to generate without adding another image-writing dependency to the test.

OpenCV performs the actual image decoding.

---

# Testing

## Python unit tests

```powershell
python -m pytest `
    -q `
    tests\test_parity.py `
    -m "not integration"
```

Validated result:

```text
7 passed
1 deselected
```

## Test collection

```powershell
python -m pytest --collect-only -q
```

## Generated ONNX fixture

```powershell
python tests\generate_test_model.py `
    --output build\v11-preflight\tiny_detector.onnx
```

The fixture has:

```text
input:  images
output: output0
```

and is checked with the ONNX model checker.

## CTest

List tests:

```powershell
ctest `
    --test-dir build\cpu-release `
    -C Release `
    -N
```

Run CPU tests:

```powershell
ctest `
    --preset cpu-release `
    --output-on-failure
```

Run NVIDIA tests:

```powershell
ctest `
    --preset nvidia-release `
    --output-on-failure
```

## C++ unit coverage

The C++ tests cover:

```text
letterbox geometry
BGR to RGB conversion
NCHW tensor layout
normalization
coordinate restoration
box clipping
malformed detection rows
confidence validation
class validation
benchmark statistics
provider discovery
ORT CPU end-to-end inference
```

## Automatic-backend CLI integration

The test:

```text
tests/test_auto_backend.py
```

creates:

```text
a deterministic PPM image
a deterministic ONNX detector fixture
an isolated output directory
```

It launches the actual C++ executable with:

```text
--backend auto
```

and verifies:

```text
successful process exit
backend_selection.json creation
detections.jsonl creation
requested backend = auto
selected backend = ort_cpu on the CPU configuration
```

## Real parity integration

The optional integration test compares:

```text
PyTorch checkpoint
ONNX Runtime inference
native C++ ORT CPU inference
```

Environment variables:

```text
EDGE_TEST_WEIGHTS
EDGE_TEST_ONNX_MODEL
EDGE_TEST_IMAGE
EDGE_TEST_EXECUTABLE
EDGE_TEST_DEVICE
EDGE_TEST_CONFIDENCE
EDGE_TEST_IOU
EDGE_TEST_MAX_DETECTIONS
```

Run:

```powershell
python -m pytest `
    -q `
    tests\test_parity.py `
    -m "integration" `
    -s
```

Validated result:

```text
1 passed
```

---

# Continuous integration

Workflow:

```text
.github/workflows/ci.yml
```

The CPU CI matrix targets:

```text
Windows
Ubuntu
```

The workflow is responsible for validating:

```text
Python syntax
Python parity unit tests
CMake configuration
CPU compilation
CTest registration
CTest execution
```

GPU testing is not assumed on ordinary public GitHub-hosted runners.

The full NVIDIA acceptance suite remains a local or self-hosted-runner responsibility.

---

# Benchmarking

## Latency

```powershell
edge_perception.exe `
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

## Throughput

```powershell
edge_perception.exe `
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
    --output-dir benchmarks\throughput
```

## Pipeline depth

Supported range:

```text
1 through 8
```

Default:

```text
2
```

For synchronous generic backends, effective pipeline depth remains one.

For fused TensorRT, additional pipeline slots permit overlap between frame processing stages.

---

# Docker

The Docker image provides a coherent Linux NVIDIA stack.

Base image:

```text
nvcr.io/nvidia/tensorrt:25.03-py3
```

Container stack:

```text
Ubuntu 24.04
CUDA 12.8 generation
TensorRT 10.9 generation
ONNX Runtime GPU 1.26
OpenCV
CMake
Ninja
native C++17 application
```

The container does not include:

```text
BDD100K
PyTorch checkpoints
ONNX deployment models
TensorRT engine artifacts
benchmark input data
generated output
```

These are mounted at runtime.

## Important engine rule

A TensorRT engine created on Windows must not be treated as a portable Linux engine.

Create a new Linux TensorRT engine inside the target Linux container/runtime.

Portable artifact:

```text
ONNX
```

Target-specific artifact:

```text
TensorRT .engine
```

## Build architecture

RTX 2060 uses CUDA compute capability 7.5:

```powershell
docker build `
    --build-arg CUDA_ARCHITECTURES=75 `
    --tag edge-perception-cpp:1.1.0 `
    .
```

## Runtime GPU access

```powershell
docker run `
    --rm `
    --gpus all `
    edge-perception-cpp:1.1.0 `
    --version
```

## ORT CUDA inference

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

## ORT TensorRT inference

```powershell
docker run `
    --rm `
    --gpus all `
    --volume "${PWD}\models\onnx:/workspace/models/onnx:ro" `
    --volume "${PWD}\models\docker-ort-cache:/workspace/cache" `
    --volume "${PWD}\samples:/workspace/inputs:ro" `
    --volume "${PWD}\outputs\docker-ort-trt:/workspace/outputs" `
    edge-perception-cpp:1.1.0 `
    --mode infer `
    --source /workspace/inputs/frame.jpg `
    --backend ort-trt `
    --model /workspace/models/onnx/yolo26m_bdd100k_fp32.onnx `
    --precision fp32 `
    --ort-cache-dir /workspace/cache `
    --output-dir /workspace/outputs
```

## Linux TensorRT engine creation

Use `trtexec` inside the image:

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

## Fused TensorRT inference

```powershell
docker run `
    --rm `
    --gpus all `
    --volume "${PWD}\models\docker-engine:/workspace/models/engine:ro" `
    --volume "${PWD}\samples:/workspace/inputs:ro" `
    --volume "${PWD}\outputs\docker-fused:/workspace/outputs" `
    edge-perception-cpp:1.1.0 `
    --mode infer `
    --source /workspace/inputs/frame.jpg `
    --backend fused-trt `
    --model /workspace/models/engine/yolo26m_bdd100k_fp16_linux_sm75.engine `
    --precision fp16 `
    --pipeline-depth 1 `
    --require-cuda-graph `
    --output-dir /workspace/outputs
```

---

# Reproducibility

Record the following for publishable results:

```text
Git commit
Git working-tree state
operating system
CPU
GPU
NVIDIA driver
compiler
host compiler toolset
CUDA toolkit
CUDA runtime
cuDNN
TensorRT
ONNX Runtime
OpenCV
CMake
model artifact path
model artifact SHA-256
executable SHA-256
source SHA-256
precision
backend
pipeline depth
CUDA Graph state
GPU postprocessing state
warm-up count
measured iteration count
timing scope
source-frame identity
fast-math state
```

Do not compare benchmark numbers without matching:

```text
hardware
software stack
artifact
precision
warm-up
iterations
timing scope
input set
pipeline depth
CUDA Graph configuration
```

---

# Troubleshooting

## `No tests were found!!!`

The current release configuration registers four CTest targets.

Delete stale build directories and reconfigure:

```powershell
Remove-Item `
    -LiteralPath build\cpu-release `
    -Recurse `
    -Force `
    -ErrorAction SilentlyContinue

cmake --preset cpu-release
cmake --build --preset cpu-release
ctest --preset cpu-release --output-on-failure
```

## Unsupported source extension `.ppm`

The deterministic automatic-backend test uses PPM.

The CLI now intentionally accepts:

```text
.ppm
```

Rebuild after updating `src/main.cpp`.

## CUDA rejects the Visual Studio compiler

Select the installed MSVC 14.39 toolset before configuring the NVIDIA build.

Do not reuse a CMake cache configured with MSVC 14.51.

## CUDA provider disappeared from Python ONNX Runtime

Check:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

When only CPU is listed, remove both ORT packages and reinstall the GPU wheel:

```powershell
python -m pip uninstall -y `
    onnxruntime `
    onnxruntime-gpu

python -m pip install `
    --no-cache-dir `
    --force-reinstall `
    --no-deps `
    onnxruntime-gpu==1.27.0
```

Restore pinned shared dependencies:

```powershell
python -m pip install `
    --no-cache-dir `
    --force-reinstall `
    numpy==2.2.6 `
    protobuf==6.33.6
```

Validate:

```powershell
python -m pip check
```

## ORT CUDA works but ORT TensorRT fails

ORT CUDA and ORT TensorRT require different transitive runtime libraries.

Confirm:

```powershell
where.exe cudnn64_9.dll
where.exe nvinfer_10.dll
where.exe nvinfer_plugin_10.dll
where.exe nvonnxparser_10.dll
```

## TensorRT engine deserialization fails

Rebuild the engine for the current:

```text
operating system
GPU
TensorRT version
CUDA stack
driver/runtime combination
engine configuration
```

## Docker cannot access the GPU

On Windows, ensure Docker Desktop uses its WSL 2 backend and that the NVIDIA driver supports WSL GPU compute.

Verify:

```powershell
docker run `
    --rm `
    --gpus all `
    nvcr.io/nvidia/tensorrt:25.03-py3 `
    nvidia-smi
```

## Docker build sends several gigabytes

Confirm `.dockerignore` exists at the repository root.

The build context must exclude:

```text
data/raw
models
runs
mlruns
build directories
benchmarks
outputs
virtual environments
```

---

# Known limitations

- TensorRT engines are not portable across arbitrary targets.
- The Linux Docker image does not reuse Windows `.engine` files.
- FP16 is the recommended deployment precision for the validated RTX 2060 path.
- The tested INT8 artifacts did not provide the desired deployment quality/performance balance.
- The project does not implement NVDEC.
- The project does not implement DeepStream.
- The project does not implement GStreamer camera integration.
- The project does not implement multi-camera synchronization.
- The project does not implement automotive safety certification.
- GPU CI requires a self-hosted NVIDIA runner or another GPU-enabled CI service.
- Public CPU CI does not validate native TensorRT or fused CUDA execution.

---

# Recommended deployment description

> Cross-platform C++17 object-detection deployment runtime with automatic ONNX Runtime backend selection, universal CPU fallback, optional CUDA and TensorRT acceleration, target-specific TensorRT engine generation, Python/C++ parity testing, reproducible benchmarking, registered automated tests and Windows/Linux build validation.

---

# Resume summary

> Built a cross-platform YOLO26M deployment runtime in C++17 with portable ONNX CPU inference, automatic CUDA/TensorRT acceleration, target-specific TensorRT engine generation, regression-tested Python/C++ parity, persistent GPU I/O binding, CUDA Graph execution and a fused CUDA pipeline reaching approximately 254 FPS on an RTX 2060.

---

# Release checklist

Before publishing a release:

```text
[ ] git diff --check passes
[ ] Python syntax checks pass
[ ] Python parity unit tests pass
[ ] CPU configure succeeds
[ ] CPU build succeeds
[ ] CPU CTest passes 4/4
[ ] NVIDIA configure succeeds
[ ] NVIDIA build succeeds
[ ] NVIDIA CTest passes 4/4
[ ] real PyTorch → ONNX → C++ parity passes
[ ] automatic CPU fallback selects ort_cpu
[ ] ONNX Runtime GPU exposes CUDA and TensorRT providers
[ ] Python dependency check passes
[ ] Docker image builds
[ ] Docker CPU smoke test passes
[ ] Docker ORT CUDA smoke test passes
[ ] Docker ORT TensorRT smoke test passes
[ ] Linux TensorRT engine is built inside the container
[ ] Docker fused TensorRT smoke test passes
[ ] Docker image is pushed
[ ] GitHub CPU CI passes on Windows and Ubuntu
[ ] release tag is created
```

---

# References

- NVIDIA TensorRT container release notes:
  https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/index.html

- NVIDIA NGC authentication:
  https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html

- Docker Desktop GPU support:
  https://docs.docker.com/desktop/features/gpu/

- NVIDIA Container Toolkit:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/

- ONNX Runtime:
  https://onnxruntime.ai/

- TensorRT:
  https://developer.nvidia.com/tensorrt

- OpenCV:
  https://opencv.org/

- BDD100K:
  https://bdd-data.berkeley.edu/