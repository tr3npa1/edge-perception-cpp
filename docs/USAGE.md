# Setup and usage

[Overview](../README.md) · [Backend contract](backend_matrix.md) ·
[Benchmarks](BENCHMARKING.md)

Commands run from the repository root unless stated otherwise. Paths below are
installation examples: replace them with existing directories on your machine.
Weights, datasets, SDKs and sample media are not included in a Git clone.

## Prerequisites

| Component | Requirement / measured configuration |
|---|---|
| Build | CMake 3.24+, Ninja, C++17 compiler |
| C++ image processing | OpenCV with core, imgproc, imgcodecs and videoio; measured with 4.11.0 |
| C++ inference | ONNX Runtime C/C++ SDK; measured with GPU 1.27.0 |
| NVIDIA extension | Compatible driver, CUDA Toolkit and TensorRT C++ SDK |
| Measured NVIDIA stack | MSVC 19.39, CUDA Toolkit 12.8.93, TensorRT 10.9.0.34, SM 75 |
| Python | Python 3.11 recommended; project requirements files |
| Data/model | Exported model plus readable input; training/evaluation also need labels and checkpoint |

The measured ORT GPU package uses a separate CUDA 13 runtime, while native
TensorRT is linked through CUDA 12.8. Both provider dependency sets must be
available; a toolkit path alone does not satisfy all runtime DLL dependencies.
Python OpenCV and the native C++ OpenCV SDK are separate installations.

## Python environments

For lightweight CPU tests:

```powershell
py -3.11 -m venv .venv-test
.\.venv-test\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv-test\Scripts\python.exe -m pip check
```

For training, export and full GPU evaluation:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-gpu.txt
python -m pip check
$env:YOLO_AUTOINSTALL = "false"
```

Install the TensorRT Python bindings matching the target SDK separately;
`requirements-tensorrt.txt` documents that boundary. Check the actual environment:

```powershell
python -c "import torch, onnxruntime as ort, tensorrt as trt; print(torch.__version__, torch.cuda.is_available()); print(ort.__version__, ort.get_available_providers()); print(trt.__version__)"
```

An advertised provider is not proof that its session can initialize. The
real-model smoke checks exercise loading and inference. Package installation is
not performed automatically by benchmark launchers.

## Native builds

### Windows CPU build

Use an x64 MSVC developer shell. Extract an ONNX Runtime **CPU C++ SDK** and
install OpenCV C++/Ninja, then set their paths:

```powershell
$env:EDGE_ONNXRUNTIME_ROOT = "C:\deps\onnxruntime-win-x64-1.27.0"
$env:EDGE_ONNXRUNTIME_RUNTIME_DIR = "$env:EDGE_ONNXRUNTIME_ROOT\lib"
$env:EDGE_OPENCV_DIR = "C:\opencv\build\x64\vc16\lib"
$env:Path = "$env:EDGE_ONNXRUNTIME_RUNTIME_DIR;C:\opencv\build\x64\vc16\bin;" + $env:Path

cmake --preset cpu-release -DPython3_EXECUTABLE="$PWD/.venv-test/Scripts/python.exe"
cmake --build --preset cpu-release
ctest --preset cpu-release --output-on-failure
```

Executable: `build/cpu-release/bin/Release/edge_perception.exe`.
This profile does not require CUDA or a trained model for its generated fixtures.

### Linux CPU build

Install a C++17 compiler, CMake, Ninja, OpenCV development files and an extracted
Linux ONNX Runtime CPU SDK. With those dependencies present:

```bash
python3.11 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt

export EDGE_ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.27.0
export LD_LIBRARY_PATH="$EDGE_ONNXRUNTIME_ROOT/lib:${LD_LIBRARY_PATH:-}"

cmake --preset cpu-release -DPython3_EXECUTABLE="$PWD/.venv-test/bin/python"
cmake --build --preset cpu-release
ctest --preset cpu-release --output-on-failure
./build/cpu-release/bin/Release/edge_perception --version
```

The CI workflow defines Windows/Linux CPU builds. Published GPU measurements
are Windows-only; local results do not certify a new remote CI run.

### NVIDIA build

Use an x64 developer shell with a CUDA-supported MSVC toolset. The measured
build used MSVC 14.39. Set the paths for your actual SDK installations:

```powershell
$env:EDGE_ONNXRUNTIME_ROOT = "C:\deps\onnxruntime-win-x64-gpu_cuda13-1.27.0"
$env:EDGE_ONNXRUNTIME_RUNTIME_DIR = "$env:EDGE_ONNXRUNTIME_ROOT\lib"
$env:EDGE_TENSORRT_ROOT = "C:\deps\TensorRT-10.9.0.34"
$env:EDGE_CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:EDGE_OPENCV_DIR = "C:\opencv\build\x64\vc16\lib"
$env:EDGE_CXX_COMPILER = (Get-Command cl.exe).Source
$env:EDGE_MSVC_COMPILER = $env:EDGE_CXX_COMPILER
$env:EDGE_CUDA_ARCHITECTURES = "75"

$runtimePaths = @(
    "$env:EDGE_ONNXRUNTIME_ROOT\lib"
    "$env:EDGE_TENSORRT_ROOT\lib"
    "$env:EDGE_TENSORRT_ROOT\bin"
    "$env:EDGE_CUDA_ROOT\bin"
    "C:\opencv\build\x64\vc16\bin"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
    "C:\deps\cudnn9-runtime"
)
$env:EDGE_RUNTIME_DIRS = $runtimePaths -join ";"
$env:Path = $env:EDGE_RUNTIME_DIRS + ";" + $env:Path

cmake --preset nvidia-release `
    -DCMAKE_CXX_COMPILER="$env:EDGE_CXX_COMPILER" `
    -DCMAKE_CUDA_ARCHITECTURES=75 `
    -DPython3_EXECUTABLE="$PWD/.venv/Scripts/python.exe"
cmake --build --preset nvidia-release
ctest --preset nvidia-release --output-on-failure
```

Adjust architecture 75 for a different GPU. Explicitly check every runtime path;
CUDA/cuDNN locations vary by package. Start a new build directory if the compiler,
CUDA version or SDK roots change. The NVIDIA preset keeps fast math and IPO off;
[alternative build profiles](BUILD_MAX_PERFORMANCE.md) require separate validation.

To register the real FP16 GPU test, reconfigure with existing fixtures:

```powershell
cmake --preset nvidia-release `
    -DEDGE_GPU_TEST_ENGINE="$PWD/models/engine/yolo26m_bdd100k_fp16.engine" `
    -DEDGE_GPU_TEST_IMAGE="$PWD/samples/frame.jpg"
cmake --build --preset nvidia-release
ctest --preset nvidia-release --output-on-failure
```

Without those fixture settings, the portable CTest targets do not validate GPU
inference. The full benchmark launcher supplies its own real fixtures.

## Dataset, checkpoint and artifacts

The download script targets `a7madmostafa/bdd100k-yolo` through Kaggle.
Configure access required by the Kaggle CLI and review dataset terms before
downloading; never commit API credentials.

```powershell
python data/download.py
python data/dataset.py
```

The prepared metadata is `data/processed/bdd100k_yolo/bdd100k.yaml`.
Use each script's `--help` for overrides and validation-only options.
The reported 20,000-image split is this prepared dataset's test split, not an
official BDD100K leaderboard submission.

If no trained checkpoint is available, training is an explicit, potentially
expensive prerequisite—not part of a benchmark command:

```powershell
python training/train_detector.py --data data/processed/bdd100k_yolo/bdd100k.yaml --device 0
```

Default checkpoint: `runs/train/yolo26m_bdd100k/weights/best.pt`.
Training a new checkpoint does not reproduce the published artifact hashes.

Export FP32/FP16 on the target NVIDIA stack:

```powershell
python training/export_model.py `
    --weights runs/train/yolo26m_bdd100k/weights/best.pt `
    --data data/processed/bdd100k_yolo/bdd100k.yaml `
    --variants onnx_fp32 onnx_fp16 engine_fp32 engine_fp16
```

Use `--dry-run` to inspect export actions first. Build TensorRT engines for the
target OS/GPU/runtime. Use the [INT8 workflow](INT8_DEPLOYMENT.md), not an
unvalidated generic quantization export, for canonical INT8 artifacts.

## Inference

Show all supported options:

```powershell
.\build\nvidia-release\bin\Release\edge_perception.exe --help
```

Single image on CPU:

```powershell
.\build\cpu-release\bin\Release\edge_perception.exe `
    --mode infer --source samples/frame.jpg `
    --backend ort-cpu --model models/onnx/yolo26m_bdd100k_fp32.onnx `
    --precision fp32 --output-dir outputs/image
```

Directory inference with fused FP16 and JSON output only:

```powershell
.\build\nvidia-release\bin\Release\edge_perception.exe `
    --mode infer --source samples/images `
    --backend fused-trt --model models/engine/yolo26m_bdd100k_fp16.engine `
    --precision fp16 --pipeline-depth 3 `
    --json-only --output-dir outputs/directory
```

Without `--json-only`, annotated outputs are also saved. Directory JSON output
is `detections.jsonl`; `--json-only` requires saving to remain enabled.
Use `--source samples/video.mp4` for video or `--source camera:0` for a camera.
Those media paths include decode/output work absent from the published timing.

Supported images: JPG/JPEG, PNG, BMP, PPM, TIF/TIFF and WEBP. The model and input
must exist. Explicit backends do not silently reroute on failure; `auto` is an
ONNX session-creation fallback policy, not retry-on-any-GPU-error logic.

## Tests

```powershell
python -m compileall -q data training tests tools
python -m pytest -q tests/test_parity.py -m "not integration"
python -m pytest -q tests/test_all_models.py tests/test_int8_workflow.py tests/test_fp16_matrix.py
ctest --preset nvidia-release --output-on-failure
```

Full workflow tests use the project GPU Python environment. Portable parity and
generated ONNX tests use `requirements-test.txt`. Regression checks cover
preprocessing geometry/layout/rounding, decoding and coordinate restoration,
engine envelopes, graph determinism, slot ownership/backpressure/stale tickets,
native worker concurrency and checkpoint integrity.

The optional real PyTorch/ONNX/C++ integration test requires explicit paths:

```powershell
$env:EDGE_TEST_WEIGHTS = "runs/train/yolo26m_bdd100k/weights/best.pt"
$env:EDGE_TEST_ONNX_MODEL = "models/onnx/yolo26m_bdd100k_fp32.onnx"
$env:EDGE_TEST_IMAGE = "samples/frame.jpg"
$env:EDGE_TEST_EXECUTABLE = "build/cpu-release/bin/Release/edge_perception.exe"
$env:EDGE_TEST_DEVICE = "cpu"
python -m pytest -q tests/test_parity.py -m integration -s
```

Unit-test success, strict artifact parity and full-dataset quality are distinct
checks. See the [measured acceptance boundaries](RESULTS_ANALYSIS.md).

## Benchmarking and containers

Use [BENCHMARKING.md](BENCHMARKING.md) for the one-command full matrix,
required canonical artifacts, timing scopes and resume behavior.

Use [DOCKER.md](DOCKER.md) to build the NVIDIA Linux image, generate Linux
engines and run inference with mounted inputs and outputs. Container build and
targeted execution checks are separate from the published Windows benchmark;
they do not establish full-dataset Linux quality or performance. Do not reuse
Windows TensorRT engines inside Linux.
