# edge-perception-cpp

High-performance YOLO26M deployment for automotive and industrial vision using BDD100K, PyTorch, MLflow, ONNX Runtime, TensorRT, CUDA, OpenCV, CMake, and native C++17.

The repository covers the complete path from dataset preparation and training through deployment export, validation, native inference, fused GPU execution, and reproducible latency/throughput benchmarking.

```text
BDD100K
→ YOLO26M training
→ MLflow experiment tracking
→ FP32 / FP16 / INT8 export
→ Python artifact validation
→ native C++ deployment
→ ONNX Runtime or TensorRT execution
→ ordered final detections
→ benchmark and parity reports
```

## Status

The Windows C++ deployment matrix has been executed and validated on the target RTX 2060 system.

- 12 valid backend/artifact cells
- 12 successful smoke tests
- 50 successful benchmark rows
- 62/62 successful processes
- 0 recorded failures
- release benchmark build configured with `EDGE_BUILD_TESTS=OFF`; runtime smoke tests provide acceptance coverage
- FP32 and FP16 final-detection count parity across all validated GPU paths
- repaired ORT TensorRT CUDA-graph path validated for FP32 and FP16
- fused TensorRT FP16: **5.03 ms mean end-to-end latency**
- fused TensorRT FP16 depth 3: **254.1 FPS sustained throughput**

The final benchmark archive was:

```text
full_matrix_20260723_204901
```

The ORT TensorRT rows were rerun after the CUDA-graph device-I/O correction and merged into the final package. The repair is recorded in the package manifest and repair metadata.


## Quick start: Windows

This is the shortest validated path for a fresh PowerShell session. It assumes that the repository, model artifacts, BDD100K validation images, and native SDKs already exist locally.

### 1. Create the Python environment

Recommended Python version: 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned PyTorch build supplies the CUDA 13 and cuDNN 9 runtime DLLs used by ONNX Runtime GPU. TensorRT Python bindings are intentionally not installed by `requirements.txt`; the C++ TensorRT paths use the native SDK.

### 2. Define the local dependency roots

Adjust only these paths when your installation layout differs:

```powershell
$ProjectRoot = (Resolve-Path ".").Path

$env:EDGE_VS_ROOT = `
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"

$env:EDGE_ONNXRUNTIME_ROOT = `
    "E:\cpp-libs\onnxruntime-win-x64-gpu_cuda13-1.27.0"

$env:EDGE_TENSORRT_ROOT = `
    "E:\cpp-libs\TensorRT-10.9.0.34"

$env:EDGE_CUDA12_ROOT = `
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

$env:EDGE_OPENCV_DIR = `
    "C:\opencv\build\x64\vc16\lib"

$env:EDGE_OPENCV_BIN = `
    "C:\opencv\build\x64\vc16\bin"
```

The native TensorRT and fused paths are compiled with CUDA 12.8. The ONNX Runtime 1.27 GPU package used here targets CUDA 13, but a standalone CUDA 13 Toolkit is not required when the matching CUDA 13 and cuDNN 9 runtime DLLs are already supplied by the Python/PyTorch environment.

### 3. Activate MSVC 14.39 in PowerShell

```powershell
$VcVarsAll = Join-Path `
    $env:EDGE_VS_ROOT `
    "VC\Auxiliary\Build\vcvarsall.bat"

$EnvironmentLines = cmd.exe /d /s /c `
    "`"$VcVarsAll`" amd64 -vcvars_ver=14.39 >nul && set"

foreach ($Line in $EnvironmentLines) {
    $Parts = $Line -split "=", 2

    if ($Parts.Count -eq 2) {
        [Environment]::SetEnvironmentVariable(
            $Parts[0],
            $Parts[1],
            "Process"
        )
    }
}

$env:EDGE_MSVC_COMPILER = (Get-Command cl.exe).Source
$env:CUDAHOSTCXX = $env:EDGE_MSVC_COMPILER

cl
```

The compiler output must identify MSVC 19.39.

### 4. Configure and build

```powershell
Remove-Item -Recurse -Force build-final -ErrorAction SilentlyContinue

cmake -S . -B build-final -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CXX_COMPILER="$env:EDGE_MSVC_COMPILER" `
    -DCMAKE_CUDA_COMPILER="$env:EDGE_CUDA12_ROOT\bin\nvcc.exe" `
    -DCMAKE_CUDA_HOST_COMPILER="$env:EDGE_MSVC_COMPILER" `
    -DCUDAToolkit_ROOT="$env:EDGE_CUDA12_ROOT" `
    -DOpenCV_DIR="$env:EDGE_OPENCV_DIR" `
    -DONNXRUNTIME_ROOT="$env:EDGE_ONNXRUNTIME_ROOT" `
    -DONNXRUNTIME_RUNTIME_DIR="$env:EDGE_ONNXRUNTIME_ROOT\lib" `
    -DTENSORRT_ROOT="$env:EDGE_TENSORRT_ROOT" `
    -DEDGE_ENABLE_TENSORRT=ON `
    -DEDGE_ENABLE_CUDA_PIPELINE=ON `
    -DEDGE_CUDA_ARCHITECTURES=75 `
    -DEDGE_ENABLE_IPO=ON `
    -DEDGE_ENABLE_NATIVE_ARCH=ON `
    -DEDGE_ENABLE_FAST_MATH=OFF `
    -DEDGE_BUILD_TESTS=OFF

cmake --build build-final `
    --target edge_perception `
    --parallel
```

Expected executable:

```text
build-final\bin\Release\edge_perception.exe
```

### 5. Discover runtime DLL directories

Run this in every fresh PowerShell process before launching GPU backends:

```powershell
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

$SitePackages = (& $Python -c `
    "import site; print(site.getsitepackages()[0])").Trim()

$TorchLib = (& $Python -c `
    "import pathlib, torch; print((pathlib.Path(torch.__file__).resolve().parent / 'lib').resolve())").Trim()

$TensorRtDllDirs = @(
    Get-ChildItem `
        -LiteralPath $env:EDGE_TENSORRT_ROOT `
        -Recurse `
        -File `
        -Filter "*.dll" |
    ForEach-Object { $_.Directory.FullName } |
    Sort-Object -Unique
)

$PythonNvidiaRoot = Join-Path $SitePackages "nvidia"
$PythonNvidiaDllDirs = @()

if (Test-Path -LiteralPath $PythonNvidiaRoot -PathType Container) {
    $PythonNvidiaDllDirs = @(
        Get-ChildItem `
            -LiteralPath $PythonNvidiaRoot `
            -Recurse `
            -File `
            -Filter "*.dll" |
        ForEach-Object { $_.Directory.FullName } |
        Sort-Object -Unique
    )
}

$RuntimeDirs = @(
    "$ProjectRoot\build-final\bin\Release"
    "$env:EDGE_ONNXRUNTIME_ROOT\lib"
    $TorchLib
    "$env:EDGE_CUDA12_ROOT\bin"
    $env:EDGE_OPENCV_BIN
) + $TensorRtDllDirs + $PythonNvidiaDllDirs + @($env:PATH)

$env:PATH = (
    $RuntimeDirs |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    Select-Object -Unique
) -join ";"
```

Verify the two dependencies that caused the most common Windows startup failures:

```powershell
where.exe cudnn64_9.dll
where.exe nvonnxparser_10.dll
```

Both commands must print a real file path.

### 6. Run a smoke benchmark

Datasets, trained weights, ONNX files, and TensorRT engines are intentionally excluded from Git. Before running this command, place the required local artifacts at the paths documented in [Deployment artifacts](#deployment-artifacts).

This checks executable startup, ONNX Runtime CUDA loading, model execution, and benchmark file generation:

```powershell
.\build-final\bin\Release\edge_perception.exe `
    --mode benchmark `
    --source .\data\raw\bdd100k\bdd100k\val\images `
    --backend ort-cuda `
    --model .\models\onnx\yolo26m_bdd100k_fp32.onnx `
    --precision fp32 `
    --pipeline-depth 1 `
    --benchmark-mode latency `
    --timing-scope end-to-end `
    --benchmark-source-frames 2 `
    --warmup 2 `
    --iterations 3 `
    --no-stage-timings `
    --retain-samples `
    --output-dir .\outputs\quick_start_smoke
```

A successful run reports:

```text
status: completed
successful: true
```

and writes:

```text
benchmark.json
summary.csv
summary.txt
samples.csv
```

For the maximum-performance path, repeat with:

```text
--backend fused-trt
--model .\models\engine\yolo26m_bdd100k_fp16.engine
--precision fp16
--require-cuda-graph
```


## Measured C++ results

### Benchmark environment

| Component | Validated value |
|---|---|
| Operating system | Windows 11 |
| GPU | NVIDIA GeForce RTX 2060, 6 GB |
| CUDA architecture | 75 |
| Compiler | MSVC 19.39 / toolset 14.39 |
| Native CUDA toolkit | CUDA 12.8 |
| TensorRT | 10.9 |
| ONNX Runtime GPU | 1.27, CUDA 13 package |
| OpenCV | 4.11 |
| Model input | `[1, 3, 640, 640]` |
| Model output | `[1, 300, 6]` |
| Benchmark source frames | 32 BDD100K validation images |
| Confidence threshold | 0.25 |
| Maximum detections | 300 |
| Latency warm-up / measured | 30 / 300 |
| Throughput warm-up / measured | 50 / 1000 |
| Benchmark repetitions | 1 |

Results are machine-specific measurements, not universal guarantees. A release-quality cross-machine report should repeat each configuration multiple times from a clean tagged commit.

### Clean end-to-end latency

Lower is better.

| Path | Precision | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|---|---|
| Fused TensorRT | FP16 | 5.03 | 5.06 | 5.44 | 5.78 |
| Native TensorRT | FP16 | 5.08 | 5.05 | 5.49 | 5.80 |
| ORT TensorRT | FP16 | 6.44 | 5.99 | 8.15 | 8.91 |
| ORT CUDA | FP16 | 11.67 | 11.55 | 12.36 | 12.80 |
| Fused TensorRT | FP32 | 15.72 | 15.53 | 16.72 | 17.22 |
| Native TensorRT | FP32 | 15.96 | 15.71 | 17.12 | 17.59 |
| ORT TensorRT | FP32 | 19.94 | 19.34 | 25.14 | 26.73 |
| ORT CUDA | FP32 | 27.35 | 27.24 | 28.15 | 28.80 |
| ORT CUDA | INT8 | 50.30 | 48.61 | 68.58 | 81.60 |
| ORT CPU | FP32 | 244.11 | 243.66 | 264.35 | 271.94 |
| ORT CPU | FP16 | 250.75 | 249.26 | 275.93 | 288.90 |
| ORT CPU | INT8 | 453.74 | 443.53 | 575.64 | 630.64 |

### Sustained throughput

Higher is better. Generic backends are synchronous and therefore use effective depth 1. The fused pipeline supports several frames in flight.

| Path | Precision | Depth | FPS |
|---|---|---|---|
| Fused TensorRT | FP16 | 3 | 254.1 |
| Fused TensorRT | FP16 | 2 | 236.5 |
| Fused TensorRT | FP16 | 1 | 200.4 |
| Native TensorRT | FP16 | 1 | 198.0 |
| ORT TensorRT | FP16 | 1 | 151.1 |
| ORT CUDA | FP16 | 1 | 84.8 |
| Fused TensorRT | FP32 | 3 | 70.5 |
| Fused TensorRT | FP32 | 2 | 68.2 |
| Fused TensorRT | FP32 | 1 | 63.1 |
| Native TensorRT | FP32 | 1 | 62.0 |
| ORT TensorRT | FP32 | 1 | 53.1 |
| ORT CUDA | FP32 | 1 | 35.3 |
| ORT CUDA | INT8 | 1 | 23.2 |
| ORT CPU | FP32 | 1 | 4.4 |
| ORT CPU | FP16 | 1 | 4.4 |
| ORT CPU | INT8 | 1 | 2.4 |

### Native TensorRT versus fused TensorRT

Both paths execute the same serialized TensorRT engine.

The generic native path uses CPU preprocessing, pinned host tensors, asynchronous copies, TensorRT `enqueueV3`, CUDA-graph replay, synchronous completion, and CPU postprocessing.

The fused path owns persistent per-slot host/device resources and executes GPU preprocessing, TensorRT inference, GPU postprocessing, and compact result transfer as one asynchronous pipeline.

| FP16 comparison | Native TensorRT | Fused TensorRT | Fused change |
|---|---:|---:|---:|
| Mean depth-1 latency | 5.0788 ms | 5.0330 ms | 0.9% lower |
| P95 depth-1 latency | 5.4920 ms | 5.4435 ms | 0.9% lower |
| Depth-1 throughput | 198.0 FPS | 200.4 FPS | 1.2% higher |
| Maximum tested throughput | 198.0 FPS | 254.1 FPS at depth 3 | 28.4% higher |

The defensible conclusion is:

> The fused FP16 pipeline preserves native TensorRT single-frame latency while increasing sustained throughput by 28.4% at depth 3.

The depth-1 latency difference is too small to claim a major latency victory. The meaningful improvement is better pipeline utilization and multi-frame concurrency.

Depth 3 increases throughput by keeping several frames in flight. It does not reduce individual frame completion latency; the depth-3 FP16 run had approximately 15.16 ms mean and 22.96 ms P95 completion latency.

Use:

- depth 1 for minimum per-frame latency;
- depth 2 for balanced live-stream operation;
- depth 3 for maximum measured throughput on this RTX 2060;
- depths 4–8 only as explicit target-hardware experiments.

### FP16 conclusion

FP16 is the preferred deployment precision on the tested GPU.

For native TensorRT:

- FP32 mean latency: 15.96 ms
- FP16 mean latency: 5.08 ms
- FP16 latency reduction: approximately 68%
- FP32 throughput: 62.0 FPS
- FP16 throughput: 198.0 FPS

The FP16 engine is also roughly half the size of the FP32 engine.

### INT8 status

The current ONNX all-convolution INT8 artifact is **not deployment-ready** in this Windows C++ stack.

Observed results:

- ORT CUDA INT8: 50.30 ms mean, 23.2 FPS
- ORT CPU INT8: 453.74 ms mean, 2.4 FPS
- FP32 smoke baseline: 36 detections
- INT8 ORT CPU smoke: 24 detections
- INT8 ORT CUDA smoke: 26 detections

The current TensorRT INT8 engine was excluded from native/fused benchmarking because it is incompatible with the validated Windows TensorRT runtime. Do not publish an INT8 speedup claim from this build.

## Project scope

The C++ application supports:

- ONNX Runtime CPU;
- ONNX Runtime CUDA Execution Provider;
- ONNX Runtime TensorRT Execution Provider with CUDA fallback;
- native TensorRT through the generic synchronous inference engine;
- fused native TensorRT CUDA execution;
- image, recursive image-directory, video, and camera sources;
- synchronous end-to-end and backend-only latency benchmarks;
- bounded streaming-throughput benchmarks;
- fused pipeline depths from 1 through 8;
- JSON, CSV, text, JSONL, image, and video outputs;
- CUDA graph replay where supported;
- deterministic ordered postprocessing for the exported final-detection head.

## Fixed model contract

The native C++ code intentionally targets one deployment contract:

```text
input name:       images
input shape:      [1, 3, 640, 640]
input layout:     NCHW
input batch:      1
output name:      output0
output shape:     [1, 300, 6]
output row:       [x1, y1, x2, y2, confidence, class_id]
host output type: FP32
```

The exported head already returns confidence-ranked, limited final detections. Therefore the postprocessor:

- does not run another NMS;
- does not sort detections again;
- preserves model row order;
- restores boxes from model coordinates to original-image coordinates;
- applies the confidence threshold;
- validates class IDs and malformed rows;
- clips boxes when configured;
- rejects invalid or degenerate boxes.

Adding another NMS or sorting pass would alter exported-model semantics and invalidate parity.

### BDD100K class order

| ID | Class |
|---:|---|
| 0 | person |
| 1 | rider |
| 2 | car |
| 3 | bus |
| 4 | truck |
| 5 | bike |
| 6 | motor |
| 7 | traffic light |
| 8 | traffic sign |
| 9 | train |

## Repository layout

```text
edge-perception-cpp/
├── data/
│   ├── download.py
│   └── dataset.py
├── training/
│   ├── train_detector.py
│   ├── export_model.py
│   └── evaluate.py
├── include/
│   ├── benchmark.hpp
│   ├── gpu_pipeline.hpp
│   ├── image_processor.hpp
│   ├── inference_engine.hpp
│   └── postprocess.hpp
├── src/
│   ├── benchmark.cpp
│   ├── gpu_pipeline.cu
│   ├── gpu_pipeline_common.cpp
│   ├── gpu_pipeline_stub.cpp
│   ├── image_processor.cpp
│   ├── inference_engine.cpp
│   ├── main.cpp
│   └── postprocess.cpp
├── tests/
├── CMakeLists.txt
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

`gpu_pipeline_stub.cpp` is intentional. It provides a controlled runtime error when the fused CUDA pipeline is disabled at build time and must not be deleted.

Datasets, model binaries, build trees, MLflow state, runtime caches, inference output, and benchmark archives are intentionally excluded from Git.

## Deployment artifacts

Expected local paths:

```text
runs/train/yolo26m_bdd100k/weights/best.pt

models/onnx/yolo26m_bdd100k_fp32.onnx
models/onnx/yolo26m_bdd100k_fp16.onnx
models/onnx/yolo26m_bdd100k_allconv_int8.onnx

models/engine/yolo26m_bdd100k_fp32.engine
models/engine/yolo26m_bdd100k_fp16.engine
models/engine/yolo26m_bdd100k_int8.engine
```

TensorRT `.engine` files are hardware- and runtime-sensitive. Rebuild them after changing GPU architecture, TensorRT version, operating system, or incompatible CUDA/runtime components.

Validated artifact hashes are recorded in the final benchmark package under:

```text
metadata/model_hashes.csv
```

## Python-side model-quality results

These are Python/Ultralytics evaluation measurements. They are model-quality and export-validation results, not the C++ end-to-end timings shown above.

| Artifact | Precision | Recall | mAP@50 | mAP@50–95 | Python inference |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | 0.756 | 0.529 | 0.602 | 0.341 | 27.65 ms |
| ONNX FP32 | 0.731 | 0.544 | 0.601 | 0.340 | 38.59 ms |
| ONNX FP16 | 0.731 | 0.544 | 0.600 | 0.340 | 15.01 ms |
| ONNX all-Conv INT8 | 0.729 | 0.490 | 0.538 | 0.286 | 57.03 ms |
| TensorRT FP32 | 0.731 | 0.544 | 0.601 | 0.340 | 23.13 ms |
| TensorRT FP16 | 0.731 | 0.544 | 0.600 | 0.340 | 6.58 ms |

The Python and C++ timing scopes differ and must not be compared without labeling the scope.

# C++ architecture

## Generic inference path

`InferenceEngine` supports ORT CPU, ORT CUDA, ORT TensorRT, and native TensorRT.

```text
OpenCV BGR frame
→ reusable CPU letterbox/preprocessing workspace
→ caller-owned NCHW tensor
→ synchronous InferenceEngine::run()
→ CPU-readable FP32 [1,300,6]
→ ordered CPU postprocessing
→ DetectionBuffer
```

Key properties:

- reusable input/output storage;
- direct preprocessing into the engine input tensor;
- ONNX Runtime I/O binding where appropriate;
- native TensorRT pinned host buffers and persistent device buffers;
- explicit TensorRT execution-context memory;
- CUDA graph replay in native TensorRT;
- fixed synchronous public completion contract;
- FP32 and IEEE-754 binary16 input support;
- no second NMS.

For ORT GPU backends, backend-only timing includes transfers and synchronization owned by the synchronous `run()` operation.

### ORT TensorRT CUDA-graph device I/O

The validated Windows ORT package uses CUDA 13, while native/fused TensorRT is linked against CUDA 12.8.

For ORT TensorRT CUDA-graph execution, the implementation:

- allocates persistent device input/output tensors through the ORT session;
- binds those stable device addresses once;
- copies each host input into the persistent ORT device tensor;
- replays graph ID 0;
- synchronizes the bound output;
- copies the device output back to the stable host tensor.

The Windows bridge dynamically resolves `cudaSetDevice`, `cudaMemcpy`, and `cudaGetErrorString` from `cudart64_13.dll`. This avoids passing CUDA 12.8 runtime objects into the CUDA 13 ORT provider.

`cudart64_13.dll` must therefore be discoverable through `PATH` when ORT TensorRT CUDA graphs are enabled.

The explicit ORT CUDA-graph copy bridge is currently Windows-specific. On non-Windows systems, disable the ORT TensorRT CUDA graph with `--no-cuda-graph` unless an equivalent platform bridge has been implemented. This limitation does not apply to the native or fused CUDA 12.8 TensorRT paths.

## Fused native TensorRT path

For a normal OpenCV frame:

```text
cv::Mat BGR
→ slot-owned pinned staging
→ asynchronous packed-image H2D
→ fused GPU resize + letterbox + color conversion + normalization + NCHW
→ TensorRT enqueueV3
→ ordered GPU postprocessing
→ compact result D2H
→ DetectionBuffer
```

For device-resident BGR, RGB, or NV12 input:

```text
device image
→ fused GPU preprocessing
→ TensorRT
→ GPU postprocessing
→ compact result D2H
```

Each slot owns persistent:

- TensorRT execution context;
- activation/workspace memory;
- main CUDA stream;
- optional TensorRT auxiliary streams;
- completion event;
- pinned host staging/result memory;
- source and model device buffers;
- postprocessing storage;
- CUDA graph state.

No steady-state per-frame slot allocation is required after initialization.


# Windows build and runtime setup

## Validated dependency split

The measured Windows build intentionally uses two CUDA generations:

| Path | Runtime generation | Purpose |
|---|---|---|
| Native TensorRT and fused CUDA pipeline | CUDA Toolkit 12.8 | Compiles and runs native CUDA/TensorRT code |
| ONNX Runtime GPU 1.27 | CUDA 13 runtime ABI | Runs ORT CUDA and ORT TensorRT providers |
| cuDNN | cuDNN 9 | Required by the ORT CUDA provider |
| TensorRT | 10.9 | Native engine execution and ORT TensorRT provider |
| OpenCV | 4.11 | Image/video I/O and generic CPU preprocessing |

The ORT CUDA 13 runtime DLLs may come from the Python/PyTorch installation. Do not require or document a standalone `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0` directory unless it actually exists on the target machine.

The native CUDA 12.8 objects and streams are never passed into the ORT CUDA 13 provider. The project keeps those runtime ownership boundaries explicit.

## Validated local layout

```text
Repository:
E:\tmkc\edge-perception-cpp

Visual Studio Build Tools:
C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools

MSVC toolset:
14.39.33519

OpenCV:
C:\opencv\build\x64\vc16

ONNX Runtime GPU:
E:\cpp-libs\onnxruntime-win-x64-gpu_cuda13-1.27.0

TensorRT:
E:\cpp-libs\TensorRT-10.9.0.34

Native CUDA toolkit:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

ORT CUDA 13 and cuDNN 9 runtime DLLs:
resolved from the active Python/PyTorch environment
```

## Build rules

- Use MSVC 19.39 as both the C++ compiler and CUDA host compiler.
- Configure CUDA architecture 75 for the RTX 2060.
- Keep fast math disabled for reproducibility and parity.
- Keep `EDGE_BUILD_TESTS=OFF` unless the test targets and their dependencies are intentionally configured.
- Delete the build directory when changing compiler, CUDA, TensorRT, ONNX Runtime, or OpenCV roots.
- Do not copy arbitrary DLLs between CUDA installations.

The complete fresh-shell configure/build sequence is provided in [Quick start: Windows](#quick-start-windows).

## Runtime search-path rules

The executable directory contains directly copied project dependencies when CMake can resolve them, but transitive runtime DLLs still need to be discoverable through `PATH`.

The runtime bootstrap must include:

- `build-final\bin\Release`;
- ONNX Runtime `lib`;
- PyTorch `torch\lib` or the equivalent directory containing `cudnn64_9.dll`;
- Python NVIDIA runtime `bin` directories when present;
- every TensorRT directory containing runtime DLLs;
- CUDA 12.8 `bin`;
- OpenCV `bin`.

TensorRT Windows archives do not always place every DLL in the same directory. Discovering the directories recursively is safer than assuming that `nvonnxparser_10.dll` is always under `bin`.


# Command-line application

Show all options:

```powershell
.\build-final\bin\Release\edge_perception.exe --help
```

Core interface:

```text
--mode infer|benchmark
--source <image|directory|video|camera:N>
--backend ort-cpu|ort-cuda|ort-trt|native-trt|fused-trt
--model <path>
--precision fp32|fp16|int8
--device <N>
--confidence <0..1>
--max-detections <1..300>
--output-dir <path>
--pipeline-depth <1..8>
```

Performance controls:

```text
--no-cuda-graph
--require-cuda-graph
--debug-parity
--no-stage-timings
--retain-samples
```

`--debug-parity` is fused-only and deliberately disables GPU postprocessing and CUDA graphs to expose the slower raw-output/CPU-postprocess comparison path.

## Recommended image inference

### Fused TensorRT FP16

```powershell
.\build-final\bin\Release\edge_perception.exe `
    --mode infer `
    --source .\samples\frame.jpg `
    --backend fused-trt `
    --model .\models\engine\yolo26m_bdd100k_fp16.engine `
    --precision fp16 `
    --pipeline-depth 1 `
    --output-dir .\outputs\fused_fp16
```

### Native TensorRT FP16

```powershell
.\build-final\bin\Release\edge_perception.exe `
    --mode infer `
    --source .\samples\frame.jpg `
    --backend native-trt `
    --model .\models\engine\yolo26m_bdd100k_fp16.engine `
    --precision fp16 `
    --output-dir .\outputs\native_fp16
```

### ORT TensorRT FP16

```powershell
.\build-final\bin\Release\edge_perception.exe `
    --mode infer `
    --source .\samples\frame.jpg `
    --backend ort-trt `
    --model .\models\onnx\yolo26m_bdd100k_fp16.onnx `
    --precision fp16 `
    --ort-cache-dir .\models\ort_trt_cache `
    --output-dir .\outputs\ort_trt_fp16
```

### ORT CUDA FP16

```powershell
.\build-final\bin\Release\edge_perception.exe `
    --mode infer `
    --source .\samples\frame.jpg `
    --backend ort-cuda `
    --model .\models\onnx\yolo26m_bdd100k_fp16.onnx `
    --precision fp16 `
    --output-dir .\outputs\ort_cuda_fp16
```

## Video inference

```powershell
.\build-final\bin\Release\edge_perception.exe `
    --mode infer `
    --source .\samples\drive.mp4 `
    --backend fused-trt `
    --model .\models\engine\yolo26m_bdd100k_fp16.engine `
    --precision fp16 `
    --pipeline-depth 2 `
    --output-dir .\outputs\video
```

Use depth 3 for maximum measured throughput when additional queueing latency is acceptable.

## Output files

With saving enabled:

```text
output-dir/
├── detections.jsonl
├── <image>_detections.jpg
└── <video>_detections.mp4
```

Use `--no-save` when measuring inference without annotation and disk-write overhead.

# Benchmarking

## Correctness rules

- warm-up iterations are excluded;
- source frames are preloaded;
- latency and throughput are separate modes;
- fused asynchronous latency is measured through completion, not submission;
- throughput timing includes drain completion;
- failed frames are excluded from the FPS numerator;
- unsupported stage timings remain unavailable instead of becoming zero;
- drawing and disk output are excluded;
- the same artifact, confidence threshold, source set, and output contract are used for comparable paths;
- no benchmark path adds NMS or reorders detections.

## Minimum-latency benchmark

```powershell
.\build-final\bin\Release\edge_perception.exe `
    --mode benchmark `
    --source .\data\raw\bdd100k\bdd100k\val\images `
    --backend fused-trt `
    --model .\models\engine\yolo26m_bdd100k_fp16.engine `
    --precision fp16 `
    --pipeline-depth 1 `
    --benchmark-mode latency `
    --timing-scope end-to-end `
    --benchmark-source-frames 32 `
    --warmup 30 `
    --iterations 300 `
    --no-stage-timings `
    --retain-samples `
    --output-dir .\benchmarks\manual_fused_fp16_latency
```

## Maximum-throughput benchmark

```powershell
.\build-final\bin\Release\edge_perception.exe `
    --mode benchmark `
    --source .\data\raw\bdd100k\bdd100k\val\images `
    --backend fused-trt `
    --model .\models\engine\yolo26m_bdd100k_fp16.engine `
    --precision fp16 `
    --pipeline-depth 3 `
    --benchmark-mode throughput `
    --timing-scope end-to-end `
    --benchmark-source-frames 32 `
    --warmup 50 `
    --iterations 1000 `
    --no-stage-timings `
    --retain-samples `
    --output-dir .\benchmarks\manual_fused_fp16_throughput
```

Each run writes:

```text
summary.txt
benchmark.json
summary.csv
samples.csv          # only with --retain-samples
```

The full matrix runner additionally creates aggregate rankings, smoke parity tables, telemetry, runtime metadata, hashes, checksums, and a packaged archive.


# Pre-commit validation

Run validation from a fresh PowerShell session after completing the runtime bootstrap from the Quick Start section.

## Required checks

```powershell
git diff --check

cmake --build build-final `
    --target edge_perception `
    --clean-first `
    --parallel

.\build-final\bin\Release\edge_perception.exe --version
```

Then run short benchmark smoke tests for these paths:

| Smoke cell | Artifact |
|---|---|
| ORT CPU FP32 | `models/onnx/yolo26m_bdd100k_fp32.onnx` |
| ORT CUDA FP32 | `models/onnx/yolo26m_bdd100k_fp32.onnx` |
| ORT TensorRT FP16 | `models/onnx/yolo26m_bdd100k_fp16.onnx` |
| Native TensorRT FP16 | `models/engine/yolo26m_bdd100k_fp16.engine` |
| Fused TensorRT FP16 depth 1 | `models/engine/yolo26m_bdd100k_fp16.engine` |
| Fused TensorRT FP16 depth 3 throughput | `models/engine/yolo26m_bdd100k_fp16.engine` |

For a smoke run, use two warm-up frames and three measured frames. Every process must:

- return exit code zero;
- report `status: completed`;
- report `successful: true`;
- create non-empty `benchmark.json`, `summary.csv`, and `samples.csv`.

The full 12-cell benchmark matrix remains the publishable performance evidence. Short smoke runs verify that the current checkout still builds, starts, loads its dependencies, executes each major runtime family, and writes benchmark artifacts.

## CTest interpretation

The validated maximum-performance configuration uses:

```text
EDGE_BUILD_TESTS=OFF
```

Therefore:

```text
No tests were found!!!
```

is expected when `ctest` is run against that build tree. It is not a test failure, but it is also not evidence that unit tests passed. Runtime acceptance comes from the executable smoke matrix.



# Troubleshooting

## `cudnn64_9.dll` is missing

Typical error:

```text
Error loading onnxruntime_providers_cuda.dll
which depends on cudnn64_9.dll which is missing
```

Resolve the active PyTorch DLL directory and prepend it to `PATH`:

```powershell
$Python = ".\.venv\Scripts\python.exe"

$TorchLib = (& $Python -c `
    "import pathlib, torch; print((pathlib.Path(torch.__file__).resolve().parent / 'lib').resolve())").Trim()

$env:PATH = "$TorchLib;$env:PATH"

where.exe cudnn64_9.dll
```

When `cudnn64_9.dll` is not present there, locate a matching cuDNN 9 runtime installation and add the directory that actually contains the DLL.

Do not blindly reinstall `onnxruntime-gpu[cuda,cudnn]` to solve this on Windows. CUDA 13 auxiliary packages may not be available from pip for every Python and platform combination.

## `nvonnxparser_10.dll` is missing

Typical error:

```text
Error loading onnxruntime_providers_tensorrt.dll
which depends on nvonnxparser_10.dll which is missing
```

Add every TensorRT directory containing DLLs:

```powershell
$TensorRtDllDirs = @(
    Get-ChildItem `
        -LiteralPath $env:EDGE_TENSORRT_ROOT `
        -Recurse `
        -File `
        -Filter "*.dll" |
    ForEach-Object { $_.Directory.FullName } |
    Sort-Object -Unique
)

$env:PATH = (
    ($TensorRtDllDirs + @($env:PATH)) |
    Select-Object -Unique
) -join ";"

where.exe nvonnxparser_10.dll
where.exe nvinfer_10.dll
where.exe nvinfer_plugin_10.dll
```

If the parser DLL cannot be found anywhere under the TensorRT root, the Windows TensorRT package is incomplete.

## ORT CUDA works but ORT TensorRT fails

ORT CUDA and ORT TensorRT have different transitive dependencies. A successful CUDA-provider run confirms cuDNN/CUDA loading, but ORT TensorRT additionally requires TensorRT runtime and parser DLLs.

Check all of these:

```powershell
where.exe cudnn64_9.dll
where.exe nvinfer_10.dll
where.exe nvinfer_plugin_10.dll
where.exe nvonnxparser_10.dll
```

## TensorRT engine deserialization fails

TensorRT engines are runtime- and hardware-specific. Rebuild the engine when the GPU architecture, TensorRT version, CUDA stack, operating system, or engine build configuration changes.

## `No tests were found!!!`

The release benchmark configuration currently uses `EDGE_BUILD_TESTS=OFF`. See [CTest interpretation](#ctest-interpretation).

## LF/CRLF warnings

Messages such as:

```text
LF will be replaced by CRLF the next time Git touches it
```

are line-ending warnings, not build errors.

Use:

```powershell
git diff --check
```

to detect actual whitespace errors before committing.

## Stale CMake cache

Delete and reconfigure the build tree after changing dependency roots or compiler/toolkit versions:

```powershell
Remove-Item -Recurse -Force build-final
```

Then rerun the configure command from the Quick Start section.

## Benchmark numbers differ from the table

The published table used 32 preloaded BDD100K frames, 30/300 warm-up/measured iterations for latency, and 50/1000 for throughput. Quick-start and pre-commit smoke tests intentionally use much smaller counts and are not performance measurements.


# Docker

The Dockerfile provides a Linux GPU build path. TensorRT engines must be rebuilt for the Linux runtime and target GPU.

```powershell
docker build `
    --build-arg CUDA_ARCHITECTURES=75 `
    --tag edge-perception-cpp:latest `
    .
```

Example:

```powershell
docker run --rm --gpus all `
    --volume "${PWD}\models:/workspace/models:ro" `
    --volume "${PWD}\samples:/workspace/inputs:ro" `
    --volume "${PWD}\outputs:/workspace/outputs" `
    edge-perception-cpp:latest `
    --mode infer `
    --source /workspace/inputs/frame.jpg `
    --backend fused-trt `
    --model /workspace/models/engine/yolo26m_bdd100k_fp16.engine `
    --precision fp16 `
    --pipeline-depth 1 `
    --output-dir /workspace/outputs
```

The host must provide a compatible NVIDIA driver and NVIDIA Container Toolkit.

# Reproducibility

Record for every publishable run:

```text
Git commit and working-tree state
operating system
CPU
GPU
NVIDIA driver
compiler and toolset
CUDA runtime/toolkit versions
cuDNN version
TensorRT version
ONNX Runtime version
OpenCV version
artifact path, size, and SHA-256
executable SHA-256
source SHA-256
precision
backend
pipeline depth
CUDA graph state
GPU postprocessing state
warm-up and measured counts
timing scope
source-frame set
fast-math state
```

The July 23 benchmark was executed from a dirty working tree and then repaired by rerunning the ORT TensorRT subset with the corrected executable. This is transparently recorded in the archive. The measured results are valid for that development state, but a future formal release benchmark should be generated from a clean tagged commit.

The repaired benchmark metadata records:

```text
src/inference_engine.cpp SHA-256:
deb35b762f88d2dfb7aa93389e48e67681eee7244d7c3a1a5dd1015686da69c8
```

The source file in this finalized repository package preserves that exact benchmarked byte sequence.

# Known boundaries

The repository does not bundle:

- BDD100K;
- trained weights or exported model binaries;
- proprietary or redistributable-restricted SDK binaries;
- NVIDIA drivers;
- Visual Studio Build Tools;
- CUDA, cuDNN, TensorRT, ONNX Runtime, or OpenCV installations;
- a universal TensorRT engine compatible with every GPU and runtime;
- NVDEC, DeepStream, GStreamer, or vendor-camera acquisition code;
- a validated real-time CPU deployment path for YOLO26M;
- a validated production INT8 deployment artifact;
- registered CTest targets in the maximum-performance release configuration.

The fused device-image API accepts CUDA BGR, RGB, and NV12 surfaces so a future decoder can connect without redesigning the inference layer.

# References

- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [NVIDIA TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [BDD100K](https://www.bdd100k.com/)
