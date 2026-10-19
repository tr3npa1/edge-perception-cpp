# Windows performance build profiles

The fused pipeline is compiled only when both native TensorRT and the CUDA
pipeline are enabled. For an RTX 2060, compile CUDA code for SM 7.5.

The published measurements use `nvidia-release`, with IPO and fast math off.
The profiles below are optional tuning experiments, not the configuration used
for those measurements. Their effects require fresh correctness and timing checks.

Run from the repository root in a shell where the MSVC compiler environment
is active. Set the dependency environment variables for your installation;
see the [native build instructions](USAGE.md#native-builds).

```powershell
$Build = "build-max"
$env:EDGE_MSVC_COMPILER = (Get-Command cl.exe).Source

cmake -S . -B $Build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CXX_COMPILER="$env:EDGE_MSVC_COMPILER" `
    -DCMAKE_CUDA_COMPILER="$env:EDGE_CUDA_ROOT\bin\nvcc.exe" `
    -DCUDAToolkit_ROOT="$env:EDGE_CUDA_ROOT" `
    -DOpenCV_DIR="C:\opencv\build\x64\vc16\lib" `
    -DONNXRUNTIME_ROOT="$env:EDGE_ONNXRUNTIME_ROOT" `
    -DONNXRUNTIME_RUNTIME_DIR="$env:EDGE_ONNXRUNTIME_ROOT\lib" `
    -DTENSORRT_ROOT="$env:EDGE_TENSORRT_ROOT" `
    -DEDGE_ENABLE_TENSORRT=ON `
    -DEDGE_ENABLE_CUDA_PIPELINE=ON `
    -DCMAKE_CUDA_ARCHITECTURES=75 `
    -DEDGE_ENABLE_IPO=ON `
    -DEDGE_ENABLE_NATIVE_ARCH=ON `
    -DEDGE_ENABLE_FAST_MATH=OFF `
    -DEDGE_BUILD_TESTS=OFF

cmake --build $Build --config Release --parallel
```

This configuration builds the core library, command-line application, and
fused CUDA/TensorRT implementation. Set `EDGE_BUILD_TESTS=ON` for a release
acceptance build.

## Fast-math build

After the primary parity build succeeds, create a separate build directory:

```powershell
cmake -S . -B build-max-fast-math -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CXX_COMPILER="$env:EDGE_MSVC_COMPILER" `
    -DCMAKE_CUDA_COMPILER="$env:EDGE_CUDA_ROOT\bin\nvcc.exe" `
    -DCUDAToolkit_ROOT="$env:EDGE_CUDA_ROOT" `
    -DOpenCV_DIR="C:\opencv\build\x64\vc16\lib" `
    -DONNXRUNTIME_ROOT="$env:EDGE_ONNXRUNTIME_ROOT" `
    -DONNXRUNTIME_RUNTIME_DIR="$env:EDGE_ONNXRUNTIME_ROOT\lib" `
    -DTENSORRT_ROOT="$env:EDGE_TENSORRT_ROOT" `
    -DEDGE_ENABLE_TENSORRT=ON `
    -DEDGE_ENABLE_CUDA_PIPELINE=ON `
    -DCMAKE_CUDA_ARCHITECTURES=75 `
    -DEDGE_ENABLE_IPO=ON `
    -DEDGE_ENABLE_NATIVE_ARCH=ON `
    -DEDGE_ENABLE_FAST_MATH=ON `
    -DEDGE_BUILD_TESTS=OFF
```

Fast math is implemented but not enabled in the primary configuration because
it changes floating-point semantics. Keep its parity and latency results
separate.

## Runtime dependency separation

The ONNX Runtime package is CUDA-13-focused, while native TensorRT 10.9 is
built through CUDA 12.8. The generic ORT backend owns its own provider streams
and allocations. The specialized native TensorRT pipeline owns only CUDA 12.8
objects. Do not pass device pointers or streams from one runtime path into the
other.

## Runtime configuration example

Depth 2 is an example, not a universally optimal setting. The complete
[FP16 matrix](results/fp16-matrix.md) measures native/fused depth-3 throughput
at 254.615/245.906 FPS. Sequential and streaming frame latency are separate
metrics. Fused FP16 passes 64/64 against the same Python engine; strict PyTorch
parity is 61/64. See the [correctness evidence](results/fp16-correctness.md).

```cpp
edge::NativeTensorRtPipelineConfig config{};
config.pipeline_depth = 2U;
config.enable_cuda_graph = true;
config.enable_gpu_postprocess = true;
config.use_high_priority_streams = true;
config.use_engine_auxiliary_streams = true;
config.minimize_runtime_instrumentation = true;
```

Use device-resident NV12/BGR/RGB input whenever the capture or decode layer can
provide it. Otherwise use page-locked BGR input; ordinary `cv::Mat` remains the
convenient fallback.
