# edge-perception-cpp — maximum practical pre-benchmark optimization audit

This package is a drop-in optimized core for the files completed before
`benchmark.hpp/.cpp`, `main.cpp`, and the final C++ tests.

It contains the original generic CPU/ONNX Runtime/native-TensorRT path and a
second, specialized native TensorRT fast path designed for the lowest practical
camera/image-to-detection latency available within the project's current
OpenCV, CUDA, TensorRT, and ONNX Runtime dependency set.

The honest target is to outperform the complete Python/Ultralytics TensorRT
wrapper path end to end. It does not claim to make TensorRT's selected model
kernels faster than TensorRT itself.

## Packaged source files

- `CMakeLists.txt`
- `include/image_processor.hpp`
- `src/image_processor.cpp`
- `include/postprocess.hpp`
- `src/postprocess.cpp`
- `include/inference_engine.hpp`
- `src/inference_engine.cpp`
- `include/gpu_pipeline.hpp`
- `src/gpu_pipeline_common.cpp`
- `src/gpu_pipeline.cu`
- `src/gpu_pipeline_stub.cpp`

## Maximum-latency path

The new `NativeTensorRtPipeline` is independent from the generic
`InferenceEngine` so the safe multi-backend interface remains available while
the TensorRT path can specialize aggressively.

### Ordinary OpenCV BGR input

```text
cv::Mat BGR
→ copy into persistent pinned slot
→ asynchronous packed H2D
→ one fused CUDA preprocessing kernel
→ TensorRT enqueueV3
→ one fused CUDA postprocessing kernel
→ one compact asynchronous D2H
→ fixed-capacity DetectionBuffer
```

### Page-locked host input

```text
externally filled pinned BGR memory
→ asynchronous 2D H2D without the extra CPU staging copy
→ fused GPU pipeline
```

### Device-resident input

```text
CUDA BGR / RGB / NV12 surface
→ fused resize + letterbox + color conversion + normalization + NCHW
→ TensorRT
→ fused postprocess
→ compact result only
```

The device path is the intended bridge for NVDEC, camera DMA, CUDA-enabled
GStreamer, or another decoder/capture layer that can provide stable CUDA
surfaces. It removes the host image transfer entirely.

## Fused CUDA preprocessing

One kernel performs:

- half-pixel bilinear resize;
- exact project letterbox geometry and padding;
- BGR, RGB, or NV12 input sampling;
- BT.601 or BT.709 NV12 conversion;
- full-range or limited-range NV12 handling;
- RGB/BGR output ordering;
- pixel scaling, mean subtraction, and standard-deviation normalization;
- direct planar NCHW writes into the TensorRT device input;
- direct FP32 or FP16 output;
- pairwise vectorized `float2` or `half2` stores.

There is no intermediate resized GPU image, letterbox canvas, normalized HWC
buffer, host NCHW tensor, or second device-input copy.

## Fused GPU postprocessing

The model already emits final NMS-free `[1,300,6]` rows. One CUDA block:

- applies the strict confidence threshold;
- uses descending-confidence early termination when enabled;
- validates confidence, coordinates, and class IDs;
- restores original-image coordinates;
- clips boxes;
- discards degenerate boxes;
- preserves model order through a shared-memory prefix scan;
- compacts accepted detections;
- transfers one fixed, small result structure to pinned host memory.

The CPU postprocessor remains selectable for parity and debugging.

## Pipeline overlap

The specialized pipeline owns a configurable ring of 1–8 slots, defaulting to
two. Every slot owns:

- a TensorRT execution context;
- exact user-managed context activation memory;
- a non-default, non-blocking, high-priority main CUDA stream;
- explicit non-blocking auxiliary streams when the engine contains them;
- a completion event;
- pinned host staging/result storage;
- persistent device source, model I/O, and postprocess storage;
- a slot-specific CUDA graph.

Separate slots allow frame N+1 preparation or transfer to overlap frame N
inference where the GPU and copy engines permit it. There is no steady-state
allocation.

## CUDA graph capture

After one ordinary warm-up run per slot and source geometry, the pipeline can
capture:

```text
H2D image copy, when required
→ fused preprocessing
→ TensorRT enqueueV3 and auxiliary-stream work
→ fused postprocessing
→ compact D2H
→ completion event
```

Graph keys include source format, dimensions, pitches, and external pointers.
A changed external surface safely causes recapture instead of replaying stale
addresses. Capture failure falls back to the ordinary asynchronous path unless
`require_cuda_graph=true`.

## TensorRT runtime tuning

- Tensor names, locations, formats, data types, and fixed shapes are validated.
- Persistent input/output addresses are bound once.
- `ExecutionContextAllocationStrategy::kUSER_MANAGED` is used.
- `updateDeviceMemorySizeForShapes()` determines the exact context-memory
  requirement for the resolved shape.
- Context memory is allocated once and supplied with `setDeviceMemoryV2()`.
- Engine auxiliary streams are explicitly supplied as non-blocking streams.
- Enqueue profiling is disabled and runtime NVTX verbosity is lowered on the
  latency path.
- An optional per-context persistent L2 activation-cache limit is exposed.
- CUDA streams use the highest available non-real-time priority by default.
- Native FP32 and FP16 external bindings are supported.

## Existing generic-path optimizations retained

### CPU preprocessing

- reusable OpenCV workspace;
- direct resize into the final letterbox ROI;
- padding applied only to border regions;
- lookup-table normalization and FP16 conversion;
- serial or OpenCV worker-pool HWC-to-NCHW conversion;
- direct writes into caller-owned model input storage;
- exact Python ties-to-even and Ultralytics-style letterbox geometry.

### CPU postprocessing

- fixed-capacity allocation-free detection storage;
- confidence-sorted early termination;
- frame-level precomputation of inverse scales and clipping bounds;
- no duplicate NMS or sorting;
- direct-overwrite storage API for compact GPU results.

### ONNX Runtime

- I/O binding with persistent input/output values;
- pinned host allocation request for GPU sessions with safe fallback;
- tuned CUDA provider controls;
- TensorRT EP engine/timing caches;
- TensorRT EP context-memory sharing and CUDA graph controls;
- CUDA EP fallback behind TensorRT EP;
- selective output synchronization;
- strict isolation between the CUDA runtime owned by ORT and the separately
  linked CUDA runtime used by native TensorRT.

## Build-system optimization

- C++17 and CUDA 17;
- Release `/O2` or `-O3` optimization;
- IPO/LTO where supported;
- dead-code/data elimination;
- optional native CPU ISA;
- `-O3` and extra device vectorization for CUDA;
- optional fast math for separately labelled speed experiments;
- explicit CUDA architecture selection;
- conditional CUDA language activation;
- core-only configuration succeeds before benchmark/main/test files exist;
- future benchmark, application, and test sources are incorporated
  automatically when those finalized files are added.

## Deliberate boundaries

No implementation can honestly contain every optimization possible on every
future machine. The package includes all practical latency optimizations that
fit the current fixed model contract and current dependency stack.

Not bundled:

- NVDEC, nvJPEG, DeepStream, GStreamer, or camera-vendor acquisition code;
- rebuilding the TensorRT engine with new tactics, plugins, or architecture-
  specific global build-route searches;
- operating-system process priority, CPU affinity, GPU clock locking, or power
  controls;
- benchmark claims.

Those are environment/application or engine-build concerns rather than missing
copies in the current inference core. The device BGR/RGB/NV12 API is already
present so an eventual decoder can feed the optimized path without redesigning
the inference layer.

## Validation completed in this package

- Public-header C++17 strict-warning compilation.
- `gpu_pipeline_common.cpp`, `gpu_pipeline_stub.cpp`, and `postprocess.cpp`
  strict-warning compilation.
- CUDA translation-unit host/API syntax parsing for the enabled TensorRT branch
  against API-shaped CUDA 12/TensorRT 10/OpenCV stubs.
- CMake configure and generate test for the core-only non-TensorRT build.
- Archive integrity test and SHA-256 manifest generation.

This environment cannot perform the final MSVC + NVCC + real TensorRT 10.9 +
RTX 2060 build or execute GPU kernels. Real compilation, parity tests, and
latency measurement on the target Windows machine remain required before any
performance claim is made.
