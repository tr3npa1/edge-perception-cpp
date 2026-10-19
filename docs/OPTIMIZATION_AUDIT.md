# edge-perception-cpp optimization architecture

This document describes the native runtime's execution paths, memory ownership,
and performance-oriented design.

The generic path supports ONNX Runtime and synchronous native TensorRT. The
specialized TensorRT pipeline combines CUDA preprocessing and postprocessing
with persistent buffers and asynchronous execution.

These optimizations reduce work surrounding TensorRT execution, including
intermediate allocations, image copies, and host-side processing. Their effect
depends on the model, hardware, input path, and pipeline configuration and must
be measured with a consistent benchmark.

The [measurement report](results/README.md) separates performance from
[quality and parity limitations](RESULTS_ANALYSIS.md). Native/fused FP32, FP16
and INT8 have full-test quality and same-depth timing evidence. Fusion does not
win every comparison. The architecture below describes implemented mechanisms,
not proof that every input path is validated or every optimization is beneficial.

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

## Fused TensorRT path

`NativeTensorRtPipeline` is independent from the generic `InferenceEngine`.
This separates TensorRT-specific scheduling and memory management from the
portable multi-backend interface.

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

- half-pixel resize; packed RGB/BGR input reproduces OpenCV fixed-point
  interpolation and 8-bit rounding, including the exact 2× downsampling case;
- exact project letterbox geometry and padding;
- BGR, RGB, or NV12 input sampling;
- BT.601 or BT.709 NV12 conversion;
- full-range or limited-range NV12 handling;
- RGB/BGR output ordering;
- pixel scaling, mean subtraction, and standard-deviation normalization in
  the CPU reference's operation order;
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

### Native streaming workers

The generic native TensorRT streaming benchmark creates persistent per-slot
workers with independent engine/context, pinned/device storage, preprocessing
workspace and nonblocking streams. Slot zero borrows the supplied engine and
processor; no instance is called concurrently. Engines duplicate weight state,
so deeper pipelines increase memory use. Reported per-slot memory is not total
process memory. Cancellation drains and joins workers before releasing resources.

Depth-3 measurements observed three active workers for every precision and
repetition. This verifies asynchronous host execution, not guaranteed overlap
of TensorRT kernels on the GPU.

### Fused execution slots

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

## Generic-path optimizations

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
- separate portable CPU and NVIDIA-accelerated build profiles;
- integrated command-line, benchmark, and test targets.

## Deliberate boundaries

The inference core operates on decoded image data and a fixed model contract.
Acquisition, engine tuning, and system-level scheduling remain separate
integration concerns.

Not bundled:

- NVDEC, nvJPEG, DeepStream, GStreamer, or camera-vendor acquisition code;
- rebuilding the TensorRT engine with new tactics, plugins, or architecture-
  specific global build-route searches;
- operating-system process priority, CPU affinity, GPU clock locking, or power
  controls.

The device BGR/RGB/NV12 API allows an external decoder or capture layer to feed
GPU-resident frames directly into the inference pipeline.

## Validation expectations

Release validation must include the portable CPU build and CTest suite, the
real MSVC/NVCC/TensorRT build on the target NVIDIA machine, Python parity
tests, and representative runtime smoke tests. Performance claims require
measurements on the named hardware and software stack; they must not be
inferred from compilation alone.
