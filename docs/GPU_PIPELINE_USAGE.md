# NativeTensorRtPipeline usage

`NativeTensorRtPipeline` is the specialized CUDA/TensorRT path. The
generic `InferenceEngine` remains useful for ONNX Runtime comparisons and
simple synchronous execution.

See the [measured results](results/README.md) and
[parity limitations](RESULTS_ANALYSIS.md) before selecting a deployment path.
The fused FP16 path passes same-artifact parity on 64/64 fixed images;
strict PyTorch parity is 61/64. See the
[correctness report](results/fp16-correctness.md) and complete
[same-depth matrix](results/fp16-matrix.md). Native/fused full-test mAP50–95
is 34.0120 for FP16 and 33.2609 for INT8. Both loaders accept raw and
Ultralytics-prefixed plans; see [INT8 validation](results/int8-status.md).
The benchmark uses OpenCV BGR input; the external-input APIs below were not
separately benchmarked in that suite.

## Synchronous OpenCV frame

```cpp
#include "gpu_pipeline.hpp"

#include <opencv2/imgcodecs.hpp>

edge::NativeTensorRtPipelineConfig config{};
config.engine_path =
    "models/engine/yolo26m_bdd100k_fp16.engine";
config.pipeline_depth = 2U;
config.enable_cuda_graph = true;
config.enable_gpu_postprocess = true;
config.use_high_priority_streams = true;
config.use_engine_auxiliary_streams = true;
config.minimize_runtime_instrumentation = true;

edge::NativeTensorRtPipeline pipeline{config};

const cv::Mat frame = cv::imread("frame.jpg", cv::IMREAD_COLOR);
const edge::GpuPipelineResult result = pipeline.run(frame);

for (const edge::Detection& detection : result.detections) {
    // detection.box is already restored to original-image coordinates.
}
```

## Two frames in flight

```cpp
const edge::GpuPipelineTicket first = pipeline.submit(frame_0);
const edge::GpuPipelineTicket second = pipeline.submit(frame_1);

const edge::GpuPipelineResult result_0 = pipeline.collect(first);
const edge::GpuPipelineResult result_1 = pipeline.collect(second);
```

`submit(cv::Mat)` copies pixels into slot-owned pinned memory before returning,
so the original `cv::Mat` can be reused immediately. Do not submit more frames
than `pipeline_depth()` without collecting an outstanding ticket.

## Externally allocated page-locked BGR frame

```cpp
edge::PinnedBgrImageView view{};
view.data = pinned_bgr_pointer;
view.width = width;
view.height = height;
view.row_stride_bytes = stride;

const edge::GpuPipelineTicket ticket =
    pipeline.submit_pinned_bgr(view);

// Keep the external page-locked storage alive and unchanged until collect().
const edge::GpuPipelineResult result = pipeline.collect(ticket);
```

This avoids the CPU copy into the pipeline's internal pinned staging buffer.

## Device-resident NV12 frame

```cpp
edge::DeviceImageView frame{};
frame.plane0 = device_y_plane;
frame.plane1 = device_interleaved_uv_plane;
frame.width = width;
frame.height = height;
frame.plane0_pitch_bytes = y_pitch;
frame.plane1_pitch_bytes = uv_pitch;
frame.device_id = 0;
frame.format = edge::DeviceImageFormat::Nv12;
frame.yuv_matrix = edge::YuvColorMatrix::Bt709;
frame.yuv_range = edge::YuvRange::Limited;

const edge::GpuPipelineTicket ticket = pipeline.submit_device(frame);
const edge::GpuPipelineResult result = pipeline.collect(ticket);
```

The device planes must stay alive and unchanged until collection. NV12 width and
height must be even. This path performs no host image transfer.

## Device-resident BGR or RGB frame

Use `DeviceImageFormat::Bgr8` or `DeviceImageFormat::Rgb8`, put the interleaved
three-channel pointer in `plane0`, leave `plane1=nullptr`, and provide the byte
pitch in `plane0_pitch_bytes`.

## Parity/debug mode

```cpp
config.enable_gpu_postprocess = false;
config.enable_cuda_graph = false;
```

This copies raw `[1,300,6]` output to host FP32 and runs the ordinary CPU
`Postprocessor`. It is slower but useful for validating GPU postprocessing.

## CUDA graph policy

For internally staged OpenCV input, graph capture is enabled by default.
External pointers are not captured by default because decoder surfaces often
rotate. Set:

```cpp
config.capture_external_sources = true;
```

only when source pointers, dimensions, and pitches remain stable. The pipeline
tracks those values and recaptures when the graph key changes.
