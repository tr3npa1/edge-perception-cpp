#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

#include "image_processor.hpp"
#include "inference_engine.hpp"
#include "postprocess.hpp"

namespace edge {

/**
 * @brief Pixel layout accepted by the device-resident fast-path entrypoint.
 */
enum class DeviceImageFormat : std::uint8_t {
    Bgr8,
    Rgb8,
    Nv12
};

/**
 * @brief YUV-to-RGB conversion matrix used for device NV12 frames.
 */
enum class YuvColorMatrix : std::uint8_t {
    Bt601,
    Bt709
};

/**
 * @brief Signal range used by a device NV12 frame.
 */
enum class YuvRange : std::uint8_t {
    Limited,
    Full
};

/**
 * @brief Non-owning view over externally managed page-locked BGR host memory.
 *
 * This overload avoids the CPU copy into the pipeline's internal pinned staging
 * buffer. The caller must keep the memory unchanged and alive until the ticket
 * returned by submit_pinned_bgr() has been collected.
 */
struct PinnedBgrImageView {
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    std::size_t row_stride_bytes = 0U;

    [[nodiscard]] bool valid() const noexcept;
};

/**
 * @brief Non-owning view over an image that already resides on the CUDA device.
 *
 * Bgr8 and Rgb8 use plane0 with one interleaved three-byte pixel per element.
 * Nv12 uses plane0 for Y and plane1 for interleaved UV. Device-resident input
 * removes host-to-device image transfer entirely and is the intended bridge to
 * NVDEC, camera-DMA, GStreamer CUDA, or another zero-copy capture pipeline.
 *
 * The caller retains ownership and must keep all referenced device memory alive
 * until the returned ticket has been collected.
 */
struct DeviceImageView {
    const void* plane0 = nullptr;
    const void* plane1 = nullptr;

    int width = 0;
    int height = 0;

    std::size_t plane0_pitch_bytes = 0U;
    std::size_t plane1_pitch_bytes = 0U;

    int device_id = 0;
    DeviceImageFormat format = DeviceImageFormat::Bgr8;

    YuvColorMatrix yuv_matrix = YuvColorMatrix::Bt709;
    YuvRange yuv_range = YuvRange::Limited;

    [[nodiscard]] bool valid() const noexcept;
};

/**
 * @brief Maximum-throughput native TensorRT pipeline configuration.
 *
 * This path is deliberately separate from the generic InferenceEngine. It is
 * specialized for the fastest complete GPU pipeline and owns multiple native
 * TensorRT execution contexts, CUDA streams, pinned staging buffers, persistent
 * device buffers, fused preprocessing kernels, fused postprocessing kernels,
 * and one CUDA graph per in-flight slot.
 */
struct NativeTensorRtPipelineConfig {
    std::filesystem::path engine_path{};

    std::string expected_input_name{"images"};
    std::string expected_output_name{"output0"};

    int device_id = 0;

    // Two slots overlap CPU frame preparation and GPU work without excessive
    // duplicated TensorRT context memory. Three can help when decode/capture is
    // irregular. Values above four are rarely beneficial for batch-one latency.
    std::size_t pipeline_depth = 2U;

    // Internal pinned/device BGR staging is allocated once per slot to this
    // upper bound. BDD100K frames are 1280x720, so the default leaves room for
    // common 1080p camera input without steady-state reallocations.
    ImageSize maximum_host_image_size{1920, 1080};

    PreprocessConfig preprocess{};
    PostprocessConfig postprocess{};

    // Capture H2D -> fused preprocess -> enqueueV3 -> fused postprocess -> D2H
    // as one CUDA graph after each slot completes one ordinary warm-up run.
    bool enable_cuda_graph = true;
    bool require_cuda_graph = false;

    // Keep postprocessing on the GPU and transfer compact finalized detections.
    // Disable only for parity/debugging, where raw [1,300,6] output is copied
    // back and processed by the ordinary CPU Postprocessor.
    bool enable_gpu_postprocess = true;

    // Request the highest available non-real-time CUDA stream priority.
    bool use_high_priority_streams = true;

    // Supply TensorRT with explicit non-blocking auxiliary streams when the
    // serialized engine was built to use within-inference multi-streaming.
    // This preserves CUDA-graph compatibility and allows all streams to use
    // the same priority policy as the main stream.
    bool use_engine_auxiliary_streams = true;

    // Disable per-layer enqueue profiling and lower runtime NVTX verbosity.
    // Profiling tools can still be used in a separately configured diagnostic
    // build, while the latency path avoids instrumentation overhead.
    bool minimize_runtime_instrumentation = true;

    // Maximum persistent L2 activation-cache allowance per execution context.
    // Zero leaves this architecture-dependent TensorRT feature disabled.
    std::size_t persistent_cache_limit_bytes = 0U;

    // Device-resident external source pointers often rotate between decoder
    // surfaces. Graph capture is therefore off for those submissions by
    // default; enable only when the pointer, shape, and pitches stay stable.
    bool capture_external_sources = false;

    /**
     * @brief Validate every static pipeline invariant.
     *
     * The fast path currently requires linear resize because its fused CUDA
     * kernel implements the same half-pixel bilinear mapping used by the CPU
     * parity path. MalformedCandidatePolicy::Throw is rejected because device
     * kernels cannot throw C++ exceptions.
     */
    void validate() const;
};

/**
 * @brief Opaque identifier for one asynchronous in-flight frame.
 */
struct GpuPipelineTicket {
    std::uint32_t slot_index = 0U;
    std::uint64_t generation = 0U;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return generation != 0U;
    }
};

/**
 * @brief Stable host-owned result returned after an asynchronous collection.
 */
struct GpuPipelineResult {
    DetectionBuffer detections{};
    PostprocessSummary summary{};
    LetterboxTransform transform{};

    std::uint64_t sequence_number = 0U;

    bool used_cuda_graph = false;
    bool source_was_device_resident = false;
};

/**
 * @brief Persistent-memory information for the optimized pipeline.
 */
struct GpuPipelineMemoryInfo {
    std::size_t pipeline_depth = 0U;
    std::size_t auxiliary_streams_per_slot = 0U;

    std::size_t pinned_host_bytes_per_slot = 0U;
    std::size_t device_bytes_per_slot = 0U;

    std::size_t total_pinned_host_bytes = 0U;
    std::size_t total_device_bytes = 0U;
};

/**
 * @brief Runtime metadata for the specialized native TensorRT fast path.
 */
struct GpuPipelineRuntimeInfo {
    std::string runtime_name{};
    std::string runtime_version{};

    TensorElementType input_element_type = TensorElementType::Float32;
    TensorElementType output_element_type = TensorElementType::Float32;

    std::array<std::int64_t, 4> input_shape{};
    std::array<std::int64_t, 3> output_shape{};

    GpuPipelineMemoryInfo memory{};

    bool gpu_postprocess_enabled = false;
    bool cuda_graph_enabled = false;
};

/**
 * @brief Fused, asynchronous, multi-slot native TensorRT deployment pipeline.
 *
 * Host BGR path:
 *
 *   cv::Mat -> persistent pinned staging -> H2D packed BGR
 *   -> fused resize/letterbox/BGR-to-RGB/normalization/NCHW kernel
 *   -> TensorRT enqueueV3
 *   -> fused confidence/class/box restoration kernel
 *   -> compact D2H result
 *
 * Device path:
 *
 *   BGR/RGB/NV12 device surface -> fused preprocessing directly into the
 *   TensorRT device input -> inference -> fused postprocessing.
 *
 * Each pipeline slot owns a separate execution context, stream, graph, event,
 * context workspace, input/output buffers, and result buffer. This permits the
 * caller to decode or prepare frame N+1 while frame N is still executing.
 *
 * The object is not internally synchronized for multiple producer threads.
 * Use one pipeline per producer or protect submit/collect externally.
 */
class NativeTensorRtPipeline final {
public:
    explicit NativeTensorRtPipeline(
        NativeTensorRtPipelineConfig config
    );

    ~NativeTensorRtPipeline();

    NativeTensorRtPipeline(const NativeTensorRtPipeline&) = delete;
    NativeTensorRtPipeline& operator=(
        const NativeTensorRtPipeline&
    ) = delete;

    NativeTensorRtPipeline(
        NativeTensorRtPipeline&& other
    ) noexcept;

    NativeTensorRtPipeline& operator=(
        NativeTensorRtPipeline&& other
    ) noexcept;

    [[nodiscard]] const NativeTensorRtPipelineConfig& config() const;
    [[nodiscard]] const GpuPipelineRuntimeInfo& runtime_info() const;

    /**
     * @brief Copy an ordinary OpenCV BGR frame into a free pinned slot and
     *        enqueue the complete GPU pipeline asynchronously.
     */
    [[nodiscard]] GpuPipelineTicket submit(
        const cv::Mat& bgr_image
    );

    /**
     * @brief Enqueue a page-locked BGR host frame without the staging copy.
     */
    [[nodiscard]] GpuPipelineTicket submit_pinned_bgr(
        PinnedBgrImageView image
    );

    /**
     * @brief Enqueue BGR, RGB, or NV12 source memory already on the GPU.
     */
    [[nodiscard]] GpuPipelineTicket submit_device(
        DeviceImageView image
    );

    /**
     * @brief Return true when one ticket's asynchronous D2H result is ready.
     *
     * @throws std::invalid_argument if the ticket does not belong to the
     *         current live generation of its slot.
     */
    [[nodiscard]] bool ready(
        GpuPipelineTicket ticket
    ) const;

    /**
     * @brief Wait for one ticket and return an owning host-side result.
     *
     * Collecting releases the slot for a future submission. Tickets may be
     * collected in any order, although FIFO collection generally gives the
     * clearest real-time behavior.
     */
    [[nodiscard]] GpuPipelineResult collect(
        GpuPipelineTicket ticket
    );

    /**
     * @brief Convenience synchronous wrapper around submit() and collect().
     */
    [[nodiscard]] GpuPipelineResult run(
        const cv::Mat& bgr_image
    );

    /**
     * @brief Convenience synchronous wrapper for device-resident input.
     */
    [[nodiscard]] GpuPipelineResult run_device(
        DeviceImageView image
    );

    [[nodiscard]] std::size_t pipeline_depth() const noexcept;
    [[nodiscard]] std::size_t in_flight_count() const noexcept;

    /**
     * @brief Return whether the specialized CUDA/TensorRT pipeline was built.
     */
    [[nodiscard]] static constexpr bool is_compiled() noexcept {
#if defined(EDGE_WITH_CUDA_PIPELINE) && EDGE_WITH_CUDA_PIPELINE
        return true;
#else
        return false;
#endif
    }

private:
    class Impl;

    [[nodiscard]] Impl& implementation();
    [[nodiscard]] const Impl& implementation() const;

    std::unique_ptr<Impl> implementation_;
};

}  // namespace edge
