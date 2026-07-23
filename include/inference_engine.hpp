#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "image_processor.hpp"
#include "postprocess.hpp"

namespace edge {

/**
 * @brief Fixed input shape used by every exported project artifact.
 */
inline constexpr std::array<std::int64_t, 4> kModelInputShape{
    1,
    3,
    640,
    640
};

/**
 * @brief Fixed end-to-end detection output shape used by every artifact.
 */
inline constexpr std::array<std::int64_t, 3> kModelOutputShape{
    1,
    static_cast<std::int64_t>(kModelMaximumDetections),
    static_cast<std::int64_t>(kDetectionValueCount)
};

/**
 * @brief Number of scalar elements in the fixed model input tensor.
 */
inline constexpr std::size_t kModelInputElementCount =
    1U * 3U * 640U * 640U;

/**
 * @brief Number of float elements in the fixed model output tensor.
 */
inline constexpr std::size_t kModelOutputElementCount =
    1U * kModelMaximumDetections * kDetectionValueCount;

/**
 * @brief Runtime implementation used to execute one model artifact.
 *
 * The three ONNX Runtime modes consume .onnx files. NativeTensorRt consumes a
 * serialized .engine file and is available only when the project is built with
 * EDGE_ENABLE_TENSORRT=ON.
 */
enum class InferenceBackend : std::uint8_t {
    OrtCpu,
    OrtCuda,
    OrtTensorRt,
    NativeTensorRt
};

/**
 * @brief Precision used to build or quantize the selected artifact.
 *
 * This describes the artifact's intended execution precision, not necessarily
 * its external input binding type. For example, the project's QDQ INT8 ONNX
 * model still exposes a Float32 input, and a TensorRT INT8 engine may also
 * expose a floating input binding.
 */
enum class ArtifactPrecision : std::uint8_t {
    Float32,
    Float16,
    Int8
};

/**
 * @brief ONNX Runtime graph-optimization policy.
 */
enum class OrtGraphOptimization : std::uint8_t {
    Disabled,
    Basic,
    Extended,
    All
};

/**
 * @brief CUDA allocator growth policy used by ONNX Runtime.
 */
enum class OrtCudaArenaExtendStrategy : std::uint8_t {
    NextPowerOfTwo,
    SameAsRequested
};

/**
 * @brief cuDNN convolution-algorithm selection policy used by ORT CUDA EP.
 */
enum class OrtCudnnConvAlgorithmSearch : std::uint8_t {
    Exhaustive,
    Heuristic,
    Default
};

/**
 * @brief Return a stable CLI/reporting name for an inference backend.
 */
[[nodiscard]] constexpr std::string_view inference_backend_name(
    InferenceBackend backend
) noexcept {
    switch (backend) {
        case InferenceBackend::OrtCpu:
            return "ort_cpu";

        case InferenceBackend::OrtCuda:
            return "ort_cuda";

        case InferenceBackend::OrtTensorRt:
            return "ort_tensorrt";

        case InferenceBackend::NativeTensorRt:
            return "native_tensorrt";
    }

    return "unknown";
}

/**
 * @brief Return a stable CLI/reporting name for an artifact precision.
 */
[[nodiscard]] constexpr std::string_view artifact_precision_name(
    ArtifactPrecision precision
) noexcept {
    switch (precision) {
        case ArtifactPrecision::Float32:
            return "fp32";

        case ArtifactPrecision::Float16:
            return "fp16";

        case ArtifactPrecision::Int8:
            return "int8";
    }

    return "unknown";
}

/**
 * @brief Return true when a backend executes model kernels on an NVIDIA GPU.
 */
[[nodiscard]] constexpr bool inference_backend_uses_gpu(
    InferenceBackend backend
) noexcept {
    switch (backend) {
        case InferenceBackend::OrtCpu:
            return false;

        case InferenceBackend::OrtCuda:
        case InferenceBackend::OrtTensorRt:
        case InferenceBackend::NativeTensorRt:
            return true;
    }

    return false;
}

/**
 * @brief ONNX Runtime session-level configuration.
 *
 * A value of zero for either thread count means "use the runtime default."
 * These values are relevant mainly to the CPU backend and to CPU fallback
 * nodes used by GPU execution providers.
 */
struct OrtSessionConfig {
    OrtGraphOptimization graph_optimization =
        OrtGraphOptimization::All;

    int intra_op_thread_count = 0;
    int inter_op_thread_count = 0;

    bool enable_cpu_memory_arena = true;
    bool enable_memory_pattern = true;

    bool enable_profiling = false;
    std::filesystem::path profiling_output{};

    /**
     * @brief Validate thread counts, enum values, and profiling configuration.
     *
     * @throws std::invalid_argument if a field is invalid.
     */
    void validate() const;
};

/**
 * @brief Shared configuration for CUDA-backed execution.
 *
 * gpu_memory_limit_bytes=0 delegates the memory limit to the runtime.
 *
 * CUDA graph capture is disabled by default so the initial implementation and
 * parity tests use the least surprising execution path. It can be enabled for
 * fixed-address, fixed-shape benchmark experiments after ordinary execution is
 * validated.
 */
struct GpuExecutionConfig {
    int device_id = 0;

    std::size_t gpu_memory_limit_bytes = 0U;

    OrtCudaArenaExtendStrategy arena_extend_strategy =
        OrtCudaArenaExtendStrategy::NextPowerOfTwo;

    OrtCudnnConvAlgorithmSearch cudnn_conv_algorithm_search =
        OrtCudnnConvAlgorithmSearch::Exhaustive;

    bool cudnn_conv_use_max_workspace = true;
    bool prefer_nhwc = false;
    bool use_tf32 = true;
    bool use_ep_level_unified_stream = true;

    // Reserved for a future explicitly asynchronous API. InferenceEngine::run()
    // has a synchronous host-output contract, so this must remain false.
    bool disable_provider_synchronization = false;

    // Applies only to the native TensorRT backend. The graph captures H2D,
    // enqueueV3, and D2H on one non-default stream after a warm-up run.
    bool enable_cuda_graph = false;

    /**
     * @brief Validate device, allocator, convolution, and synchronization
     *        configuration.
     *
     * @throws std::invalid_argument if a field is invalid.
     */
    void validate() const;
};

/**
 * @brief ONNX Runtime TensorRT Execution Provider configuration.
 *
 * TensorRT EP is registered ahead of CUDA EP. CUDA fallback remains enabled by
 * default so unsupported TensorRT subgraphs can still execute on the GPU
 * instead of unexpectedly falling all the way back to CPU.
 *
 * FP16 or INT8 provider flags are derived from ArtifactPrecision rather than
 * duplicated here.
 */
struct OrtTensorRtConfig {
    bool enable_cuda_fallback = true;

    std::size_t maximum_workspace_bytes = 0U;
    std::size_t minimum_subgraph_size = 1U;
    std::size_t maximum_partition_iterations = 1000U;

    bool enable_engine_cache = true;
    std::filesystem::path engine_cache_directory{
        "models/ort_trt_cache"
    };

    bool enable_timing_cache = true;
    std::filesystem::path timing_cache_directory{
        "models/ort_trt_cache"
    };

    bool enable_context_memory_sharing = true;
    bool enable_cuda_graph = true;

    int builder_optimization_level = 5;
    int auxiliary_streams = -1;

    bool dump_partitioned_subgraphs = false;

    /**
     * @brief Validate workspace, partitioning, and cache configuration.
     *
     * @throws std::invalid_argument if a field is invalid.
     */
    void validate() const;
};

/**
 * @brief Complete construction configuration for one inference engine.
 *
 * artifact_path must point to:
 *
 *     .onnx   for OrtCpu, OrtCuda, or OrtTensorRt
 *     .engine for NativeTensorRt
 *
 * expected_input_name and expected_output_name are validated together with
 * the fixed tensor shapes and element types when the artifact is loaded.
 */
struct InferenceEngineConfig {
    std::filesystem::path artifact_path{};

    InferenceBackend backend = InferenceBackend::OrtCpu;
    ArtifactPrecision artifact_precision =
        ArtifactPrecision::Float32;

    std::string expected_input_name{"images"};
    std::string expected_output_name{"output0"};

    OrtSessionConfig ort{};
    GpuExecutionConfig gpu{};
    OrtTensorRtConfig ort_tensorrt{};

    /**
     * @brief Validate configuration before a runtime or model is constructed.
     *
     * Model existence, tensor metadata, and backend availability are checked
     * by InferenceEngine construction.
     *
     * @throws std::invalid_argument if a field or backend/artifact combination
     *         is invalid.
     */
    void validate() const;
};

/**
 * @brief Tensor metadata discovered from the loaded artifact.
 *
 * Construction requires this information to match the project's fixed
 * shape contracts:
 *
 *     input  [1, 3, 640, 640]
 *     output [1, 300, 6]
 *
 * The external input binding must be Float32 or Float16. ONNX artifacts expose
 * Float32 output. A native TensorRT engine may expose Float32 or Float16 output;
 * InferenceEngine always normalizes the public CPU-readable output to Float32
 * before returning DetectionTensorView.
 */
struct InferenceModelInfo {
    std::string input_name{};
    std::string output_name{};

    std::array<std::int64_t, 4> input_shape{};
    std::array<std::int64_t, 3> output_shape{};

    TensorElementType input_element_type =
        TensorElementType::Float32;

    TensorElementType backend_output_element_type =
        TensorElementType::Float32;

    TensorElementType host_output_element_type =
        TensorElementType::Float32;

    std::size_t input_element_count = 0U;
    std::size_t input_byte_count = 0U;

    std::size_t output_element_count = 0U;
    std::size_t backend_output_byte_count = 0U;
    std::size_t host_output_byte_count = 0U;
};

/**
 * @brief Sizes and locations of persistent buffers owned by the engine.
 *
 * These fields cover buffers controlled directly by this project. They do not
 * attempt to estimate opaque workspace, allocator arenas, weights, tactics, or
 * other memory owned internally by ONNX Runtime, CUDA, or TensorRT.
 */
struct InferenceBufferInfo {
    std::size_t host_input_bytes = 0U;
    std::size_t host_output_bytes = 0U;

    std::size_t device_input_bytes = 0U;
    std::size_t device_output_bytes = 0U;

    // Explicit native TensorRT execution-context workspace. ORT owns and
    // manages its provider workspace internally, so this remains zero there.
    std::size_t device_workspace_bytes = 0U;

    bool host_input_is_pinned = false;
    bool host_output_is_pinned = false;
};

/**
 * @brief Runtime and provider information captured after initialization.
 *
 * active_execution_providers uses priority order for ONNX Runtime sessions.
 * Native TensorRT reports an empty provider list because it does not execute
 * through ONNX Runtime's provider system.
 */
struct InferenceRuntimeInfo {
    InferenceBackend backend = InferenceBackend::OrtCpu;
    ArtifactPrecision artifact_precision =
        ArtifactPrecision::Float32;

    std::string runtime_name{};
    std::string runtime_version{};

    std::vector<std::string> active_execution_providers{};

    InferenceModelInfo model{};
    InferenceBufferInfo buffers{};
};

/**
 * @brief Own and execute one fixed-shape YOLO26M inference runtime.
 *
 * The engine deliberately uses a private implementation so this public header
 * exposes no ONNX Runtime, CUDA, or TensorRT headers. Backend APIs, provider
 * options, streams, sessions, execution contexts, and native handles remain
 * isolated inside src/inference_engine.cpp.
 *
 * Input ownership:
 *
 *     1. input_tensor() exposes persistent CPU-writable storage.
 *     2. ImageProcessor writes NCHW FP32 or FP16 values directly into it.
 *     3. run() performs the backend-specific transfer/binding and execution.
 *
 * This removes an additional host-side tensor copy. The generic engine keeps
 * preprocessing on the CPU for backend parity. The separate
 * NativeTensorRtPipeline in gpu_pipeline.hpp provides fused CUDA preprocessing,
 * direct device input, asynchronous multi-frame execution, and GPU
 * postprocessing for the maximum-performance native TensorRT path.
 *
 * Output ownership:
 *
 *     run() completes synchronously and returns a CPU-readable FP32
 *     DetectionTensorView over persistent engine-owned storage. If a native
 *     TensorRT engine exposes FP16 output, the implementation converts the
 *     small fixed output tensor to FP32 before publishing the view.
 *
 * The returned output view remains valid until the next run(), move operation,
 * or engine destruction. It must not outlive the engine.
 *
 * Threading:
 *
 *     One InferenceEngine instance is not safe for concurrent run() calls
 *     because it owns mutable input/output buffers and backend execution state.
 *     Use one engine instance per inference worker.
 */
class InferenceEngine final {
public:
    explicit InferenceEngine(
        InferenceEngineConfig config
    );

    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    InferenceEngine(InferenceEngine&& other) noexcept;
    InferenceEngine& operator=(
        InferenceEngine&& other
    ) noexcept;

    /**
     * @brief Return the validated construction configuration.
     *
     * @throws std::logic_error if called on a moved-from engine.
     */
    [[nodiscard]] const InferenceEngineConfig& config() const;

    /**
     * @brief Return model, runtime, provider, and buffer metadata.
     *
     * @throws std::logic_error if called on a moved-from engine.
     */
    [[nodiscard]] const InferenceRuntimeInfo& runtime_info() const;

    /**
     * @brief Return the external input element type discovered from the model.
     *
     * This can differ from artifact_precision. In particular, the INT8 QDQ
     * model exposes Float32 input.
     *
     * @throws std::logic_error if called on a moved-from engine.
     */
    [[nodiscard]] TensorElementType input_element_type() const;

    /**
     * @brief Return the persistent CPU-writable model input buffer.
     *
     * ImageProcessor should preprocess directly into this view. The element
     * type and capacity exactly match the loaded model's external input.
     *
     * The view remains valid until the engine is moved or destroyed.
     *
     * @throws std::logic_error if called on a moved-from engine.
     */
    [[nodiscard]] MutableTensorView input_tensor();

    /**
     * @brief Execute one synchronous inference using the current input buffer.
     *
     * The call does not perform image preprocessing or detection
     * postprocessing. On return, all backend work and any required device-to-
     * host output transfer are complete, so the returned FP32 view can be
     * passed immediately to Postprocessor::process().
     *
     * Persistent model buffers are reused. Backend runtimes may still perform
     * one-time allocations or graph capture during initial warm-up runs.
     *
     * @throws std::logic_error if called on a moved-from engine.
     * @throws std::runtime_error if backend execution or synchronization fails.
     */
    [[nodiscard]] DetectionTensorView run();

    /**
     * @brief Return the output produced by the most recent successful run().
     *
     * @throws std::logic_error if no successful inference has completed or the
     *         engine has been moved from.
     */
    [[nodiscard]] DetectionTensorView output_tensor() const;

    /**
     * @brief Return true after at least one successful run().
     *
     * A moved-from engine returns false.
     */
    [[nodiscard]] bool has_output() const noexcept;

    /**
     * @brief Return the number of successful runs completed by this engine.
     *
     * A moved-from engine returns zero.
     */
    [[nodiscard]] std::uint64_t completed_run_count() const noexcept;

    /**
     * @brief Return whether native TensorRT support was compiled into this
     *        project binary.
     *
     * Runtime DLL availability, GPU compatibility, and engine compatibility
     * are checked separately when a native TensorRT engine is constructed.
     */
    [[nodiscard]] static constexpr bool native_tensorrt_is_compiled()
        noexcept {
#if defined(EDGE_WITH_TENSORRT) && EDGE_WITH_TENSORRT
        return true;
#else
        return false;
#endif
    }

private:
    class Impl;

    /**
     * @brief Return the live implementation or throw for a moved-from object.
     */
    [[nodiscard]] Impl& implementation();

    /**
     * @brief Return the live implementation or throw for a moved-from object.
     */
    [[nodiscard]] const Impl& implementation() const;

    std::unique_ptr<Impl> implementation_;
};

}  // namespace edge
