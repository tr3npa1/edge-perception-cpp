#include "gpu_pipeline.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace edge {
namespace {

constexpr std::size_t kPackedPixelBytes = 3U;
constexpr unsigned int kPostprocessThreads = 512U;
constexpr unsigned int kPreprocessBlockX = 32U;
constexpr unsigned int kPreprocessBlockY = 8U;

[[nodiscard]] std::string path_to_utf8(
    const std::filesystem::path& path
) {
#if defined(__cpp_lib_char8_t)
    const std::u8string utf8 = path.u8string();
    return std::string{
        reinterpret_cast<const char*>(utf8.data()),
        utf8.size()
    };
#else
    return path.u8string();
#endif
}

void validate_engine_file(
    const std::filesystem::path& path
) {
    std::error_code error{};

    if (!std::filesystem::exists(path, error) || error) {
        throw std::runtime_error(
            "TensorRT engine does not exist: " + path_to_utf8(path) + "."
        );
    }

    if (!std::filesystem::is_regular_file(path, error) || error) {
        throw std::runtime_error(
            "TensorRT engine is not a regular file: "
            + path_to_utf8(path)
            + "."
        );
    }

    const std::uintmax_t size = std::filesystem::file_size(path, error);

    if (error || size == 0U) {
        throw std::runtime_error(
            "TensorRT engine is empty or unreadable: "
            + path_to_utf8(path)
            + "."
        );
    }
}

[[nodiscard]] std::vector<std::uint8_t> read_binary_file(
    const std::filesystem::path& path
) {
    std::ifstream input{path, std::ios::binary | std::ios::ate};

    if (!input.is_open()) {
        throw std::runtime_error(
            "Failed to open TensorRT engine: " + path_to_utf8(path) + "."
        );
    }

    const std::streampos end = input.tellg();

    if (end == std::streampos{-1}) {
        throw std::runtime_error(
            "Failed to determine TensorRT engine size."
        );
    }

    const std::streamoff signed_size = end - std::streampos{0};

    if (signed_size <= 0) {
        throw std::runtime_error("TensorRT engine is empty.");
    }

    const auto unsigned_size = static_cast<std::uintmax_t>(signed_size);

    if (
        unsigned_size
        > static_cast<std::uintmax_t>(
            std::numeric_limits<std::size_t>::max()
        )
        || unsigned_size
            > static_cast<std::uintmax_t>(
                std::numeric_limits<std::streamsize>::max()
            )
    ) {
        throw std::runtime_error("TensorRT engine is too large to read.");
    }

    std::vector<std::uint8_t> bytes(
        static_cast<std::size_t>(unsigned_size)
    );

    input.seekg(0, std::ios::beg);
    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size())
    );

    if (!input || input.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error(
            "Failed to read the complete TensorRT engine."
        );
    }

    return bytes;
}

void check_cuda(
    cudaError_t status,
    std::string_view operation
) {
    if (status == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string{operation}
        + " failed: "
        + cudaGetErrorString(status)
        + "."
    );
}

void check_last_kernel(
    std::string_view operation
) {
    check_cuda(cudaPeekAtLastError(), operation);
}

[[nodiscard]] std::size_t checked_image_bytes(
    ImageSize size
) {
    if (!size.valid()) {
        throw std::invalid_argument("Image dimensions must be positive.");
    }

    constexpr std::size_t maximum =
        std::numeric_limits<std::size_t>::max();

    const std::size_t width = static_cast<std::size_t>(size.width);
    const std::size_t height = static_cast<std::size_t>(size.height);

    if (width > maximum / kPackedPixelBytes) {
        throw std::overflow_error("Image row byte count overflows size_t.");
    }

    const std::size_t row_bytes = width * kPackedPixelBytes;

    if (height > maximum / row_bytes) {
        throw std::overflow_error("Image byte count overflows size_t.");
    }

    return row_bytes * height;
}

[[nodiscard]] std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    std::string_view description
) {
    if (
        left != 0U
        && right > std::numeric_limits<std::size_t>::max() / left
    ) {
        throw std::overflow_error(
            std::string{description} + " overflows size_t."
        );
    }

    return left * right;
}

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(
        Severity severity,
        const char* message
    ) noexcept override {
        if (message == nullptr || severity > Severity::kWARNING) {
            return;
        }

        const char* label = "warning";

        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                label = "internal-error";
                break;
            case Severity::kERROR:
                label = "error";
                break;
            case Severity::kWARNING:
                label = "warning";
                break;
            case Severity::kINFO:
                label = "info";
                break;
            case Severity::kVERBOSE:
                label = "verbose";
                break;
        }

        std::lock_guard<std::mutex> lock{mutex_};
        std::cerr << "[TensorRT " << label << "] " << message << '\n';
    }

private:
    std::mutex mutex_{};
};

[[nodiscard]] TensorRtLogger& tensorrt_logger() {
    static TensorRtLogger logger{};
    return logger;
}

void initialize_tensorrt_plugins() {
    static std::once_flag once{};
    static bool success = false;

    std::call_once(
        once,
        [] {
            success = initLibNvInferPlugins(&tensorrt_logger(), "");
        }
    );

    if (!success) {
        throw std::runtime_error("TensorRT plugin registration failed.");
    }
}

[[nodiscard]] std::string tensorrt_runtime_version() {
    const std::int32_t encoded = getInferLibVersion();

    if (encoded <= 0) {
        return "unknown";
    }

    const std::int32_t major = encoded / 10000;
    const std::int32_t minor = (encoded / 100) % 100;
    const std::int32_t patch = encoded % 100;

    return std::to_string(major)
        + "."
        + std::to_string(minor)
        + "."
        + std::to_string(patch);
}

[[nodiscard]] TensorElementType tensor_type_from_tensorrt(
    nvinfer1::DataType type,
    std::string_view role
) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return TensorElementType::Float32;
        case nvinfer1::DataType::kHALF:
            return TensorElementType::Float16;
        default:
            throw std::runtime_error(
                std::string{role}
                + " uses an unsupported TensorRT element type."
            );
    }
}

template <std::size_t Rank>
[[nodiscard]] std::array<std::int64_t, Rank> fixed_shape_from_tensorrt(
    const nvinfer1::Dims& dimensions,
    std::string_view role
) {
    if (dimensions.nbDims != static_cast<int>(Rank)) {
        throw std::runtime_error(
            std::string{role}
            + " rank mismatch. Expected "
            + std::to_string(Rank)
            + ", received "
            + std::to_string(dimensions.nbDims)
            + "."
        );
    }

    std::array<std::int64_t, Rank> shape{};

    for (std::size_t index = 0U; index < Rank; ++index) {
        if (dimensions.d[index] <= 0) {
            throw std::runtime_error(
                std::string{role}
                + " contains an unresolved or invalid dimension."
            );
        }

        shape[index] = static_cast<std::int64_t>(dimensions.d[index]);
    }

    return shape;
}

[[nodiscard]] bool contains_dynamic_dimension(
    const nvinfer1::Dims& dimensions
) noexcept {
    for (int index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] < 0) {
            return true;
        }
    }

    return false;
}

template <std::size_t Rank>
[[nodiscard]] nvinfer1::Dims shape_to_tensorrt(
    const std::array<std::int64_t, Rank>& shape
) {
    nvinfer1::Dims dimensions{};
    dimensions.nbDims = static_cast<int>(Rank);

    for (std::size_t index = 0U; index < Rank; ++index) {
        if (
            shape[index] <= 0
            || shape[index]
                > static_cast<std::int64_t>(
                    std::numeric_limits<std::int32_t>::max()
                )
        ) {
            throw std::runtime_error(
                "Tensor shape cannot be represented by TensorRT."
            );
        }

        dimensions.d[index] = static_cast<std::int32_t>(shape[index]);
    }

    return dimensions;
}

class PinnedBuffer final {
public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(std::size_t bytes) {
        allocate(bytes);
    }

    ~PinnedBuffer() {
        release();
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          bytes_(std::exchange(other.bytes_, 0U)) {}

    void allocate(std::size_t bytes) {
        if (bytes == 0U) {
            throw std::invalid_argument("Pinned allocation cannot be empty.");
        }

        release();
        check_cuda(
            cudaHostAlloc(&data_, bytes, cudaHostAllocPortable),
            "cudaHostAlloc"
        );
        bytes_ = bytes;
        std::memset(data_, 0, bytes_);
    }

    [[nodiscard]] void* data() noexcept {
        return data_;
    }

private:
    void release() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFreeHost(data_));
        }

        data_ = nullptr;
        bytes_ = 0U;
    }

    void* data_ = nullptr;
    std::size_t bytes_ = 0U;
};

class DeviceBuffer final {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t bytes) {
        allocate(bytes);
    }

    ~DeviceBuffer() {
        release();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          bytes_(std::exchange(other.bytes_, 0U)) {}

    void allocate(std::size_t bytes) {
        if (bytes == 0U) {
            throw std::invalid_argument("Device allocation cannot be empty.");
        }

        release();
        check_cuda(cudaMalloc(&data_, bytes), "cudaMalloc");
        bytes_ = bytes;
    }

    [[nodiscard]] void* data() noexcept {
        return data_;
    }

private:
    void release() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
        }

        data_ = nullptr;
        bytes_ = 0U;
    }

    void* data_ = nullptr;
    std::size_t bytes_ = 0U;
};

class CudaStream final {
public:
    CudaStream() = default;

    CudaStream(int device_id, bool high_priority) {
        create(device_id, high_priority);
    }

    ~CudaStream() {
        release();
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept
        : stream_(std::exchange(other.stream_, nullptr)) {}

    void create(int device_id, bool high_priority) {
        release();
        check_cuda(cudaSetDevice(device_id), "cudaSetDevice");

        int least_priority = 0;
        int greatest_priority = 0;

        check_cuda(
            cudaDeviceGetStreamPriorityRange(
                &least_priority,
                &greatest_priority
            ),
            "cudaDeviceGetStreamPriorityRange"
        );

        const int priority = high_priority
            ? greatest_priority
            : least_priority;

        check_cuda(
            cudaStreamCreateWithPriority(
                &stream_,
                cudaStreamNonBlocking,
                priority
            ),
            "cudaStreamCreateWithPriority"
        );
    }

    [[nodiscard]] cudaStream_t get() const noexcept {
        return stream_;
    }

private:
    void release() noexcept {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
        }

        stream_ = nullptr;
    }

    cudaStream_t stream_ = nullptr;
};

class CudaEvent final {
public:
    CudaEvent() = default;

    explicit CudaEvent(bool create_now) {
        if (create_now) {
            create();
        }
    }

    ~CudaEvent() {
        release();
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept
        : event_(std::exchange(other.event_, nullptr)) {}

    void create() {
        release();
        check_cuda(
            cudaEventCreateWithFlags(&event_, cudaEventDisableTiming),
            "cudaEventCreateWithFlags"
        );
    }

    void record(cudaStream_t stream) {
        check_cuda(cudaEventRecord(event_, stream), "cudaEventRecord");
    }

    [[nodiscard]] bool ready() const {
        const cudaError_t status = cudaEventQuery(event_);

        if (status == cudaSuccess) {
            return true;
        }

        if (status == cudaErrorNotReady) {
            return false;
        }

        check_cuda(status, "cudaEventQuery");
        return false;
    }

    void synchronize() const {
        check_cuda(cudaEventSynchronize(event_), "cudaEventSynchronize");
    }

private:
    void release() noexcept {
        if (event_ != nullptr) {
            static_cast<void>(cudaEventDestroy(event_));
        }

        event_ = nullptr;
    }

    cudaEvent_t event_ = nullptr;
};

class CudaGraph final {
public:
    CudaGraph() = default;

    ~CudaGraph() {
        reset();
    }

    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
    CudaGraph(CudaGraph&&) = delete;
    CudaGraph& operator=(CudaGraph&&) = delete;

    [[nodiscard]] bool ready() const noexcept {
        return executable_ != nullptr;
    }

    void begin(cudaStream_t stream) {
        reset();
        check_cuda(
            cudaStreamBeginCapture(
                stream,
                cudaStreamCaptureModeGlobal
            ),
            "cudaStreamBeginCapture"
        );
    }

    void end_and_instantiate(cudaStream_t stream) {
        check_cuda(
            cudaStreamEndCapture(stream, &graph_),
            "cudaStreamEndCapture"
        );

        check_cuda(
            cudaGraphInstantiate(
                &executable_,
                graph_,
                nullptr,
                nullptr,
                0U
            ),
            "cudaGraphInstantiate"
        );
    }

    void abort(cudaStream_t stream) noexcept {
        cudaGraph_t abandoned = nullptr;
        static_cast<void>(cudaStreamEndCapture(stream, &abandoned));

        if (abandoned != nullptr) {
            static_cast<void>(cudaGraphDestroy(abandoned));
        }

        static_cast<void>(cudaGetLastError());
        reset();
    }

    void launch(cudaStream_t stream) const {
        if (executable_ == nullptr) {
            throw std::logic_error("CUDA graph is not instantiated.");
        }

        check_cuda(
            cudaGraphLaunch(executable_, stream),
            "cudaGraphLaunch"
        );
    }

    void reset() noexcept {
        if (executable_ != nullptr) {
            static_cast<void>(cudaGraphExecDestroy(executable_));
        }

        if (graph_ != nullptr) {
            static_cast<void>(cudaGraphDestroy(graph_));
        }

        executable_ = nullptr;
        graph_ = nullptr;
    }

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
};

enum class SourceMode : std::uint8_t {
    InternalPinnedBgr,
    ExternalPinnedBgr,
    DeviceBgr,
    DeviceRgb,
    DeviceNv12
};

struct SourceDescriptor {
    SourceMode mode = SourceMode::InternalPinnedBgr;

    const void* plane0 = nullptr;
    const void* plane1 = nullptr;

    int width = 0;
    int height = 0;

    std::size_t pitch0 = 0U;
    std::size_t pitch1 = 0U;

    YuvColorMatrix yuv_matrix = YuvColorMatrix::Bt709;
    YuvRange yuv_range = YuvRange::Limited;
};

struct GraphKey {
    SourceMode mode = SourceMode::InternalPinnedBgr;
    int width = 0;
    int height = 0;
    std::size_t pitch0 = 0U;
    std::size_t pitch1 = 0U;
    const void* plane0 = nullptr;
    const void* plane1 = nullptr;

    [[nodiscard]] bool operator==(const GraphKey& other) const noexcept {
        return mode == other.mode
            && width == other.width
            && height == other.height
            && pitch0 == other.pitch0
            && pitch1 == other.pitch1
            && plane0 == other.plane0
            && plane1 == other.plane1;
    }
};

struct PreprocessKernelParams {
    int source_width = 0;
    int source_height = 0;

    int resized_width = 0;
    int resized_height = 0;

    int network_width = 0;
    int network_height = 0;

    int pad_left = 0;
    int pad_top = 0;

    std::size_t pitch0 = 0U;
    std::size_t pitch1 = 0U;

    int source_mode = 0;
    int output_is_rgb = 1;
    int yuv_matrix = 1;
    int yuv_range = 0;

    float pixel_scale = 1.0F / 255.0F;
    float mean0 = 0.0F;
    float mean1 = 0.0F;
    float mean2 = 0.0F;
    float inv_std0 = 1.0F;
    float inv_std1 = 1.0F;
    float inv_std2 = 1.0F;

    float pad_r = 114.0F;
    float pad_g = 114.0F;
    float pad_b = 114.0F;
};

struct DeviceDetectionPacked {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
    float confidence;
    std::int32_t class_id;
};

struct DevicePostprocessSummary {
    std::uint32_t candidates_examined;
    std::uint32_t rejected_by_confidence;
    std::uint32_t rejected_as_malformed;
    std::uint32_t rejected_as_degenerate;
    std::uint32_t detections_written;
    std::uint32_t maximum_reached;
};

struct DevicePostprocessResult {
    DevicePostprocessSummary summary;
    DeviceDetectionPacked detections[kModelMaximumDetections];
};

struct PostprocessKernelParams {
    float confidence_threshold = 0.25F;
    float class_id_tolerance = 1.0e-3F;

    std::uint32_t maximum_detections =
        static_cast<std::uint32_t>(kModelMaximumDetections);

    std::uint32_t class_count =
        static_cast<std::uint32_t>(kBdd100kClassCount);

    std::uint32_t clip_boxes = 1U;
    std::uint32_t discard_degenerate = 1U;
    std::uint32_t sorted_by_confidence = 1U;

    float inverse_scale_x = 1.0F;
    float inverse_scale_y = 1.0F;
    float offset_x = 0.0F;
    float offset_y = 0.0F;
    float maximum_x = 0.0F;
    float maximum_y = 0.0F;
};

__device__ __forceinline__ int clamp_integer(
    int value,
    int low,
    int high
) {
    return max(low, min(value, high));
}

__device__ __forceinline__ float clamp_float(
    float value,
    float low,
    float high
) {
    return fminf(high, fmaxf(low, value));
}

__device__ __forceinline__ float3 read_interleaved_rgb(
    const std::uint8_t* source,
    std::size_t pitch,
    int x,
    int y,
    bool source_is_rgb
) {
    const std::uint8_t* pixel = source
        + static_cast<std::size_t>(y) * pitch
        + static_cast<std::size_t>(x) * 3U;

    if (source_is_rgb) {
        return make_float3(
            static_cast<float>(pixel[0]),
            static_cast<float>(pixel[1]),
            static_cast<float>(pixel[2])
        );
    }

    return make_float3(
        static_cast<float>(pixel[2]),
        static_cast<float>(pixel[1]),
        static_cast<float>(pixel[0])
    );
}

__device__ __forceinline__ float3 sample_interleaved_bilinear(
    const std::uint8_t* source,
    const PreprocessKernelParams& params,
    float source_x,
    float source_y,
    bool source_is_rgb
) {
    const int x0_unclamped = static_cast<int>(floorf(source_x));
    const int y0_unclamped = static_cast<int>(floorf(source_y));

    const float weight_x = source_x - floorf(source_x);
    const float weight_y = source_y - floorf(source_y);

    const int x0 = clamp_integer(
        x0_unclamped,
        0,
        params.source_width - 1
    );

    const int y0 = clamp_integer(
        y0_unclamped,
        0,
        params.source_height - 1
    );

    const int x1 = clamp_integer(
        x0_unclamped + 1,
        0,
        params.source_width - 1
    );

    const int y1 = clamp_integer(
        y0_unclamped + 1,
        0,
        params.source_height - 1
    );

    const float3 top_left = read_interleaved_rgb(
        source,
        params.pitch0,
        x0,
        y0,
        source_is_rgb
    );

    const float3 top_right = read_interleaved_rgb(
        source,
        params.pitch0,
        x1,
        y0,
        source_is_rgb
    );

    const float3 bottom_left = read_interleaved_rgb(
        source,
        params.pitch0,
        x0,
        y1,
        source_is_rgb
    );

    const float3 bottom_right = read_interleaved_rgb(
        source,
        params.pitch0,
        x1,
        y1,
        source_is_rgb
    );

    const float inverse_x = 1.0F - weight_x;
    const float inverse_y = 1.0F - weight_y;

    const float top_r = top_left.x * inverse_x + top_right.x * weight_x;
    const float top_g = top_left.y * inverse_x + top_right.y * weight_x;
    const float top_b = top_left.z * inverse_x + top_right.z * weight_x;

    const float bottom_r =
        bottom_left.x * inverse_x + bottom_right.x * weight_x;

    const float bottom_g =
        bottom_left.y * inverse_x + bottom_right.y * weight_x;

    const float bottom_b =
        bottom_left.z * inverse_x + bottom_right.z * weight_x;

    return make_float3(
        top_r * inverse_y + bottom_r * weight_y,
        top_g * inverse_y + bottom_g * weight_y,
        top_b * inverse_y + bottom_b * weight_y
    );
}

__device__ __forceinline__ float sample_plane_bilinear(
    const std::uint8_t* plane,
    std::size_t pitch,
    int width,
    int height,
    float source_x,
    float source_y
) {
    const int x0_unclamped = static_cast<int>(floorf(source_x));
    const int y0_unclamped = static_cast<int>(floorf(source_y));

    const float weight_x = source_x - floorf(source_x);
    const float weight_y = source_y - floorf(source_y);

    const int x0 = clamp_integer(x0_unclamped, 0, width - 1);
    const int y0 = clamp_integer(y0_unclamped, 0, height - 1);
    const int x1 = clamp_integer(x0_unclamped + 1, 0, width - 1);
    const int y1 = clamp_integer(y0_unclamped + 1, 0, height - 1);

    const float top_left = static_cast<float>(
        plane[static_cast<std::size_t>(y0) * pitch
            + static_cast<std::size_t>(x0)]
    );

    const float top_right = static_cast<float>(
        plane[static_cast<std::size_t>(y0) * pitch
            + static_cast<std::size_t>(x1)]
    );

    const float bottom_left = static_cast<float>(
        plane[static_cast<std::size_t>(y1) * pitch
            + static_cast<std::size_t>(x0)]
    );

    const float bottom_right = static_cast<float>(
        plane[static_cast<std::size_t>(y1) * pitch
            + static_cast<std::size_t>(x1)]
    );

    const float top = top_left * (1.0F - weight_x)
        + top_right * weight_x;

    const float bottom = bottom_left * (1.0F - weight_x)
        + bottom_right * weight_x;

    return top * (1.0F - weight_y) + bottom * weight_y;
}

__device__ __forceinline__ float2 sample_nv12_uv_bilinear(
    const std::uint8_t* uv_plane,
    std::size_t pitch,
    int width,
    int height,
    float source_x,
    float source_y
) {
    const int uv_width = max(1, width / 2);
    const int uv_height = max(1, height / 2);

    const float uv_x = source_x * 0.5F;
    const float uv_y = source_y * 0.5F;

    const int x0_unclamped = static_cast<int>(floorf(uv_x));
    const int y0_unclamped = static_cast<int>(floorf(uv_y));

    const float weight_x = uv_x - floorf(uv_x);
    const float weight_y = uv_y - floorf(uv_y);

    const int x0 = clamp_integer(x0_unclamped, 0, uv_width - 1);
    const int y0 = clamp_integer(y0_unclamped, 0, uv_height - 1);
    const int x1 = clamp_integer(x0_unclamped + 1, 0, uv_width - 1);
    const int y1 = clamp_integer(y0_unclamped + 1, 0, uv_height - 1);

    const std::uint8_t* top_left_pointer = uv_plane
        + static_cast<std::size_t>(y0) * pitch
        + static_cast<std::size_t>(x0) * 2U;

    const std::uint8_t* top_right_pointer = uv_plane
        + static_cast<std::size_t>(y0) * pitch
        + static_cast<std::size_t>(x1) * 2U;

    const std::uint8_t* bottom_left_pointer = uv_plane
        + static_cast<std::size_t>(y1) * pitch
        + static_cast<std::size_t>(x0) * 2U;

    const std::uint8_t* bottom_right_pointer = uv_plane
        + static_cast<std::size_t>(y1) * pitch
        + static_cast<std::size_t>(x1) * 2U;

    const float2 top_left = make_float2(
        static_cast<float>(top_left_pointer[0]),
        static_cast<float>(top_left_pointer[1])
    );

    const float2 top_right = make_float2(
        static_cast<float>(top_right_pointer[0]),
        static_cast<float>(top_right_pointer[1])
    );

    const float2 bottom_left = make_float2(
        static_cast<float>(bottom_left_pointer[0]),
        static_cast<float>(bottom_left_pointer[1])
    );

    const float2 bottom_right = make_float2(
        static_cast<float>(bottom_right_pointer[0]),
        static_cast<float>(bottom_right_pointer[1])
    );

    const float2 top = make_float2(
        top_left.x * (1.0F - weight_x) + top_right.x * weight_x,
        top_left.y * (1.0F - weight_x) + top_right.y * weight_x
    );

    const float2 bottom = make_float2(
        bottom_left.x * (1.0F - weight_x)
            + bottom_right.x * weight_x,
        bottom_left.y * (1.0F - weight_x)
            + bottom_right.y * weight_x
    );

    return make_float2(
        top.x * (1.0F - weight_y) + bottom.x * weight_y,
        top.y * (1.0F - weight_y) + bottom.y * weight_y
    );
}

__device__ __forceinline__ float3 yuv_to_rgb(
    float y,
    float u,
    float v,
    int matrix,
    int range
) {
    float red = 0.0F;
    float green = 0.0F;
    float blue = 0.0F;

    u -= 128.0F;
    v -= 128.0F;

    if (range == static_cast<int>(YuvRange::Limited)) {
        const float scaled_y = 1.16438356F * (y - 16.0F);

        if (matrix == static_cast<int>(YuvColorMatrix::Bt709)) {
            red = scaled_y + 1.79274107F * v;
            green = scaled_y - 0.21324861F * u - 0.53290933F * v;
            blue = scaled_y + 2.11240179F * u;
        } else {
            red = scaled_y + 1.59602678F * v;
            green = scaled_y - 0.39176229F * u - 0.81296764F * v;
            blue = scaled_y + 2.01723214F * u;
        }
    } else if (matrix == static_cast<int>(YuvColorMatrix::Bt709)) {
        red = y + 1.5748F * v;
        green = y - 0.187324F * u - 0.468124F * v;
        blue = y + 1.8556F * u;
    } else {
        red = y + 1.402F * v;
        green = y - 0.344136F * u - 0.714136F * v;
        blue = y + 1.772F * u;
    }

    return make_float3(
        clamp_float(red, 0.0F, 255.0F),
        clamp_float(green, 0.0F, 255.0F),
        clamp_float(blue, 0.0F, 255.0F)
    );
}

__device__ __forceinline__ float3 sample_source_rgb(
    const std::uint8_t* plane0,
    const std::uint8_t* plane1,
    const PreprocessKernelParams& params,
    float source_x,
    float source_y
) {
    if (params.source_mode == static_cast<int>(SourceMode::DeviceNv12)) {
        const float y = sample_plane_bilinear(
            plane0,
            params.pitch0,
            params.source_width,
            params.source_height,
            source_x,
            source_y
        );

        const float2 uv = sample_nv12_uv_bilinear(
            plane1,
            params.pitch1,
            params.source_width,
            params.source_height,
            source_x,
            source_y
        );

        return yuv_to_rgb(
            y,
            uv.x,
            uv.y,
            params.yuv_matrix,
            params.yuv_range
        );
    }

    const bool source_is_rgb =
        params.source_mode == static_cast<int>(SourceMode::DeviceRgb);

    return sample_interleaved_bilinear(
        plane0,
        params,
        source_x,
        source_y,
        source_is_rgb
    );
}

__device__ __forceinline__ float normalized_channel(
    float pixel,
    float scale,
    float mean,
    float inverse_std
) {
    return (pixel * scale - mean) * inverse_std;
}

struct NormalizedPixel {
    float channel0;
    float channel1;
    float channel2;
};

__device__ __forceinline__ NormalizedPixel preprocess_one_pixel(
    const std::uint8_t* plane0,
    const std::uint8_t* plane1,
    const PreprocessKernelParams& params,
    int network_x,
    int network_y
) {
    const int resized_x = network_x - params.pad_left;
    const int resized_y = network_y - params.pad_top;

    float3 rgb = make_float3(
        params.pad_r,
        params.pad_g,
        params.pad_b
    );

    if (
        resized_x >= 0
        && resized_x < params.resized_width
        && resized_y >= 0
        && resized_y < params.resized_height
    ) {
        const float source_x =
            (static_cast<float>(resized_x) + 0.5F)
            * static_cast<float>(params.source_width)
            / static_cast<float>(params.resized_width)
            - 0.5F;

        const float source_y =
            (static_cast<float>(resized_y) + 0.5F)
            * static_cast<float>(params.source_height)
            / static_cast<float>(params.resized_height)
            - 0.5F;

        rgb = sample_source_rgb(
            plane0,
            plane1,
            params,
            source_x,
            source_y
        );
    }

    const float first = params.output_is_rgb != 0 ? rgb.x : rgb.z;
    const float second = rgb.y;
    const float third = params.output_is_rgb != 0 ? rgb.z : rgb.x;

    return NormalizedPixel{
        normalized_channel(
            first,
            params.pixel_scale,
            params.mean0,
            params.inv_std0
        ),
        normalized_channel(
            second,
            params.pixel_scale,
            params.mean1,
            params.inv_std1
        ),
        normalized_channel(
            third,
            params.pixel_scale,
            params.mean2,
            params.inv_std2
        )
    };
}

template <typename OutputType>
__global__ void fused_preprocess_kernel(
    const std::uint8_t* plane0,
    const std::uint8_t* plane1,
    OutputType* output,
    PreprocessKernelParams params
) {
    const int x0 = (
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x)
    ) * 2;

    const int y = static_cast<int>(blockIdx.y)
        * static_cast<int>(blockDim.y)
        + static_cast<int>(threadIdx.y);

    if (y >= params.network_height || x0 >= params.network_width) {
        return;
    }

    const int x1 = x0 + 1;

    const NormalizedPixel first = preprocess_one_pixel(
        plane0,
        plane1,
        params,
        x0,
        y
    );

    const NormalizedPixel second = x1 < params.network_width
        ? preprocess_one_pixel(
              plane0,
              plane1,
              params,
              x1,
              y
          )
        : first;

    const std::size_t plane_size =
        static_cast<std::size_t>(params.network_width)
        * static_cast<std::size_t>(params.network_height);

    const std::size_t index =
        static_cast<std::size_t>(y)
        * static_cast<std::size_t>(params.network_width)
        + static_cast<std::size_t>(x0);

    if constexpr (std::is_same<OutputType, float>::value) {
        if (x1 < params.network_width) {
            reinterpret_cast<float2*>(output + index)[0] =
                make_float2(first.channel0, second.channel0);

            reinterpret_cast<float2*>(output + plane_size + index)[0] =
                make_float2(first.channel1, second.channel1);

            reinterpret_cast<float2*>(output + 2U * plane_size + index)[0] =
                make_float2(first.channel2, second.channel2);
        } else {
            output[index] = first.channel0;
            output[plane_size + index] = first.channel1;
            output[2U * plane_size + index] = first.channel2;
        }
    } else {
        if (x1 < params.network_width) {
            reinterpret_cast<__half2*>(output + index)[0] =
                __floats2half2_rn(first.channel0, second.channel0);

            reinterpret_cast<__half2*>(output + plane_size + index)[0] =
                __floats2half2_rn(first.channel1, second.channel1);

            reinterpret_cast<__half2*>(output + 2U * plane_size + index)[0] =
                __floats2half2_rn(first.channel2, second.channel2);
        } else {
            output[index] = __float2half_rn(first.channel0);
            output[plane_size + index] = __float2half_rn(first.channel1);
            output[2U * plane_size + index] = __float2half_rn(first.channel2);
        }
    }
}

template <typename InputType>
__device__ __forceinline__ float output_value(
    const InputType* data,
    std::size_t index
) {
    if constexpr (std::is_same<InputType, float>::value) {
        return data[index];
    } else {
        return __half2float(data[index]);
    }
}

template <typename InputType>
__global__ void output_to_float_kernel(
    const InputType* input,
    float* output,
    std::size_t count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x)
        * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    if (index < count) {
        output[index] = output_value(input, index);
    }
}

template <typename InputType>
__global__ void fused_postprocess_kernel(
    const InputType* model_output,
    PostprocessKernelParams params,
    DevicePostprocessResult* result
) {
    __shared__ std::uint32_t scan[kPostprocessThreads];
    __shared__ std::uint32_t rejected_confidence;
    __shared__ std::uint32_t rejected_malformed;
    __shared__ std::uint32_t rejected_degenerate;
    __shared__ std::uint32_t confidence_cutoff;

    const unsigned int thread = threadIdx.x;

    if (thread == 0U) {
        rejected_confidence = 0U;
        rejected_malformed = 0U;
        rejected_degenerate = 0U;
        confidence_cutoff = static_cast<std::uint32_t>(
            kModelMaximumDetections
        );
    }

    __syncthreads();

    if (thread < kModelMaximumDetections) {
        const std::size_t row =
            static_cast<std::size_t>(thread) * kDetectionValueCount;

        const float confidence = output_value(
            model_output,
            row + 4U
        );

        if (
            params.sorted_by_confidence != 0U
            && isfinite(confidence)
            && !(confidence > params.confidence_threshold)
        ) {
            atomicMin(&confidence_cutoff, thread);
        }
    }

    __syncthreads();

    bool valid = false;
    DeviceDetectionPacked decoded{};

    const bool active = thread < kModelMaximumDetections
        && (
            params.sorted_by_confidence == 0U
            || thread <= confidence_cutoff
        );

    if (active) {
        const std::size_t row =
            static_cast<std::size_t>(thread) * kDetectionValueCount;

        const float confidence = output_value(model_output, row + 4U);

        if (!isfinite(confidence)) {
            atomicAdd(&rejected_malformed, 1U);
        } else if (!(confidence > params.confidence_threshold)) {
            atomicAdd(&rejected_confidence, 1U);
        } else if (confidence > 1.0F) {
            atomicAdd(&rejected_malformed, 1U);
        } else {
            const float x_min = output_value(model_output, row + 0U);
            const float y_min = output_value(model_output, row + 1U);
            const float x_max = output_value(model_output, row + 2U);
            const float y_max = output_value(model_output, row + 3U);
            const float class_value = output_value(model_output, row + 5U);

            const float rounded_class = roundf(class_value);

            const bool coordinates_are_finite =
                isfinite(x_min)
                && isfinite(y_min)
                && isfinite(x_max)
                && isfinite(y_max);

            const bool class_is_valid =
                isfinite(class_value)
                && fabsf(class_value - rounded_class)
                    <= params.class_id_tolerance
                && rounded_class >= 0.0F
                && rounded_class
                    < static_cast<float>(params.class_count);

            if (!coordinates_are_finite || !class_is_valid) {
                atomicAdd(&rejected_malformed, 1U);
            } else {
                float restored_x_min = x_min * params.inverse_scale_x
                    + params.offset_x;

                float restored_y_min = y_min * params.inverse_scale_y
                    + params.offset_y;

                float restored_x_max = x_max * params.inverse_scale_x
                    + params.offset_x;

                float restored_y_max = y_max * params.inverse_scale_y
                    + params.offset_y;

                if (params.clip_boxes != 0U) {
                    restored_x_min = clamp_float(
                        restored_x_min,
                        0.0F,
                        params.maximum_x
                    );

                    restored_y_min = clamp_float(
                        restored_y_min,
                        0.0F,
                        params.maximum_y
                    );

                    restored_x_max = clamp_float(
                        restored_x_max,
                        0.0F,
                        params.maximum_x
                    );

                    restored_y_max = clamp_float(
                        restored_y_max,
                        0.0F,
                        params.maximum_y
                    );
                }

                const bool positive_area =
                    restored_x_max > restored_x_min
                    && restored_y_max > restored_y_min;

                if (
                    params.discard_degenerate != 0U
                    && !positive_area
                ) {
                    atomicAdd(&rejected_degenerate, 1U);
                } else {
                    valid = true;
                    decoded = DeviceDetectionPacked{
                        restored_x_min,
                        restored_y_min,
                        restored_x_max,
                        restored_y_max,
                        confidence,
                        static_cast<std::int32_t>(rounded_class)
                    };
                }
            }
        }
    }

    scan[thread] = valid ? 1U : 0U;
    __syncthreads();

    for (unsigned int offset = 1U; offset < kPostprocessThreads; offset <<= 1U) {
        std::uint32_t addend = 0U;

        if (thread >= offset) {
            addend = scan[thread - offset];
        }

        __syncthreads();
        scan[thread] += addend;
        __syncthreads();
    }

    if (valid) {
        const std::uint32_t position = scan[thread] - 1U;

        if (position < params.maximum_detections) {
            result->detections[position] = decoded;
        }
    }

    __syncthreads();

    if (thread == 0U) {
        const std::uint32_t total_valid =
            scan[kModelMaximumDetections - 1U];

        const std::uint32_t examined =
            params.sorted_by_confidence != 0U
            && confidence_cutoff
                < static_cast<std::uint32_t>(kModelMaximumDetections)
                ? confidence_cutoff + 1U
                : static_cast<std::uint32_t>(kModelMaximumDetections);

        const std::uint32_t inferred_remaining_rejections =
            params.sorted_by_confidence != 0U
            && confidence_cutoff
                < static_cast<std::uint32_t>(kModelMaximumDetections)
                ? static_cast<std::uint32_t>(kModelMaximumDetections)
                    - examined
                : 0U;

        result->summary = DevicePostprocessSummary{
            examined,
            rejected_confidence + inferred_remaining_rejections,
            rejected_malformed,
            rejected_degenerate,
            min(total_valid, params.maximum_detections),
            total_valid > params.maximum_detections ? 1U : 0U
        };
    }
}

[[nodiscard]] PreprocessKernelParams make_preprocess_params(
    const SourceDescriptor& source,
    const LetterboxTransform& transform,
    const PreprocessConfig& config
) {
    return PreprocessKernelParams{
        source.width,
        source.height,
        transform.resized_size.width,
        transform.resized_size.height,
        transform.network_size.width,
        transform.network_size.height,
        transform.pad_left,
        transform.pad_top,
        source.pitch0,
        source.pitch1,
        static_cast<int>(source.mode),
        config.output_channel_order == ChannelOrder::RGB ? 1 : 0,
        static_cast<int>(source.yuv_matrix),
        static_cast<int>(source.yuv_range),
        config.pixel_scale,
        config.mean[0],
        config.mean[1],
        config.mean[2],
        1.0F / config.stddev[0],
        1.0F / config.stddev[1],
        1.0F / config.stddev[2],
        static_cast<float>(config.padding_bgr[2]),
        static_cast<float>(config.padding_bgr[1]),
        static_cast<float>(config.padding_bgr[0])
    };
}

[[nodiscard]] PostprocessKernelParams make_postprocess_params(
    const LetterboxTransform& transform,
    const PostprocessConfig& config
) {
    return PostprocessKernelParams{
        config.confidence_threshold,
        config.class_id_tolerance,
        static_cast<std::uint32_t>(config.maximum_detections),
        static_cast<std::uint32_t>(config.class_count),
        config.clip_boxes ? 1U : 0U,
        config.discard_degenerate_boxes ? 1U : 0U,
        config.output_sorted_by_confidence ? 1U : 0U,
        1.0F / transform.scale_x,
        1.0F / transform.scale_y,
        -static_cast<float>(transform.pad_left) / transform.scale_x,
        -static_cast<float>(transform.pad_top) / transform.scale_y,
        static_cast<float>(transform.original_size.width),
        static_cast<float>(transform.original_size.height)
    };
}

void launch_preprocess(
    const SourceDescriptor& source,
    const LetterboxTransform& transform,
    const PreprocessConfig& config,
    TensorElementType output_type,
    void* device_output,
    cudaStream_t stream
) {
    const PreprocessKernelParams params = make_preprocess_params(
        source,
        transform,
        config
    );

    const dim3 block{kPreprocessBlockX, kPreprocessBlockY, 1U};

    const dim3 grid{
        static_cast<unsigned int>(
            (params.network_width + static_cast<int>(block.x) * 2 - 1)
            / (static_cast<int>(block.x) * 2)
        ),
        static_cast<unsigned int>(
            (params.network_height + static_cast<int>(block.y) - 1)
            / static_cast<int>(block.y)
        ),
        1U
    };

    const auto* plane0 = static_cast<const std::uint8_t*>(source.plane0);
    const auto* plane1 = static_cast<const std::uint8_t*>(source.plane1);

    if (output_type == TensorElementType::Float32) {
        fused_preprocess_kernel<float><<<grid, block, 0U, stream>>>(
            plane0,
            plane1,
            static_cast<float*>(device_output),
            params
        );
    } else {
        fused_preprocess_kernel<__half><<<grid, block, 0U, stream>>>(
            plane0,
            plane1,
            static_cast<__half*>(device_output),
            params
        );
    }

    check_last_kernel("fused_preprocess_kernel launch");
}

void launch_postprocess(
    const void* model_output,
    TensorElementType output_type,
    const LetterboxTransform& transform,
    const PostprocessConfig& config,
    DevicePostprocessResult* result,
    cudaStream_t stream
) {
    const PostprocessKernelParams params = make_postprocess_params(
        transform,
        config
    );

    if (output_type == TensorElementType::Float32) {
        fused_postprocess_kernel<float>
            <<<1U, kPostprocessThreads, 0U, stream>>>(
                static_cast<const float*>(model_output),
                params,
                result
            );
    } else {
        fused_postprocess_kernel<__half>
            <<<1U, kPostprocessThreads, 0U, stream>>>(
                static_cast<const __half*>(model_output),
                params,
                result
            );
    }

    check_last_kernel("fused_postprocess_kernel launch");
}

void launch_output_conversion(
    const void* model_output,
    TensorElementType output_type,
    float* float_output,
    cudaStream_t stream
) {
    constexpr unsigned int threads = 256U;
    const unsigned int blocks = static_cast<unsigned int>(
        (kModelOutputElementCount + threads - 1U) / threads
    );

    if (output_type == TensorElementType::Float32) {
        check_cuda(
            cudaMemcpyAsync(
                float_output,
                model_output,
                kModelOutputElementCount * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync TensorRT FP32 output"
        );
        return;
    }

    output_to_float_kernel<__half><<<blocks, threads, 0U, stream>>>(
        static_cast<const __half*>(model_output),
        float_output,
        kModelOutputElementCount
    );

    check_last_kernel("output_to_float_kernel launch");
}

struct PipelineSlot final {
    explicit PipelineSlot(
        std::uint32_t index,
        int device_id,
        bool high_priority,
        std::size_t auxiliary_stream_count
    )
        : slot_index(index),
          stream(device_id, high_priority),
          completion(true) {
        auxiliary_streams.reserve(auxiliary_stream_count);
        auxiliary_stream_handles.reserve(auxiliary_stream_count);

        for (std::size_t stream_index = 0U;
             stream_index < auxiliary_stream_count;
             ++stream_index) {
            auxiliary_streams.emplace_back(
                device_id,
                high_priority
            );
            auxiliary_stream_handles.push_back(
                auxiliary_streams.back().get()
            );
        }
    }

    std::uint32_t slot_index = 0U;

    // Destruction order is deliberate. Reverse member destruction releases
    // the CUDA graph first, then the TensorRT context, then every buffer, and
    // finally the completion event, auxiliary streams, and main stream.
    CudaStream stream;
    std::vector<CudaStream> auxiliary_streams{};
    std::vector<cudaStream_t> auxiliary_stream_handles{};
    CudaEvent completion;

    PinnedBuffer host_bgr_staging{};
    PinnedBuffer host_result{};
    PinnedBuffer host_raw_output{};

    DeviceBuffer device_bgr_staging{};
    DeviceBuffer device_input{};
    DeviceBuffer device_output{};
    DeviceBuffer device_postprocess_result{};
    DeviceBuffer device_float_output{};
    DeviceBuffer context_workspace{};

    std::unique_ptr<nvinfer1::IExecutionContext> context{};
    CudaGraph graph;

    std::size_t context_workspace_bytes = 0U;

    LetterboxTransform transform{};
    SourceDescriptor source{};
    GraphKey graph_key{};

    bool graph_key_valid = false;
    bool graph_warmed = false;
    bool graph_capture_failed = false;

    bool in_flight = false;
    bool used_cuda_graph = false;
    bool source_was_device_resident = false;

    std::uint64_t generation = 0U;
    std::uint64_t sequence = 0U;
};

[[nodiscard]] GraphKey make_graph_key(
    const SourceDescriptor& source
) noexcept {
    return GraphKey{
        source.mode,
        source.width,
        source.height,
        source.pitch0,
        source.pitch1,
        source.mode == SourceMode::InternalPinnedBgr
            ? nullptr
            : source.plane0,
        source.mode == SourceMode::InternalPinnedBgr
            ? nullptr
            : source.plane1
    };
}

[[nodiscard]] bool source_is_device_resident(
    SourceMode mode
) noexcept {
    return mode == SourceMode::DeviceBgr
        || mode == SourceMode::DeviceRgb
        || mode == SourceMode::DeviceNv12;
}

}  // namespace

class NativeTensorRtPipeline::Impl final {
public:
    explicit Impl(
        NativeTensorRtPipelineConfig config
    )
        : config_(std::move(config)),
          cpu_postprocessor_(config_.postprocess) {
        config_.validate();
        validate_engine_file(config_.engine_path);

        check_cuda(cudaSetDevice(config_.device_id), "cudaSetDevice");
        initialize_tensorrt_plugins();

        runtime_.reset(
            nvinfer1::createInferRuntime(tensorrt_logger())
        );

        if (runtime_ == nullptr) {
            throw std::runtime_error("TensorRT failed to create IRuntime.");
        }

        const std::vector<std::uint8_t> bytes = read_binary_file(
            config_.engine_path
        );

        engine_.reset(
            runtime_->deserializeCudaEngine(bytes.data(), bytes.size())
        );

        if (engine_ == nullptr) {
            throw std::runtime_error(
                "TensorRT failed to deserialize the native engine."
            );
        }

        inspect_engine();
        create_slots();
        populate_runtime_info();
    }

    ~Impl() {
        static_cast<void>(cudaSetDevice(config_.device_id));

        for (const auto& slot : slots_) {
            if (slot != nullptr && slot->in_flight) {
                static_cast<void>(cudaStreamSynchronize(slot->stream.get()));
            }
        }
    }

    [[nodiscard]] const NativeTensorRtPipelineConfig& config() const noexcept {
        return config_;
    }

    [[nodiscard]] const GpuPipelineRuntimeInfo& runtime_info() const noexcept {
        return runtime_info_;
    }

    [[nodiscard]] GpuPipelineTicket submit(
        const cv::Mat& image
    ) {
        if (
            image.empty()
            || image.dims != 2
            || image.type() != CV_8UC3
            || image.data == nullptr
        ) {
            throw std::invalid_argument(
                "submit() requires a non-empty CV_8UC3 BGR image."
            );
        }

        if (
            image.cols > config_.maximum_host_image_size.width
            || image.rows > config_.maximum_host_image_size.height
        ) {
            throw std::invalid_argument(
                "Input image exceeds maximum_host_image_size."
            );
        }

        PipelineSlot& slot = acquire_slot();

        const std::size_t row_bytes = checked_multiply(
            static_cast<std::size_t>(image.cols),
            3U,
            "BGR row size"
        );

        auto* destination = static_cast<std::uint8_t*>(
            slot.host_bgr_staging.data()
        );

        if (
            image.isContinuous()
            && image.step[0] == row_bytes
        ) {
            std::memcpy(
                destination,
                image.data,
                row_bytes * static_cast<std::size_t>(image.rows)
            );
        } else {
            for (int row = 0; row < image.rows; ++row) {
                std::memcpy(
                    destination
                        + static_cast<std::size_t>(row) * row_bytes,
                    image.ptr(row),
                    row_bytes
                );
            }
        }

        slot.source = SourceDescriptor{
            SourceMode::InternalPinnedBgr,
            slot.host_bgr_staging.data(),
            nullptr,
            image.cols,
            image.rows,
            row_bytes,
            0U,
            YuvColorMatrix::Bt709,
            YuvRange::Limited
        };

        return enqueue_slot(slot);
    }

    [[nodiscard]] GpuPipelineTicket submit_pinned_bgr(
        PinnedBgrImageView image
    ) {
        if (!image.valid()) {
            throw std::invalid_argument(
                "submit_pinned_bgr() received an invalid image view."
            );
        }

        if (
            image.width > config_.maximum_host_image_size.width
            || image.height > config_.maximum_host_image_size.height
        ) {
            throw std::invalid_argument(
                "Pinned image exceeds maximum_host_image_size."
            );
        }

        PipelineSlot& slot = acquire_slot();

        slot.source = SourceDescriptor{
            SourceMode::ExternalPinnedBgr,
            image.data,
            nullptr,
            image.width,
            image.height,
            image.row_stride_bytes,
            0U,
            YuvColorMatrix::Bt709,
            YuvRange::Limited
        };

        return enqueue_slot(slot);
    }

    [[nodiscard]] GpuPipelineTicket submit_device(
        DeviceImageView image
    ) {
        if (!image.valid()) {
            throw std::invalid_argument(
                "submit_device() received an invalid device image view."
            );
        }

        if (image.device_id != config_.device_id) {
            throw std::invalid_argument(
                "Device image and TensorRT pipeline use different CUDA "
                "device IDs."
            );
        }

        if (
            image.format == DeviceImageFormat::Nv12
            && ((image.width & 1) != 0 || (image.height & 1) != 0)
        ) {
            throw std::invalid_argument(
                "NV12 input requires even width and height."
            );
        }

        PipelineSlot& slot = acquire_slot();

        SourceMode mode = SourceMode::DeviceBgr;

        switch (image.format) {
            case DeviceImageFormat::Bgr8:
                mode = SourceMode::DeviceBgr;
                break;
            case DeviceImageFormat::Rgb8:
                mode = SourceMode::DeviceRgb;
                break;
            case DeviceImageFormat::Nv12:
                mode = SourceMode::DeviceNv12;
                break;
        }

        slot.source = SourceDescriptor{
            mode,
            image.plane0,
            image.plane1,
            image.width,
            image.height,
            image.plane0_pitch_bytes,
            image.plane1_pitch_bytes,
            image.yuv_matrix,
            image.yuv_range
        };

        return enqueue_slot(slot);
    }

    [[nodiscard]] bool ready(
        GpuPipelineTicket ticket
    ) const {
        const PipelineSlot& slot = validate_ticket(ticket);
        return slot.completion.ready();
    }

    [[nodiscard]] GpuPipelineResult collect(
        GpuPipelineTicket ticket
    ) {
        PipelineSlot& slot = validate_ticket(ticket);
        slot.completion.synchronize();

        try {
            GpuPipelineResult result{};
            result.transform = slot.transform;
            result.sequence_number = slot.sequence;
            result.used_cuda_graph = slot.used_cuda_graph;
            result.source_was_device_resident =
                slot.source_was_device_resident;

            if (config_.enable_gpu_postprocess) {
                const auto* host = static_cast<const DevicePostprocessResult*>(
                    slot.host_result.data()
                );

                const std::size_t detection_count =
                    static_cast<std::size_t>(
                        host->summary.detections_written
                    );

                if (detection_count > kModelMaximumDetections) {
                    throw std::runtime_error(
                        "GPU postprocessor returned an impossible detection count."
                    );
                }

                Detection* const detections =
                    result.detections.resize_for_overwrite(
                        detection_count
                    );

                if (detections == nullptr) {
                    throw std::logic_error(
                        "Failed to reserve compact GPU detection storage."
                    );
                }

                for (std::size_t index = 0U; index < detection_count; ++index) {
                    const DeviceDetectionPacked& source =
                        host->detections[index];

                    detections[index] = Detection{
                        BoundingBox{
                            source.x_min,
                            source.y_min,
                            source.x_max,
                            source.y_max
                        },
                        source.confidence,
                        source.class_id
                    };
                }

                result.summary = PostprocessSummary{
                    static_cast<std::size_t>(
                        host->summary.candidates_examined
                    ),
                    static_cast<std::size_t>(
                        host->summary.rejected_by_confidence
                    ),
                    static_cast<std::size_t>(
                        host->summary.rejected_as_malformed
                    ),
                    static_cast<std::size_t>(
                        host->summary.rejected_as_degenerate
                    ),
                    detection_count,
                    host->summary.maximum_reached != 0U
                };
            } else {
                const auto* raw = static_cast<const float*>(
                    slot.host_raw_output.data()
                );

                result.summary = cpu_postprocessor_.process(
                    DetectionTensorView::contiguous(
                        raw,
                        kModelMaximumDetections
                    ),
                    slot.transform,
                    result.detections
                );
            }

            slot.in_flight = false;
            --in_flight_count_;
            return result;
        } catch (...) {
            slot.in_flight = false;
            --in_flight_count_;
            throw;
        }
    }

    [[nodiscard]] GpuPipelineResult run(
        const cv::Mat& image
    ) {
        return collect(submit(image));
    }

    [[nodiscard]] GpuPipelineResult run_device(
        DeviceImageView image
    ) {
        return collect(submit_device(image));
    }

    [[nodiscard]] std::size_t pipeline_depth() const noexcept {
        return slots_.size();
    }

    [[nodiscard]] std::size_t in_flight_count() const noexcept {
        return in_flight_count_;
    }

private:
    void inspect_engine() {
        if (engine_->getNbIOTensors() != 2) {
            throw std::runtime_error(
                "The optimized pipeline requires exactly one input and one "
                "output tensor."
            );
        }

        for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* const name = engine_->getIOTensorName(index);

            if (name == nullptr) {
                throw std::runtime_error(
                    "TensorRT returned a null I/O tensor name."
                );
            }

            const nvinfer1::TensorIOMode mode =
                engine_->getTensorIOMode(name);

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input_name_ = name;
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                output_name_ = name;
            }
        }

        if (
            input_name_ != config_.expected_input_name
            || output_name_ != config_.expected_output_name
        ) {
            throw std::runtime_error(
                "TensorRT tensor names do not match the configured contract."
            );
        }

        if (
            engine_->getTensorLocation(input_name_.c_str())
                != nvinfer1::TensorLocation::kDEVICE
            || engine_->getTensorLocation(output_name_.c_str())
                != nvinfer1::TensorLocation::kDEVICE
        ) {
            throw std::runtime_error(
                "The optimized pipeline requires device-resident TensorRT "
                "input and output tensors."
            );
        }

        if (
            engine_->getTensorFormat(input_name_.c_str())
                != nvinfer1::TensorFormat::kLINEAR
            || engine_->getTensorFormat(output_name_.c_str())
                != nvinfer1::TensorFormat::kLINEAR
        ) {
            throw std::runtime_error(
                "The optimized pipeline requires linear TensorRT I/O formats."
            );
        }

        input_type_ = tensor_type_from_tensorrt(
            engine_->getTensorDataType(input_name_.c_str()),
            "TensorRT input"
        );

        output_type_ = tensor_type_from_tensorrt(
            engine_->getTensorDataType(output_name_.c_str()),
            "TensorRT output"
        );

        input_bytes_ = checked_multiply(
            kModelInputElementCount,
            tensor_element_size(input_type_),
            "TensorRT input byte count"
        );

        output_bytes_ = checked_multiply(
            kModelOutputElementCount,
            tensor_element_size(output_type_),
            "TensorRT output byte count"
        );
    }

    void create_slots() {
        slots_.reserve(config_.pipeline_depth);

        const std::size_t maximum_source_bytes = checked_image_bytes(
            config_.maximum_host_image_size
        );

        const int reported_auxiliary_stream_count =
            engine_->getNbAuxStreams();

        if (reported_auxiliary_stream_count < 0) {
            throw std::runtime_error(
                "TensorRT reported a negative auxiliary-stream count."
            );
        }

        const std::size_t auxiliary_stream_count =
            config_.use_engine_auxiliary_streams
                ? static_cast<std::size_t>(
                      reported_auxiliary_stream_count
                  )
                : 0U;

        for (std::size_t index = 0U; index < config_.pipeline_depth; ++index) {
            auto slot = std::make_unique<PipelineSlot>(
                static_cast<std::uint32_t>(index),
                config_.device_id,
                config_.use_high_priority_streams,
                auxiliary_stream_count
            );

            slot->context.reset(
                engine_->createExecutionContext(
                    nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED
                )
            );

            if (slot->context == nullptr) {
                throw std::runtime_error(
                    "TensorRT failed to create a user-managed execution "
                    "context for a pipeline slot."
                );
            }

            if (config_.minimize_runtime_instrumentation) {
                slot->context->setEnqueueEmitsProfile(false);
                static_cast<void>(
                    slot->context->setNvtxVerbosity(
                        nvinfer1::ProfilingVerbosity::kNONE
                    )
                );
            }

            if (config_.persistent_cache_limit_bytes != 0U) {
                slot->context->setPersistentCacheLimit(
                    config_.persistent_cache_limit_bytes
                );
            }

            const nvinfer1::Dims engine_input_shape =
                engine_->getTensorShape(input_name_.c_str());

            if (contains_dynamic_dimension(engine_input_shape)) {
                if (!slot->context->setInputShape(
                        input_name_.c_str(),
                        shape_to_tensorrt(kModelInputShape)
                    )) {
                    throw std::runtime_error(
                        "TensorRT rejected fixed [1,3,640,640] input shape."
                    );
                }
            }

            const auto input_shape = fixed_shape_from_tensorrt<4U>(
                slot->context->getTensorShape(input_name_.c_str()),
                "TensorRT input"
            );

            const auto output_shape = fixed_shape_from_tensorrt<3U>(
                slot->context->getTensorShape(output_name_.c_str()),
                "TensorRT output"
            );

            if (input_shape != kModelInputShape) {
                throw std::runtime_error(
                    "TensorRT input shape is not [1,3,640,640]."
                );
            }

            if (output_shape != kModelOutputShape) {
                throw std::runtime_error(
                    "TensorRT output shape is not [1,300,6]."
                );
            }

            slot->context_workspace_bytes =
                slot->context->updateDeviceMemorySizeForShapes();

            if (
                slot->context_workspace_bytes
                > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max()
                )
            ) {
                throw std::runtime_error(
                    "TensorRT context workspace does not fit int64_t."
                );
            }

            if (slot->context_workspace_bytes != 0U) {
                slot->context_workspace.allocate(
                    slot->context_workspace_bytes
                );

                slot->context->setDeviceMemoryV2(
                    slot->context_workspace.data(),
                    static_cast<std::int64_t>(
                        slot->context_workspace_bytes
                    )
                );
            }

            slot->host_bgr_staging.allocate(maximum_source_bytes);
            slot->device_bgr_staging.allocate(maximum_source_bytes);
            slot->device_input.allocate(input_bytes_);
            slot->device_output.allocate(output_bytes_);

            if (config_.enable_gpu_postprocess) {
                slot->device_postprocess_result.allocate(
                    sizeof(DevicePostprocessResult)
                );
                slot->host_result.allocate(
                    sizeof(DevicePostprocessResult)
                );
            } else {
                slot->device_float_output.allocate(
                    kModelOutputElementCount * sizeof(float)
                );
                slot->host_raw_output.allocate(
                    kModelOutputElementCount * sizeof(float)
                );
            }

            if (!slot->context->setInputTensorAddress(
                    input_name_.c_str(),
                    slot->device_input.data()
                )) {
                throw std::runtime_error(
                    "TensorRT rejected a persistent input address."
                );
            }

            if (!slot->context->setOutputTensorAddress(
                    output_name_.c_str(),
                    slot->device_output.data()
                )) {
                throw std::runtime_error(
                    "TensorRT rejected a persistent output address."
                );
            }

            slots_.push_back(std::move(slot));
        }
    }

    void populate_runtime_info() {
        runtime_info_.runtime_name = "TensorRT fused CUDA pipeline";
        runtime_info_.runtime_version = tensorrt_runtime_version();
        runtime_info_.input_element_type = input_type_;
        runtime_info_.output_element_type = output_type_;
        runtime_info_.input_shape = kModelInputShape;
        runtime_info_.output_shape = kModelOutputShape;
        runtime_info_.gpu_postprocess_enabled =
            config_.enable_gpu_postprocess;
        runtime_info_.cuda_graph_enabled = config_.enable_cuda_graph;

        std::size_t host_per_slot = checked_image_bytes(
            config_.maximum_host_image_size
        );

        std::size_t device_per_slot = host_per_slot
            + input_bytes_
            + output_bytes_;

        if (config_.enable_gpu_postprocess) {
            host_per_slot += sizeof(DevicePostprocessResult);
            device_per_slot += sizeof(DevicePostprocessResult);
        } else {
            host_per_slot += kModelOutputElementCount * sizeof(float);
            device_per_slot += kModelOutputElementCount * sizeof(float);
        }

        if (!slots_.empty()) {
            device_per_slot += slots_.front()->context_workspace_bytes;
        }

        runtime_info_.memory.pipeline_depth = slots_.size();
        runtime_info_.memory.auxiliary_streams_per_slot =
            slots_.empty()
                ? 0U
                : slots_.front()->auxiliary_streams.size();
        runtime_info_.memory.pinned_host_bytes_per_slot = host_per_slot;
        runtime_info_.memory.device_bytes_per_slot = device_per_slot;
        runtime_info_.memory.total_pinned_host_bytes = checked_multiply(
            host_per_slot,
            slots_.size(),
            "Total pinned pipeline memory"
        );
        runtime_info_.memory.total_device_bytes = checked_multiply(
            device_per_slot,
            slots_.size(),
            "Total device pipeline memory"
        );
    }

    [[nodiscard]] PipelineSlot& acquire_slot() {
        for (std::size_t offset = 0U; offset < slots_.size(); ++offset) {
            const std::size_t index =
                (next_slot_index_ + offset) % slots_.size();

            PipelineSlot& slot = *slots_[index];

            if (!slot.in_flight) {
                next_slot_index_ = (index + 1U) % slots_.size();
                return slot;
            }
        }

        throw std::runtime_error(
            "The asynchronous pipeline is full. Collect an outstanding "
            "ticket before submitting another frame."
        );
    }

    [[nodiscard]] GpuPipelineTicket enqueue_slot(
        PipelineSlot& slot
    ) {
        slot.transform =
            ImageProcessor::calculate_letterbox_transform_unchecked(
            ImageSize{slot.source.width, slot.source.height},
            config_.preprocess
        );

        slot.source_was_device_resident = source_is_device_resident(
            slot.source.mode
        );

        const bool external =
            slot.source.mode != SourceMode::InternalPinnedBgr;

        const bool graph_eligible =
            config_.enable_cuda_graph
            && (!external || config_.capture_external_sources);

        const GraphKey new_key = make_graph_key(slot.source);

        if (!slot.graph_key_valid || !(slot.graph_key == new_key)) {
            slot.graph.reset();
            slot.graph_key = new_key;
            slot.graph_key_valid = true;
            slot.graph_warmed = false;
            slot.graph_capture_failed = false;
        }

        try {
            slot.used_cuda_graph = false;

            if (graph_eligible && slot.graph.ready()) {
                slot.graph.launch(slot.stream.get());
                slot.used_cuda_graph = true;
            } else if (
                graph_eligible
                && slot.graph_warmed
                && !slot.graph_capture_failed
            ) {
                bool captured = false;

                try {
                    slot.graph.begin(slot.stream.get());
                    enqueue_operations(slot);
                    slot.graph.end_and_instantiate(slot.stream.get());
                    slot.graph.launch(slot.stream.get());
                    captured = true;
                    slot.used_cuda_graph = true;
                } catch (const std::exception& exception) {
                    slot.graph.abort(slot.stream.get());
                    slot.graph_capture_failed = true;

                    if (config_.require_cuda_graph) {
                        throw;
                    }

                    std::cerr
                        << "[CUDA graph warning] Capture disabled for pipeline "
                        << "slot after failure: "
                        << exception.what()
                        << '\n';
                }

                if (!captured) {
                    enqueue_operations(slot);
                }
            } else {
                enqueue_operations(slot);

                if (graph_eligible) {
                    slot.graph_warmed = true;
                }
            }

            slot.completion.record(slot.stream.get());
            slot.in_flight = true;
            ++in_flight_count_;

            ++slot.generation;
            if (slot.generation == 0U) {
                ++slot.generation;
            }

            slot.sequence = ++sequence_counter_;

            return GpuPipelineTicket{
                slot.slot_index,
                slot.generation
            };
        } catch (...) {
            // A backend or capture failure can occur after work was already
            // placed on the slot stream. Drain it before making the slot
            // available again so later submissions never race stale work.
            static_cast<void>(cudaStreamSynchronize(slot.stream.get()));
            slot.in_flight = false;
            throw;
        }
    }

    void enqueue_operations(PipelineSlot& slot) {
        SourceDescriptor kernel_source = slot.source;

        if (slot.source.mode == SourceMode::InternalPinnedBgr) {
            const std::size_t bytes = checked_multiply(
                slot.source.pitch0,
                static_cast<std::size_t>(slot.source.height),
                "Packed BGR transfer size"
            );

            check_cuda(
                cudaMemcpyAsync(
                    slot.device_bgr_staging.data(),
                    slot.host_bgr_staging.data(),
                    bytes,
                    cudaMemcpyHostToDevice,
                    slot.stream.get()
                ),
                "cudaMemcpyAsync packed BGR input"
            );

            kernel_source.plane0 = slot.device_bgr_staging.data();
        } else if (slot.source.mode == SourceMode::ExternalPinnedBgr) {
            const std::size_t row_bytes = checked_multiply(
                static_cast<std::size_t>(slot.source.width),
                3U,
                "Pinned BGR row size"
            );

            check_cuda(
                cudaMemcpy2DAsync(
                    slot.device_bgr_staging.data(),
                    row_bytes,
                    slot.source.plane0,
                    slot.source.pitch0,
                    row_bytes,
                    static_cast<std::size_t>(slot.source.height),
                    cudaMemcpyHostToDevice,
                    slot.stream.get()
                ),
                "cudaMemcpy2DAsync pinned BGR input"
            );

            kernel_source.plane0 = slot.device_bgr_staging.data();
            kernel_source.pitch0 = row_bytes;
            kernel_source.mode = SourceMode::InternalPinnedBgr;
        }

        launch_preprocess(
            kernel_source,
            slot.transform,
            config_.preprocess,
            input_type_,
            slot.device_input.data(),
            slot.stream.get()
        );

        // TensorRT applies user-provided auxiliary streams to the next
        // enqueueV3 call, so refresh them for every ordinary or captured
        // enqueue. The streams are persistent, distinct, and non-blocking.
        if (!slot.auxiliary_stream_handles.empty()) {
            slot.context->setAuxStreams(
                slot.auxiliary_stream_handles.data(),
                static_cast<std::int32_t>(
                    slot.auxiliary_stream_handles.size()
                )
            );
        }

        if (!slot.context->enqueueV3(slot.stream.get())) {
            throw std::runtime_error("TensorRT enqueueV3 returned false.");
        }

        if (config_.enable_gpu_postprocess) {
            launch_postprocess(
                slot.device_output.data(),
                output_type_,
                slot.transform,
                config_.postprocess,
                static_cast<DevicePostprocessResult*>(
                    slot.device_postprocess_result.data()
                ),
                slot.stream.get()
            );

            check_cuda(
                cudaMemcpyAsync(
                    slot.host_result.data(),
                    slot.device_postprocess_result.data(),
                    sizeof(DevicePostprocessResult),
                    cudaMemcpyDeviceToHost,
                    slot.stream.get()
                ),
                "cudaMemcpyAsync compact detection result"
            );
        } else {
            launch_output_conversion(
                slot.device_output.data(),
                output_type_,
                static_cast<float*>(slot.device_float_output.data()),
                slot.stream.get()
            );

            check_cuda(
                cudaMemcpyAsync(
                    slot.host_raw_output.data(),
                    slot.device_float_output.data(),
                    kModelOutputElementCount * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    slot.stream.get()
                ),
                "cudaMemcpyAsync raw FP32 model output"
            );
        }
    }

    [[nodiscard]] PipelineSlot& validate_ticket(
        GpuPipelineTicket ticket
    ) {
        if (!ticket.valid() || ticket.slot_index >= slots_.size()) {
            throw std::invalid_argument("GPU pipeline ticket is invalid.");
        }

        PipelineSlot& slot = *slots_[ticket.slot_index];

        if (!slot.in_flight || slot.generation != ticket.generation) {
            throw std::invalid_argument(
                "GPU pipeline ticket is stale or has already been collected."
            );
        }

        return slot;
    }

    [[nodiscard]] const PipelineSlot& validate_ticket(
        GpuPipelineTicket ticket
    ) const {
        if (!ticket.valid() || ticket.slot_index >= slots_.size()) {
            throw std::invalid_argument("GPU pipeline ticket is invalid.");
        }

        const PipelineSlot& slot = *slots_[ticket.slot_index];

        if (!slot.in_flight || slot.generation != ticket.generation) {
            throw std::invalid_argument(
                "GPU pipeline ticket is stale or has already been collected."
            );
        }

        return slot;
    }

    NativeTensorRtPipelineConfig config_{};
    GpuPipelineRuntimeInfo runtime_info_{};
    Postprocessor cpu_postprocessor_{};

    std::unique_ptr<nvinfer1::IRuntime> runtime_{};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{};

    std::string input_name_{};
    std::string output_name_{};

    TensorElementType input_type_ = TensorElementType::Float32;
    TensorElementType output_type_ = TensorElementType::Float32;

    std::size_t input_bytes_ = 0U;
    std::size_t output_bytes_ = 0U;

    std::vector<std::unique_ptr<PipelineSlot>> slots_{};
    std::size_t next_slot_index_ = 0U;
    std::size_t in_flight_count_ = 0U;
    std::uint64_t sequence_counter_ = 0U;
};

NativeTensorRtPipeline::NativeTensorRtPipeline(
    NativeTensorRtPipelineConfig config
)
    : implementation_(
          std::make_unique<Impl>(std::move(config))
      ) {}

NativeTensorRtPipeline::~NativeTensorRtPipeline() = default;

NativeTensorRtPipeline::NativeTensorRtPipeline(
    NativeTensorRtPipeline&& other
) noexcept = default;

NativeTensorRtPipeline& NativeTensorRtPipeline::operator=(
    NativeTensorRtPipeline&& other
) noexcept = default;

const NativeTensorRtPipelineConfig& NativeTensorRtPipeline::config() const {
    return implementation().config();
}

const GpuPipelineRuntimeInfo& NativeTensorRtPipeline::runtime_info() const {
    return implementation().runtime_info();
}

GpuPipelineTicket NativeTensorRtPipeline::submit(
    const cv::Mat& image
) {
    return implementation().submit(image);
}

GpuPipelineTicket NativeTensorRtPipeline::submit_pinned_bgr(
    PinnedBgrImageView image
) {
    return implementation().submit_pinned_bgr(image);
}

GpuPipelineTicket NativeTensorRtPipeline::submit_device(
    DeviceImageView image
) {
    return implementation().submit_device(image);
}

bool NativeTensorRtPipeline::ready(
    GpuPipelineTicket ticket
) const {
    return implementation().ready(ticket);
}

GpuPipelineResult NativeTensorRtPipeline::collect(
    GpuPipelineTicket ticket
) {
    return implementation().collect(ticket);
}

GpuPipelineResult NativeTensorRtPipeline::run(
    const cv::Mat& image
) {
    return implementation().run(image);
}

GpuPipelineResult NativeTensorRtPipeline::run_device(
    DeviceImageView image
) {
    return implementation().run_device(image);
}

std::size_t NativeTensorRtPipeline::pipeline_depth() const noexcept {
    return implementation_ != nullptr
        ? implementation_->pipeline_depth()
        : 0U;
}

std::size_t NativeTensorRtPipeline::in_flight_count() const noexcept {
    return implementation_ != nullptr
        ? implementation_->in_flight_count()
        : 0U;
}

NativeTensorRtPipeline::Impl& NativeTensorRtPipeline::implementation() {
    if (implementation_ == nullptr) {
        throw std::logic_error(
            "NativeTensorRtPipeline is in a moved-from state."
        );
    }

    return *implementation_;
}

const NativeTensorRtPipeline::Impl&
NativeTensorRtPipeline::implementation() const {
    if (implementation_ == nullptr) {
        throw std::logic_error(
            "NativeTensorRtPipeline is in a moved-from state."
        );
    }

    return *implementation_;
}

}  // namespace edge
