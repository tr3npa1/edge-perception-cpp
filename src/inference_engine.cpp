#include "inference_engine.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#if defined(EDGE_WITH_TENSORRT) && EDGE_WITH_TENSORRT
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#endif

namespace edge {
namespace {

constexpr std::size_t kHostAlignment = 64U;
constexpr std::uint32_t kOutputWriteSentinelBits = 0x7FC0D00DU;

/**
 * @brief Return a quiet-NaN payload reserved for output-write verification.
 */
[[nodiscard]] float output_write_sentinel() noexcept {
    float value = 0.0F;
    const std::uint32_t bits = kOutputWriteSentinelBits;

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/**
 * @brief Return true when a float still contains the exact write sentinel.
 */
[[nodiscard]] bool is_output_write_sentinel(float value) noexcept {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits == kOutputWriteSentinelBits;
}

/**
 * @brief Convert a filesystem path to UTF-8 for diagnostics and provider
 *        option strings.
 */
[[nodiscard]] std::string path_to_utf8(
    const std::filesystem::path& path
) {
#if defined(__cpp_lib_char8_t)
    const std::u8string utf8 = path.u8string();

    return std::string(
        reinterpret_cast<const char*>(utf8.data()),
        utf8.size()
    );
#else
    return path.u8string();
#endif
}

/**
 * @brief Return a lower-case ASCII copy of a string.
 */
[[nodiscard]] std::string ascii_lowercase(
    std::string value
) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        }
    );

    return value;
}

/**
 * @brief Return true when a string is non-empty and contains no embedded NUL.
 */
[[nodiscard]] bool is_valid_tensor_name(
    const std::string& name
) noexcept {
    return !name.empty()
        && name.find('\0') == std::string::npos;
}

/**
 * @brief Check whether an inference-backend enum value is supported.
 */
[[nodiscard]] bool is_supported_backend(
    InferenceBackend backend
) noexcept {
    switch (backend) {
        case InferenceBackend::OrtCpu:
        case InferenceBackend::OrtCuda:
        case InferenceBackend::OrtTensorRt:
        case InferenceBackend::NativeTensorRt:
            return true;
    }

    return false;
}

/**
 * @brief Check whether an artifact-precision enum value is supported.
 */
[[nodiscard]] bool is_supported_precision(
    ArtifactPrecision precision
) noexcept {
    switch (precision) {
        case ArtifactPrecision::Float32:
        case ArtifactPrecision::Float16:
        case ArtifactPrecision::Int8:
            return true;
    }

    return false;
}

/**
 * @brief Check whether an ORT graph-optimization enum value is supported.
 */
[[nodiscard]] bool is_supported_graph_optimization(
    OrtGraphOptimization optimization
) noexcept {
    switch (optimization) {
        case OrtGraphOptimization::Disabled:
        case OrtGraphOptimization::Basic:
        case OrtGraphOptimization::Extended:
        case OrtGraphOptimization::All:
            return true;
    }

    return false;
}

/**
 * @brief Check whether an ORT CUDA arena strategy enum value is supported.
 */
[[nodiscard]] bool is_supported_cuda_arena_strategy(
    OrtCudaArenaExtendStrategy strategy
) noexcept {
    switch (strategy) {
        case OrtCudaArenaExtendStrategy::NextPowerOfTwo:
        case OrtCudaArenaExtendStrategy::SameAsRequested:
            return true;
    }

    return false;
}

/**
 * @brief Check whether an ORT cuDNN search enum value is supported.
 */
[[nodiscard]] bool is_supported_cudnn_search(
    OrtCudnnConvAlgorithmSearch search
) noexcept {
    switch (search) {
        case OrtCudnnConvAlgorithmSearch::Exhaustive:
        case OrtCudnnConvAlgorithmSearch::Heuristic:
        case OrtCudnnConvAlgorithmSearch::Default:
            return true;
    }

    return false;
}

/**
 * @brief Convert the project CUDA arena strategy to an ORT option string.
 */
[[nodiscard]] std::string_view ort_cuda_arena_strategy_name(
    OrtCudaArenaExtendStrategy strategy
) {
    switch (strategy) {
        case OrtCudaArenaExtendStrategy::NextPowerOfTwo:
            return "kNextPowerOfTwo";

        case OrtCudaArenaExtendStrategy::SameAsRequested:
            return "kSameAsRequested";
    }

    throw std::invalid_argument(
        "Unsupported ORT CUDA arena strategy."
    );
}

/**
 * @brief Convert the project cuDNN search policy to an ORT option string.
 */
[[nodiscard]] std::string_view ort_cudnn_search_name(
    OrtCudnnConvAlgorithmSearch search
) {
    switch (search) {
        case OrtCudnnConvAlgorithmSearch::Exhaustive:
            return "EXHAUSTIVE";

        case OrtCudnnConvAlgorithmSearch::Heuristic:
            return "HEURISTIC";

        case OrtCudnnConvAlgorithmSearch::Default:
            return "DEFAULT";
    }

    throw std::invalid_argument(
        "Unsupported ORT cuDNN convolution search policy."
    );
}

/**
 * @brief Convert the project graph-optimization enum to ONNX Runtime.
 */
[[nodiscard]] GraphOptimizationLevel to_ort_graph_optimization(
    OrtGraphOptimization optimization
) {
    switch (optimization) {
        case OrtGraphOptimization::Disabled:
            return ORT_DISABLE_ALL;

        case OrtGraphOptimization::Basic:
            return ORT_ENABLE_BASIC;

        case OrtGraphOptimization::Extended:
            return ORT_ENABLE_EXTENDED;

        case OrtGraphOptimization::All:
            return ORT_ENABLE_ALL;
    }

    throw std::invalid_argument(
        "Unsupported ONNX Runtime graph-optimization level."
    );
}

/**
 * @brief Return the expected model-input type for an ONNX artifact variant.
 */
[[nodiscard]] TensorElementType expected_onnx_input_type(
    ArtifactPrecision precision
) {
    switch (precision) {
        case ArtifactPrecision::Float32:
        case ArtifactPrecision::Int8:
            return TensorElementType::Float32;

        case ArtifactPrecision::Float16:
            return TensorElementType::Float16;
    }

    throw std::invalid_argument(
        "Unsupported artifact precision."
    );
}

/**
 * @brief Convert a supported ONNX tensor type to the project type enum.
 */
[[nodiscard]] TensorElementType tensor_type_from_onnx(
    ONNXTensorElementDataType type,
    std::string_view tensor_role
) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return TensorElementType::Float32;

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return TensorElementType::Float16;

        default:
            throw std::runtime_error(
                std::string{tensor_role}
                + " uses an unsupported ONNX element type: "
                + std::to_string(static_cast<int>(type))
                + "."
            );
    }
}

/**
 * @brief Convert a project tensor type to the matching ONNX enum.
 */
[[nodiscard]] ONNXTensorElementDataType tensor_type_to_onnx(
    TensorElementType type
) {
    switch (type) {
        case TensorElementType::Float32:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

        case TensorElementType::Float16:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }

    throw std::runtime_error(
        "Unsupported project tensor element type."
    );
}

/**
 * @brief Return true when a provider name is present in a provider list.
 */
[[nodiscard]] bool contains_provider(
    const std::vector<std::string>& providers,
    std::string_view required_provider
) {
    return std::any_of(
        providers.begin(),
        providers.end(),
        [required_provider](const std::string& provider) {
            return provider == required_provider;
        }
    );
}

/**
 * @brief Reject a requested ONNX Runtime provider that is unavailable.
 */
void require_ort_provider(
    const std::vector<std::string>& available_providers,
    std::string_view required_provider
) {
    if (contains_provider(available_providers, required_provider)) {
        return;
    }

    std::string available_text{};

    for (const std::string& provider : available_providers) {
        if (!available_text.empty()) {
            available_text += ", ";
        }

        available_text += provider;
    }

    if (available_text.empty()) {
        available_text = "none";
    }

    throw std::runtime_error(
        "Required ONNX Runtime provider '"
        + std::string{required_provider}
        + "' is unavailable. Available providers: "
        + available_text
        + "."
    );
}

/**
 * @brief Create a directory tree when a configured runtime feature needs it.
 */
void ensure_directory_exists(
    const std::filesystem::path& directory,
    std::string_view purpose
) {
    std::error_code error{};

    if (std::filesystem::exists(directory, error)) {
        if (error) {
            throw std::runtime_error(
                "Failed to inspect "
                + std::string{purpose}
                + " directory '"
                + path_to_utf8(directory)
                + "': "
                + error.message()
                + "."
            );
        }

        if (!std::filesystem::is_directory(directory, error) || error) {
            throw std::runtime_error(
                "Configured "
                + std::string{purpose}
                + " path is not a directory: "
                + path_to_utf8(directory)
                + "."
            );
        }

        return;
    }

    if (
        !std::filesystem::create_directories(directory, error)
        && error
    ) {
        throw std::runtime_error(
            "Failed to create "
            + std::string{purpose}
            + " directory '"
            + path_to_utf8(directory)
            + "': "
            + error.message()
            + "."
        );
    }
}

/**
 * @brief Validate that an artifact path exists and names a regular file.
 */
void validate_artifact_file(
    const std::filesystem::path& artifact_path
) {
    std::error_code error{};

    const bool exists = std::filesystem::exists(
        artifact_path,
        error
    );

    if (error) {
        throw std::runtime_error(
            "Failed to inspect model artifact '"
            + path_to_utf8(artifact_path)
            + "': "
            + error.message()
            + "."
        );
    }

    if (!exists) {
        throw std::runtime_error(
            "Model artifact does not exist: "
            + path_to_utf8(artifact_path)
            + "."
        );
    }

    const bool is_regular = std::filesystem::is_regular_file(
        artifact_path,
        error
    );

    if (error || !is_regular) {
        throw std::runtime_error(
            "Model artifact is not a regular file: "
            + path_to_utf8(artifact_path)
            + "."
        );
    }

    const std::uintmax_t file_size = std::filesystem::file_size(
        artifact_path,
        error
    );

    if (error || file_size == 0U) {
        throw std::runtime_error(
            "Model artifact is empty or unreadable: "
            + path_to_utf8(artifact_path)
            + "."
        );
    }
}

#if defined(EDGE_WITH_TENSORRT) && EDGE_WITH_TENSORRT

/**
 * @brief Read one binary artifact through std::filesystem-compatible streams.
 */
[[nodiscard]] std::vector<std::uint8_t> read_binary_file(
    const std::filesystem::path& file_path
) {
    std::ifstream input(
        file_path,
        std::ios::binary | std::ios::ate
    );

    if (!input.is_open()) {
        throw std::runtime_error(
            "Failed to open binary artifact: "
            + path_to_utf8(file_path)
            + "."
        );
    }

    const std::streampos end_position = input.tellg();

    if (end_position == std::streampos{-1}) {
        throw std::runtime_error(
            "Failed to determine artifact size: "
            + path_to_utf8(file_path)
            + "."
        );
    }

    const std::streamoff signed_size =
        end_position - std::streampos{0};

    if (signed_size <= 0) {
        throw std::runtime_error(
            "Binary artifact is empty: "
            + path_to_utf8(file_path)
            + "."
        );
    }

    const auto unsigned_size = static_cast<std::uintmax_t>(
        signed_size
    );

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
        throw std::runtime_error(
            "Binary artifact is too large to read safely: "
            + path_to_utf8(file_path)
            + "."
        );
    }

    std::vector<std::uint8_t> bytes(
        static_cast<std::size_t>(unsigned_size)
    );

    input.seekg(0, std::ios::beg);

    if (!input.good()) {
        throw std::runtime_error(
            "Failed to seek within binary artifact: "
            + path_to_utf8(file_path)
            + "."
        );
    }

    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size())
    );

    if (
        !input
        || input.gcount()
            != static_cast<std::streamsize>(bytes.size())
    ) {
        throw std::runtime_error(
            "Failed to read the complete binary artifact: "
            + path_to_utf8(file_path)
            + "."
        );
    }

    return bytes;
}

#endif

/**
 * @brief Own a naturally reusable, cache-line-aligned CPU buffer.
 */
class AlignedHostBuffer final {
public:
    AlignedHostBuffer() = default;

    explicit AlignedHostBuffer(
        std::size_t byte_count
    ) {
        allocate(byte_count);
    }

    ~AlignedHostBuffer() {
        release();
    }

    AlignedHostBuffer(const AlignedHostBuffer&) = delete;
    AlignedHostBuffer& operator=(const AlignedHostBuffer&) = delete;

    AlignedHostBuffer(
        AlignedHostBuffer&& other
    ) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          byte_count_(std::exchange(other.byte_count_, 0U)) {}

    AlignedHostBuffer& operator=(
        AlignedHostBuffer&& other
    ) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            byte_count_ = std::exchange(other.byte_count_, 0U);
        }

        return *this;
    }

    void allocate(
        std::size_t byte_count
    ) {
        if (byte_count == 0U) {
            throw std::invalid_argument(
                "Cannot allocate a zero-byte host buffer."
            );
        }

        release();

        data_ = ::operator new(
            byte_count,
            std::align_val_t{kHostAlignment}
        );

        byte_count_ = byte_count;
        std::memset(data_, 0, byte_count_);
    }

    [[nodiscard]] void* data() noexcept {
        return data_;
    }

    [[nodiscard]] const void* data() const noexcept {
        return data_;
    }

    [[nodiscard]] std::size_t byte_count() const noexcept {
        return byte_count_;
    }

private:
    void release() noexcept {
        if (data_ != nullptr) {
            ::operator delete(
                data_,
                std::align_val_t{kHostAlignment}
            );
        }

        data_ = nullptr;
        byte_count_ = 0U;
    }

    void* data_ = nullptr;
    std::size_t byte_count_ = 0U;
};

/**
 * @brief Return the process-wide ONNX Runtime environment.
 */
[[nodiscard]] Ort::Env& ort_environment() {
    static Ort::Env environment{
        ORT_LOGGING_LEVEL_WARNING,
        "edge-perception-cpp"
    };

    return environment;
}

/**
 * @brief Convert an ORT tensor shape to a fixed array and validate rank.
 */
template <std::size_t Rank>
[[nodiscard]] std::array<std::int64_t, Rank> fixed_shape_from_ort(
    const std::vector<std::int64_t>& shape,
    std::string_view tensor_role
) {
    if (shape.size() != Rank) {
        throw std::runtime_error(
            std::string{tensor_role}
            + " has rank "
            + std::to_string(shape.size())
            + ", but rank "
            + std::to_string(Rank)
            + " is required."
        );
    }

    std::array<std::int64_t, Rank> result{};
    std::copy(shape.begin(), shape.end(), result.begin());
    return result;
}

/**
 * @brief Require exact equality between discovered and expected dimensions.
 */
template <std::size_t Rank>
void require_exact_shape(
    const std::array<std::int64_t, Rank>& actual,
    const std::array<std::int64_t, Rank>& expected,
    std::string_view tensor_role
) {
    if (actual == expected) {
        return;
    }

    std::string actual_text{"["};
    std::string expected_text{"["};

    for (std::size_t index = 0U; index < Rank; ++index) {
        if (index != 0U) {
            actual_text += ", ";
            expected_text += ", ";
        }

        actual_text += std::to_string(actual[index]);
        expected_text += std::to_string(expected[index]);
    }

    actual_text += "]";
    expected_text += "]";

    throw std::runtime_error(
        std::string{tensor_role}
        + " shape mismatch. Expected "
        + expected_text
        + ", received "
        + actual_text
        + "."
    );
}

/**
 * @brief Translate an ONNX Runtime exception into project-level context.
 */
[[noreturn]] void throw_ort_failure(
    std::string_view operation,
    const Ort::Exception& exception
) {
    throw std::runtime_error(
        std::string{operation}
        + " failed in ONNX Runtime: "
        + exception.what()
    );
}

#if defined(EDGE_WITH_TENSORRT) && EDGE_WITH_TENSORRT

/**
 * @brief Return the version of the TensorRT runtime library loaded at runtime.
 *
 * This can differ from the compile-time header version when Windows resolves a
 * different nvinfer DLL. Reporting the runtime value makes dependency problems
 * visible in benchmark metadata instead of silently presenting header macros.
 */
[[nodiscard]] std::string tensorrt_runtime_version() {
    const std::int32_t encoded_version = getInferLibVersion();

    if (encoded_version <= 0) {
        return "unknown";
    }

    const std::int32_t major = encoded_version / 10000;
    const std::int32_t minor = (encoded_version / 100) % 100;
    const std::int32_t patch = encoded_version % 100;

    return std::to_string(major)
        + "."
        + std::to_string(minor)
        + "."
        + std::to_string(patch);
}

/**
 * @brief Throw a descriptive exception for a failed CUDA Runtime call.
 */
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

/**
 * @brief Own the non-default CUDA stream used by native TensorRT.
 */
class CudaStream final {
public:
    CudaStream() = default;

    explicit CudaStream(
        int device_id
    ) {
        create(device_id);
    }

    ~CudaStream() {
        release();
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept
        : stream_(std::exchange(other.stream_, nullptr)) {}

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            release();
            stream_ = std::exchange(other.stream_, nullptr);
        }

        return *this;
    }

    void create(
        int device_id
    ) {
        release();

        check_cuda(
            cudaSetDevice(device_id),
            "cudaSetDevice"
        );

        check_cuda(
            cudaStreamCreateWithFlags(
                &stream_,
                cudaStreamNonBlocking
            ),
            "cudaStreamCreateWithFlags"
        );
    }

    [[nodiscard]] cudaStream_t get() const noexcept {
        return stream_;
    }

    void synchronize() const {
        check_cuda(
            cudaStreamSynchronize(stream_),
            "cudaStreamSynchronize"
        );
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

/**
 * @brief Own page-locked host storage for asynchronous PCIe transfers.
 */
class PinnedHostBuffer final {
public:
    PinnedHostBuffer() = default;

    explicit PinnedHostBuffer(
        std::size_t byte_count
    ) {
        allocate(byte_count);
    }

    ~PinnedHostBuffer() {
        release();
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          byte_count_(std::exchange(other.byte_count_, 0U)) {}

    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            byte_count_ = std::exchange(other.byte_count_, 0U);
        }

        return *this;
    }

    void allocate(
        std::size_t byte_count
    ) {
        if (byte_count == 0U) {
            throw std::invalid_argument(
                "Cannot allocate a zero-byte pinned host buffer."
            );
        }

        release();

        check_cuda(
            cudaHostAlloc(
                &data_,
                byte_count,
                cudaHostAllocDefault
            ),
            "cudaHostAlloc"
        );

        byte_count_ = byte_count;
        std::memset(data_, 0, byte_count_);
    }

    [[nodiscard]] void* data() noexcept {
        return data_;
    }

    [[nodiscard]] const void* data() const noexcept {
        return data_;
    }

    [[nodiscard]] std::size_t byte_count() const noexcept {
        return byte_count_;
    }

private:
    void release() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFreeHost(data_));
        }

        data_ = nullptr;
        byte_count_ = 0U;
    }

    void* data_ = nullptr;
    std::size_t byte_count_ = 0U;
};

/**
 * @brief Own a persistent CUDA device allocation.
 */
class DeviceBuffer final {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(
        std::size_t byte_count
    ) {
        allocate(byte_count);
    }

    ~DeviceBuffer() {
        release();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          byte_count_(std::exchange(other.byte_count_, 0U)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            byte_count_ = std::exchange(other.byte_count_, 0U);
        }

        return *this;
    }

    void allocate(
        std::size_t byte_count
    ) {
        if (byte_count == 0U) {
            throw std::invalid_argument(
                "Cannot allocate a zero-byte CUDA device buffer."
            );
        }

        release();

        check_cuda(
            cudaMalloc(&data_, byte_count),
            "cudaMalloc"
        );

        byte_count_ = byte_count;
    }

    [[nodiscard]] void* data() noexcept {
        return data_;
    }

    [[nodiscard]] const void* data() const noexcept {
        return data_;
    }

    [[nodiscard]] std::size_t byte_count() const noexcept {
        return byte_count_;
    }

private:
    void release() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
        }

        data_ = nullptr;
        byte_count_ = 0U;
    }

    void* data_ = nullptr;
    std::size_t byte_count_ = 0U;
};

/**
 * @brief Own one captured CUDA graph and its executable instance.
 */
class CudaGraph final {
public:
    CudaGraph() = default;

    ~CudaGraph() {
        release();
    }

    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
    CudaGraph(CudaGraph&&) = delete;
    CudaGraph& operator=(CudaGraph&&) = delete;

    [[nodiscard]] bool ready() const noexcept {
        return executable_ != nullptr;
    }

    void capture_begin(
        cudaStream_t stream
    ) {
        release();

        check_cuda(
            cudaStreamBeginCapture(
                stream,
                cudaStreamCaptureModeGlobal
            ),
            "cudaStreamBeginCapture"
        );
    }

    void capture_end_and_instantiate(
        cudaStream_t stream
    ) {
        check_cuda(
            cudaStreamEndCapture(stream, &graph_),
            "cudaStreamEndCapture"
        );

        check_cuda(
            cudaGraphInstantiate(
                &executable_,
                graph_,
                0U
            ),
            "cudaGraphInstantiate"
        );
    }

    /**
     * @brief End and discard a failed stream capture without throwing.
     *
     * CUDA requires a capturing stream to be ended even when an operation
     * inside the capture path fails. This method restores the stream to an
     * ordinary execution state and destroys any partial graph returned by the
     * runtime.
     */
    void abort_capture(
        cudaStream_t stream
    ) noexcept {
        cudaGraph_t abandoned_graph = nullptr;

        static_cast<void>(
            cudaStreamEndCapture(stream, &abandoned_graph)
        );

        if (abandoned_graph != nullptr) {
            static_cast<void>(
                cudaGraphDestroy(abandoned_graph)
            );
        }

        // Clear a possible sticky error from an invalidated capture before the
        // stream is reused by a later ordinary inference call.
        static_cast<void>(cudaGetLastError());
        release();
    }

    void launch(
        cudaStream_t stream
    ) const {
        if (executable_ == nullptr) {
            throw std::logic_error(
                "CUDA graph launch requested before graph instantiation."
            );
        }

        check_cuda(
            cudaGraphLaunch(executable_, stream),
            "cudaGraphLaunch"
        );
    }

private:
    void release() noexcept {
        if (executable_ != nullptr) {
            static_cast<void>(cudaGraphExecDestroy(executable_));
        }

        if (graph_ != nullptr) {
            static_cast<void>(cudaGraphDestroy(graph_));
        }

        executable_ = nullptr;
        graph_ = nullptr;
    }

    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
};

#endif

/**
 * @brief Common runtime interface hidden behind InferenceEngine::Impl.
 */
class BackendRunner {
public:
    virtual ~BackendRunner() = default;

    BackendRunner(const BackendRunner&) = delete;
    BackendRunner& operator=(const BackendRunner&) = delete;
    BackendRunner(BackendRunner&&) = delete;
    BackendRunner& operator=(BackendRunner&&) = delete;

    [[nodiscard]] virtual MutableTensorView input_tensor() = 0;
    [[nodiscard]] virtual DetectionTensorView run() = 0;
    [[nodiscard]] virtual DetectionTensorView output_tensor() const = 0;

    [[nodiscard]] virtual const InferenceRuntimeInfo& runtime_info()
        const noexcept = 0;

protected:
    BackendRunner() = default;
};

/**
 * @brief Execute ONNX models through CPU, CUDA, or TensorRT EP sessions.
 */
class OrtBackendRunner final : public BackendRunner {
public:
    explicit OrtBackendRunner(
    const InferenceEngineConfig& config
)
    : config_(config),
      run_options_{} {
    try {
        // TensorRT EP uses ONNX Runtime's process-wide logger while its
        // execution provider is being configured. Ensure the Ort::Env exists
        // before appending any execution provider.
        Ort::Env& environment = ort_environment();

        configure_session_options();
        configure_execution_providers();

        session_ = Ort::Session{
            environment,
            config_.artifact_path.c_str(),
            session_options_
        };

        inspect_and_validate_model();

        // ORT_TRT_CUDA_GRAPH_DEVICE_IO_FIX
        // Graph ID 0 owns one persistent device input/output pair.
        if (
            config_.backend == InferenceBackend::OrtTensorRt
            && config_.ort_tensorrt.enable_cuda_graph
        ) {
            run_options_.AddConfigEntry("gpu_graph_id", "0");
        }
        allocate_tensors();
        populate_runtime_info();
    } catch (const Ort::Exception& exception) {
        throw_ort_failure(
            "ONNX Runtime engine initialization",
            exception
        );
    }
}

    [[nodiscard]] MutableTensorView input_tensor() override {
        if (runtime_info_.model.input_element_type
            == TensorElementType::Float32) {
            return MutableTensorView::float32(
                static_cast<float*>(host_input_data_),
                runtime_info_.model.input_element_count
            );
        }

        return MutableTensorView::float16(
            static_cast<std::uint16_t*>(host_input_data_),
            runtime_info_.model.input_element_count
        );
    }

    [[nodiscard]] DetectionTensorView run() override {
        try {
            if (uses_device_bound_io_) {

                copy_host_input_to_device();

                session_.Run(
                    run_options_,
                    io_binding_
                );

                io_binding_.SynchronizeOutputs();


                copy_device_output_to_host();
            } else {
                const char* const input_names[] = {
                    runtime_info_.model.input_name.c_str()
                };

                const char* const output_names[] = {
                    runtime_info_.model.output_name.c_str()
                };

                session_.Run(
                    run_options_,
                    input_names,
                    &input_value_,
                    1U,
                    output_names,
                    &output_value_,
                    1U
                );
            }
        } catch (const Ort::Exception& exception) {
            throw_ort_failure(
                "ONNX Runtime inference",
                exception
            );
        }

        has_output_ = true;
        return output_tensor();
    }

    [[nodiscard]] DetectionTensorView output_tensor() const override {
        if (!has_output_) {
            throw std::logic_error(
                "ONNX Runtime output requested before a successful run."
            );
        }

        return DetectionTensorView::contiguous(
            host_output_data_,
            kModelMaximumDetections
        );
    }

    [[nodiscard]] const InferenceRuntimeInfo& runtime_info()
        const noexcept override {
        return runtime_info_;
    }

private:
    void configure_session_options() {
        session_options_.SetGraphOptimizationLevel(
            to_ort_graph_optimization(
                config_.ort.graph_optimization
            )
        );

        if (config_.ort.intra_op_thread_count > 0) {
            session_options_.SetIntraOpNumThreads(
                config_.ort.intra_op_thread_count
            );
        }

        if (config_.ort.inter_op_thread_count > 0) {
            session_options_.SetInterOpNumThreads(
                config_.ort.inter_op_thread_count
            );
        }

        if (config_.ort.enable_cpu_memory_arena) {
            session_options_.EnableCpuMemArena();
        } else {
            session_options_.DisableCpuMemArena();
        }

        if (config_.ort.enable_memory_pattern) {
            session_options_.EnableMemPattern();
        } else {
            session_options_.DisableMemPattern();
        }

        if (config_.ort.enable_profiling) {
            const std::filesystem::path parent =
                config_.ort.profiling_output.parent_path();

            if (!parent.empty()) {
                ensure_directory_exists(
                    parent,
                    "ONNX Runtime profiling"
                );
            }

            session_options_.EnableProfiling(
                config_.ort.profiling_output.c_str()
            );
        }
    }

    void configure_execution_providers() {
        const std::vector<std::string> available_providers =
            Ort::GetAvailableProviders();

        switch (config_.backend) {
            case InferenceBackend::OrtCpu:
                require_ort_provider(
                    available_providers,
                    "CPUExecutionProvider"
                );
                return;

            case InferenceBackend::OrtCuda:
                require_ort_provider(
                    available_providers,
                    "CUDAExecutionProvider"
                );
                append_cuda_provider();
                return;

            case InferenceBackend::OrtTensorRt:
                require_ort_provider(
                    available_providers,
                    "TensorrtExecutionProvider"
                );
                append_tensorrt_provider();

                if (config_.ort_tensorrt.enable_cuda_fallback) {
                    require_ort_provider(
                        available_providers,
                        "CUDAExecutionProvider"
                    );
                    append_cuda_provider();
                }

                return;

            case InferenceBackend::NativeTensorRt:
                break;
        }

        throw std::logic_error(
            "Native TensorRT was routed to the ONNX Runtime runner."
        );
    }

    void append_cuda_provider() {
        Ort::CUDAProviderOptions options{};

        std::unordered_map<std::string, std::string> values{
            {"device_id", std::to_string(config_.gpu.device_id)},
            {"do_copy_in_default_stream", "1"},
            {
                "arena_extend_strategy",
                std::string{
                    ort_cuda_arena_strategy_name(
                        config_.gpu.arena_extend_strategy
                    )
                }
            },
            {
                "cudnn_conv_algo_search",
                std::string{
                    ort_cudnn_search_name(
                        config_.gpu.cudnn_conv_algorithm_search
                    )
                }
            },
            {
                "cudnn_conv_use_max_workspace",
                config_.gpu.cudnn_conv_use_max_workspace ? "1" : "0"
            },
            {
                "prefer_nhwc",
                config_.gpu.prefer_nhwc ? "1" : "0"
            },
            {
                "use_tf32",
                config_.gpu.use_tf32 ? "1" : "0"
            },
            {
                "use_ep_level_unified_stream",
                config_.gpu.use_ep_level_unified_stream ? "1" : "0"
            }
        };

        if (config_.gpu.gpu_memory_limit_bytes != 0U) {
            values.emplace(
                "gpu_mem_limit",
                std::to_string(
                    config_.gpu.gpu_memory_limit_bytes
                )
            );
        }

        options.Update(values);

        session_options_.AppendExecutionProvider_CUDA_V2(
            *options
        );
    }

    void append_tensorrt_provider() {
        const OrtTensorRtConfig& trt = config_.ort_tensorrt;

        if (trt.enable_engine_cache) {
            ensure_directory_exists(
                trt.engine_cache_directory,
                "TensorRT EP engine cache"
            );
        }

        if (trt.enable_timing_cache) {
            ensure_directory_exists(
                trt.timing_cache_directory,
                "TensorRT EP timing cache"
            );
        }

        Ort::TensorRTProviderOptions options{};

        std::unordered_map<std::string, std::string> values{
            {"device_id", std::to_string(config_.gpu.device_id)},
            {
                "trt_max_partition_iterations",
                std::to_string(trt.maximum_partition_iterations)
            },
            {
                "trt_min_subgraph_size",
                std::to_string(trt.minimum_subgraph_size)
            },
            {
                "trt_fp16_enable",
                config_.artifact_precision
                        == ArtifactPrecision::Float16
                    ? "1"
                    : "0"
            },
            {
                "trt_int8_enable",
                config_.artifact_precision
                        == ArtifactPrecision::Int8
                    ? "1"
                    : "0"
            },
            {
                "trt_dump_subgraphs",
                trt.dump_partitioned_subgraphs ? "1" : "0"
            },
            {
                "trt_engine_cache_enable",
                trt.enable_engine_cache ? "1" : "0"
            },
            {
                "trt_timing_cache_enable",
                trt.enable_timing_cache ? "1" : "0"
            },
            {
                "trt_context_memory_sharing_enable",
                trt.enable_context_memory_sharing ? "1" : "0"
            },
            {
                "trt_cuda_graph_enable",
                trt.enable_cuda_graph ? "1" : "0"
            },
            {
                "trt_builder_optimization_level",
                std::to_string(trt.builder_optimization_level)
            },
            {
                "trt_auxiliary_streams",
                std::to_string(trt.auxiliary_streams)
            }
        };

        if (trt.maximum_workspace_bytes != 0U) {
            values.emplace(
                "trt_max_workspace_size",
                std::to_string(trt.maximum_workspace_bytes)
            );
        }

        if (trt.enable_engine_cache) {
            values.emplace(
                "trt_engine_cache_path",
                path_to_utf8(trt.engine_cache_directory)
            );
        }

        if (trt.enable_timing_cache) {
            values.emplace(
                "trt_timing_cache_path",
                path_to_utf8(trt.timing_cache_directory)
            );
        }

        options.Update(values);

        session_options_.AppendExecutionProvider_TensorRT_V2(
            *options
        );
    }

    void inspect_and_validate_model() {
        if (session_.GetInputCount() != 1U) {
            throw std::runtime_error(
                "The project requires exactly one model input."
            );
        }

        if (session_.GetOutputCount() != 1U) {
            throw std::runtime_error(
                "The project requires exactly one model output."
            );
        }

        Ort::AllocatorWithDefaultOptions allocator{};

        const Ort::AllocatedStringPtr input_name =
            session_.GetInputNameAllocated(0U, allocator);

        const Ort::AllocatedStringPtr output_name =
            session_.GetOutputNameAllocated(0U, allocator);

        if (input_name == nullptr || output_name == nullptr) {
            throw std::runtime_error(
                "ONNX Runtime returned a null tensor name."
            );
        }

        runtime_info_.model.input_name = input_name.get();
        runtime_info_.model.output_name = output_name.get();

        if (
            runtime_info_.model.input_name
                != config_.expected_input_name
            || runtime_info_.model.output_name
                != config_.expected_output_name
        ) {
            throw std::runtime_error(
                "Model tensor-name mismatch. Expected input '"
                + config_.expected_input_name
                + "' and output '"
                + config_.expected_output_name
                + "', received input '"
                + runtime_info_.model.input_name
                + "' and output '"
                + runtime_info_.model.output_name
                + "'."
            );
        }

        const Ort::TypeInfo input_type_info =
            session_.GetInputTypeInfo(0U);

        const Ort::TypeInfo output_type_info =
            session_.GetOutputTypeInfo(0U);

        if (
            input_type_info.GetONNXType() != ONNX_TYPE_TENSOR
            || output_type_info.GetONNXType() != ONNX_TYPE_TENSOR
        ) {
            throw std::runtime_error(
                "Both model input and output must be dense tensors."
            );
        }

        const Ort::ConstTensorTypeAndShapeInfo input_tensor_info =
            input_type_info.GetTensorTypeAndShapeInfo();

        const Ort::ConstTensorTypeAndShapeInfo output_tensor_info =
            output_type_info.GetTensorTypeAndShapeInfo();

        runtime_info_.model.input_shape = fixed_shape_from_ort<4U>(
            input_tensor_info.GetShape(),
            "Model input"
        );

        runtime_info_.model.output_shape = fixed_shape_from_ort<3U>(
            output_tensor_info.GetShape(),
            "Model output"
        );

        require_exact_shape(
            runtime_info_.model.input_shape,
            kModelInputShape,
            "Model input"
        );

        require_exact_shape(
            runtime_info_.model.output_shape,
            kModelOutputShape,
            "Model output"
        );

        runtime_info_.model.input_element_type = tensor_type_from_onnx(
            input_tensor_info.GetElementType(),
            "Model input"
        );

        runtime_info_.model.backend_output_element_type =
            tensor_type_from_onnx(
                output_tensor_info.GetElementType(),
                "Model output"
            );

        runtime_info_.model.host_output_element_type =
            TensorElementType::Float32;

        if (
            runtime_info_.model.input_element_type
            != expected_onnx_input_type(
                config_.artifact_precision
            )
        ) {
            throw std::runtime_error(
                "ONNX input type does not match the configured artifact "
                "precision."
            );
        }

        if (
            runtime_info_.model.backend_output_element_type
            != TensorElementType::Float32
        ) {
            throw std::runtime_error(
                "Project ONNX artifacts must expose FP32 output0."
            );
        }

        runtime_info_.model.input_element_count =
            kModelInputElementCount;

        runtime_info_.model.input_byte_count =
            kModelInputElementCount
            * tensor_element_size(
                runtime_info_.model.input_element_type
            );

        runtime_info_.model.output_element_count =
            kModelOutputElementCount;

        runtime_info_.model.backend_output_byte_count =
            kModelOutputElementCount * sizeof(float);

        runtime_info_.model.host_output_byte_count =
            kModelOutputElementCount * sizeof(float);
    }

    /**
     * @brief Allocate persistent host tensors for canonical Session::Run.
     *
     * ORT CUDA/TensorRT EP receives CPU Ort::Value objects and performs the
     * required H2D/D2H transfers. Native TensorRT keeps its separate CUDA 12.8
     * pinned-memory implementation.
     */
    void allocate_tensors() {
        allocate_pageable_cpu_tensors();

        uses_device_bound_io_ =
            config_.backend == InferenceBackend::OrtTensorRt
            && config_.ort_tensorrt.enable_cuda_graph;

        if (!uses_device_bound_io_) {
            return;
        }

        allocate_cuda_graph_device_tensors();

        io_binding_ = Ort::IoBinding{session_};

        io_binding_.BindInput(
            runtime_info_.model.input_name.c_str(),
            device_input_value_
        );

        io_binding_.BindOutput(
            runtime_info_.model.output_name.c_str(),
            device_output_value_
        );
    }

    /**
     * @brief Reject an output tensor that ORT did not fully overwrite.
     */
    void validate_completed_output() const {
        for (
            std::size_t index = 0U;
            index < runtime_info_.model.output_element_count;
            ++index
        ) {
            if (is_output_write_sentinel(host_output_data_[index])) {
                throw std::runtime_error(
                    "ONNX Runtime did not overwrite the complete output tensor; "
                    "the write sentinel remained at flat index "
                    + std::to_string(index)
                    + "."
                );
            }
        }
    }

    /**
     * @brief Allocate persistent pageable tensors around project-owned memory.
     */
    void allocate_pageable_cpu_tensors() {
        const InferenceModelInfo& model = runtime_info_.model;

        aligned_host_input_.allocate(model.input_byte_count);
        aligned_host_output_.allocate(
            model.host_output_byte_count
        );

        host_input_data_ = aligned_host_input_.data();
        host_output_data_ = static_cast<float*>(
            aligned_host_output_.data()
        );

        cpu_memory_info_ = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

        input_value_ = Ort::Value::CreateTensor(
            cpu_memory_info_,
            host_input_data_,
            model.input_byte_count,
            kModelInputShape.data(),
            kModelInputShape.size(),
            tensor_type_to_onnx(model.input_element_type)
        );

        output_value_ = Ort::Value::CreateTensor(
            cpu_memory_info_,
            host_output_data_,
            model.host_output_byte_count,
            kModelOutputShape.data(),
            kModelOutputShape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        );

        host_buffers_are_pinned_ = false;
    }

    void allocate_cuda_graph_device_tensors() {
        const InferenceModelInfo& model = runtime_info_.model;

        // Allocate through this exact ORT Session. No CUDA 12.8 pointer or
        // stream enters the ORT CUDA/TensorRT execution path.
        cuda_memory_info_ = Ort::MemoryInfo{
            "Cuda",
            OrtArenaAllocator,
            config_.gpu.device_id,
            OrtMemTypeDefault
        };

        cuda_allocator_ = Ort::Allocator{
            session_,
            cuda_memory_info_
        };

        device_input_value_ = Ort::Value::CreateTensor(
            cuda_allocator_,
            kModelInputShape.data(),
            kModelInputShape.size(),
            tensor_type_to_onnx(
                model.input_element_type
            )
        );

        device_output_value_ = Ort::Value::CreateTensor(
            cuda_allocator_,
            kModelOutputShape.data(),
            kModelOutputShape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        );
    }

    // ORT_TRT_CUDA13_RUNTIME_COPY_FIX
#if defined(_WIN32)
    class Cuda13Runtime final {
    public:
        using CudaError = int;

        using CudaSetDeviceFunction =
            CudaError (*)(int);

        using CudaMemcpyFunction =
            CudaError (*)(
                void*,
                const void*,
                std::size_t,
                int
            );

        using CudaGetErrorStringFunction =
            const char* (*)(CudaError);

        Cuda13Runtime() {
            module_ = GetModuleHandleW(
                L"cudart64_13.dll"
            );

            if (module_ == nullptr) {
                module_ = LoadLibraryW(
                    L"cudart64_13.dll"
                );
            }

            if (module_ == nullptr) {
                throw std::runtime_error(
                    "CUDA 13 runtime DLL cudart64_13.dll "
                    "could not be loaded."
                );
            }

            cuda_set_device_ =
                reinterpret_cast<CudaSetDeviceFunction>(
                    GetProcAddress(
                        module_,
                        "cudaSetDevice"
                    )
                );

            cuda_memcpy_ =
                reinterpret_cast<CudaMemcpyFunction>(
                    GetProcAddress(
                        module_,
                        "cudaMemcpy"
                    )
                );

            cuda_get_error_string_ =
                reinterpret_cast<CudaGetErrorStringFunction>(
                    GetProcAddress(
                        module_,
                        "cudaGetErrorString"
                    )
                );

            if (
                cuda_set_device_ == nullptr
                || cuda_memcpy_ == nullptr
                || cuda_get_error_string_ == nullptr
            ) {
                throw std::runtime_error(
                    "CUDA 13 runtime DLL is missing required "
                    "cudaSetDevice/cudaMemcpy/cudaGetErrorString exports."
                );
            }
        }

        void copy(
            int device_id,
            void* destination,
            const void* source,
            std::size_t byte_count,
            int copy_kind,
            const char* operation
        ) const {
            if (
                destination == nullptr
                || source == nullptr
            ) {
                throw std::runtime_error(
                    std::string{operation}
                    + ": null source or destination pointer."
                );
            }

            const CudaError set_device_status =
                cuda_set_device_(device_id);

            check(
                set_device_status,
                "cudaSetDevice"
            );

            const CudaError copy_status =
                cuda_memcpy_(
                    destination,
                    source,
                    byte_count,
                    copy_kind
                );

            check(
                copy_status,
                operation
            );
        }

    private:
        void check(
            CudaError status,
            const char* operation
        ) const {
            constexpr CudaError cuda_success = 0;

            if (status == cuda_success) {
                return;
            }

            const char* const description =
                cuda_get_error_string_(status);

            throw std::runtime_error(
                std::string{operation}
                + " failed with CUDA error "
                + std::to_string(status)
                + ": "
                + (
                    description != nullptr
                        ? description
                        : "unknown CUDA error"
                )
            );
        }

        HMODULE module_ = nullptr;

        CudaSetDeviceFunction cuda_set_device_ =
            nullptr;

        CudaMemcpyFunction cuda_memcpy_ =
            nullptr;

        CudaGetErrorStringFunction
            cuda_get_error_string_ = nullptr;
    };

    static Cuda13Runtime& cuda13_runtime() {
        static Cuda13Runtime runtime{};
        return runtime;
    }
#endif

    void copy_host_input_to_device() {
#if defined(_WIN32)
        // cudaMemcpyHostToDevice
        constexpr int host_to_device = 1;

        void* const device_destination =
            device_input_value_.GetTensorMutableRawData();

        cuda13_runtime().copy(
            config_.gpu.device_id,
            device_destination,
            host_input_data_,
            runtime_info_.model.input_byte_count,
            host_to_device,
            "CUDA 13 host-to-device input copy"
        );
#else
        throw std::runtime_error(
            "ORT TensorRT CUDA-graph device copies "
            "are currently implemented only for Windows."
        );
#endif
    }

    void copy_device_output_to_host() {
#if defined(_WIN32)
        // cudaMemcpyDeviceToHost
        constexpr int device_to_host = 2;

        void* const device_source =
            device_output_value_.GetTensorMutableRawData();

        cuda13_runtime().copy(
            config_.gpu.device_id,
            host_output_data_,
            device_source,
            runtime_info_.model.host_output_byte_count,
            device_to_host,
            "CUDA 13 device-to-host output copy"
        );
#else
        throw std::runtime_error(
            "ORT TensorRT CUDA-graph device copies "
            "are currently implemented only for Windows."
        );
#endif
    }

    void populate_runtime_info() {
        runtime_info_.backend = config_.backend;
        runtime_info_.artifact_precision =
            config_.artifact_precision;

        runtime_info_.runtime_name = "ONNX Runtime";
        runtime_info_.runtime_version = Ort::GetVersionString();

        switch (config_.backend) {
            case InferenceBackend::OrtCpu:
                runtime_info_.active_execution_providers = {
                    "CPUExecutionProvider"
                };
                break;

            case InferenceBackend::OrtCuda:
                runtime_info_.active_execution_providers = {
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider"
                };
                break;

            case InferenceBackend::OrtTensorRt:
                runtime_info_.active_execution_providers = {
                    "TensorrtExecutionProvider"
                };

                if (config_.ort_tensorrt.enable_cuda_fallback) {
                    runtime_info_.active_execution_providers.push_back(
                        "CUDAExecutionProvider"
                    );
                }

                runtime_info_.active_execution_providers.push_back(
                    "CPUExecutionProvider"
                );
                break;

            case InferenceBackend::NativeTensorRt:
                throw std::logic_error(
                    "Invalid ORT backend while populating runtime info."
                );
        }

        runtime_info_.buffers.host_input_bytes =
            runtime_info_.model.input_byte_count;

        runtime_info_.buffers.host_output_bytes =
            runtime_info_.model.host_output_byte_count;

        runtime_info_.buffers.host_input_is_pinned =
            host_buffers_are_pinned_;

        runtime_info_.buffers.host_output_is_pinned =
            host_buffers_are_pinned_;
    }

    InferenceEngineConfig config_{};
    InferenceRuntimeInfo runtime_info_{};

    Ort::SessionOptions session_options_{};
    Ort::Session session_{nullptr};
    Ort::RunOptions run_options_{};

    Ort::MemoryInfo cpu_memory_info_{nullptr};

    // Device allocator and tensors belong to this ORT Session.
    Ort::MemoryInfo cuda_memory_info_{nullptr};
    Ort::Allocator cuda_allocator_{nullptr};

    AlignedHostBuffer aligned_host_input_{};
    AlignedHostBuffer aligned_host_output_{};

    void* host_input_data_ = nullptr;
    float* host_output_data_ = nullptr;

    Ort::Value input_value_{nullptr};
    Ort::Value output_value_{nullptr};

    Ort::Value device_input_value_{nullptr};
    Ort::Value device_output_value_{nullptr};

    // Destroy binding before the tensors and allocator it references.
    Ort::IoBinding io_binding_{nullptr};

    bool host_buffers_are_pinned_ = false;
    bool uses_device_bound_io_ = false;
    bool has_output_ = false;
};

#if defined(EDGE_WITH_TENSORRT) && EDGE_WITH_TENSORRT

/**
 * @brief Process-wide TensorRT logger that reports warnings and failures.
 */
class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(
        Severity severity,
        const char* message
    ) noexcept override {
        if (severity > Severity::kWARNING || message == nullptr) {
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
        std::cerr
            << "[TensorRT "
            << label
            << "] "
            << message
            << '\n';
    }

private:
    std::mutex mutex_{};
};

/**
 * @brief Return the logger whose lifetime covers every TensorRT object.
 */
[[nodiscard]] TensorRtLogger& tensorrt_logger() {
    static TensorRtLogger logger{};
    return logger;
}

/**
 * @brief Register NVIDIA TensorRT plugins once for the process.
 */
void initialize_tensorrt_plugins() {
    static std::once_flag once{};
    static bool initialized = false;

    std::call_once(
        once,
        [] {
            initialized = initLibNvInferPlugins(
                &tensorrt_logger(),
                ""
            );
        }
    );

    if (!initialized) {
        throw std::runtime_error(
            "TensorRT plugin registration failed."
        );
    }
}

/**
 * @brief Convert a supported TensorRT binding type to the project type enum.
 */
[[nodiscard]] TensorElementType tensor_type_from_tensorrt(
    nvinfer1::DataType type,
    std::string_view tensor_role
) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return TensorElementType::Float32;

        case nvinfer1::DataType::kHALF:
            return TensorElementType::Float16;

        default:
            throw std::runtime_error(
                std::string{tensor_role}
                + " uses an unsupported TensorRT element type."
            );
    }
}

/**
 * @brief Return true when a TensorRT shape contains a dynamic dimension.
 */
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

/**
 * @brief Convert a fixed project shape to TensorRT Dims.
 */
template <std::size_t Rank>
[[nodiscard]] nvinfer1::Dims shape_to_tensorrt_dims(
    const std::array<std::int64_t, Rank>& shape
) {
    nvinfer1::Dims dimensions{};
    dimensions.nbDims = static_cast<int>(Rank);

    for (std::size_t index = 0U; index < Rank; ++index) {
        if (
            shape[index] < 0
            || shape[index]
                > static_cast<std::int64_t>(
                    std::numeric_limits<std::int32_t>::max()
                )
        ) {
            throw std::runtime_error(
                "Project tensor shape cannot be represented by TensorRT."
            );
        }

        dimensions.d[index] = static_cast<std::int32_t>(
            shape[index]
        );
    }

    return dimensions;
}

/**
 * @brief Convert TensorRT dimensions to a fixed project array.
 */
template <std::size_t Rank>
[[nodiscard]] std::array<std::int64_t, Rank> fixed_shape_from_tensorrt(
    const nvinfer1::Dims& dimensions,
    std::string_view tensor_role
) {
    if (dimensions.nbDims != static_cast<int>(Rank)) {
        throw std::runtime_error(
            std::string{tensor_role}
            + " has rank "
            + std::to_string(dimensions.nbDims)
            + ", but rank "
            + std::to_string(Rank)
            + " is required."
        );
    }

    std::array<std::int64_t, Rank> shape{};

    for (std::size_t index = 0U; index < Rank; ++index) {
        if (dimensions.d[index] <= 0) {
            throw std::runtime_error(
                std::string{tensor_role}
                + " contains an unresolved or invalid dimension."
            );
        }

        shape[index] = static_cast<std::int64_t>(
            dimensions.d[index]
        );
    }

    return shape;
}

/**
 * @brief Execute a serialized native TensorRT engine with persistent buffers.
 */
class NativeTensorRtRunner final : public BackendRunner {
public:
    explicit NativeTensorRtRunner(
        const InferenceEngineConfig& config
    )
        : config_(config),
          cuda_stream_(config.gpu.device_id) {
        initialize_tensorrt_plugins();

        runtime_.reset(
            nvinfer1::createInferRuntime(tensorrt_logger())
        );

        if (runtime_ == nullptr) {
            throw std::runtime_error(
                "TensorRT failed to create IRuntime."
            );
        }

        const std::vector<std::uint8_t> serialized_engine =
            read_binary_file(config_.artifact_path);

        engine_.reset(
            runtime_->deserializeCudaEngine(
                serialized_engine.data(),
                serialized_engine.size()
            )
        );

        if (engine_ == nullptr) {
            throw std::runtime_error(
                "TensorRT could not deserialize engine: "
                + path_to_utf8(config_.artifact_path)
                + "."
            );
        }

        inspect_tensor_names_and_types();

        // User-managed context memory makes activation/workspace ownership
        // explicit, prevents opaque per-context allocation, improves memory
        // accounting, and guarantees a stable address for CUDA graph capture.
        context_.reset(
            engine_->createExecutionContext(
                nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED
            )
        );

        if (context_ == nullptr) {
            throw std::runtime_error(
                "TensorRT failed to create a user-managed IExecutionContext."
            );
        }

        const std::int64_t signed_workspace_bytes =
            engine_->getDeviceMemorySizeV2();

        if (signed_workspace_bytes < 0) {
            throw std::runtime_error(
                "TensorRT reported a negative execution-context workspace."
            );
        }

        if (
            static_cast<std::uint64_t>(signed_workspace_bytes)
            > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()
            )
        ) {
            throw std::runtime_error(
                "TensorRT execution-context workspace does not fit size_t."
            );
        }

        context_workspace_bytes_ =
            static_cast<std::size_t>(signed_workspace_bytes);

        if (context_workspace_bytes_ != 0U) {
            context_device_memory_.allocate(
                context_workspace_bytes_
            );

            context_->setDeviceMemoryV2(
                context_device_memory_.data(),
                signed_workspace_bytes
            );
        }

        resolve_and_validate_shapes();
        allocate_and_bind_buffers();
        populate_runtime_info();
    }

    [[nodiscard]] MutableTensorView input_tensor() override {
        if (runtime_info_.model.input_element_type
            == TensorElementType::Float32) {
            return MutableTensorView::float32(
                static_cast<float*>(host_input_.data()),
                runtime_info_.model.input_element_count
            );
        }

        return MutableTensorView::float16(
            static_cast<std::uint16_t*>(host_input_.data()),
            runtime_info_.model.input_element_count
        );
    }

    [[nodiscard]] DetectionTensorView run() override {
        if (!config_.gpu.enable_cuda_graph) {
            enqueue_pipeline();
            cuda_stream_.synchronize();
        } else if (!graph_warmup_complete_) {
            // TensorRT may perform lazy resource updates on its first enqueue.
            // Warm once before attempting stream capture.
            enqueue_pipeline();
            cuda_stream_.synchronize();
            graph_warmup_complete_ = true;
        } else if (!cuda_graph_.ready()) {
            cuda_graph_.capture_begin(cuda_stream_.get());

            try {
                enqueue_pipeline();
                cuda_graph_.capture_end_and_instantiate(
                    cuda_stream_.get()
                );
            } catch (...) {
                cuda_graph_.abort_capture(cuda_stream_.get());
                throw;
            }

            cuda_graph_.launch(cuda_stream_.get());
            cuda_stream_.synchronize();
        } else {
            cuda_graph_.launch(cuda_stream_.get());
            cuda_stream_.synchronize();
        }

        publish_host_output();
        has_output_ = true;
        return output_tensor();
    }

    [[nodiscard]] DetectionTensorView output_tensor() const override {
        if (!has_output_) {
            throw std::logic_error(
                "TensorRT output requested before a successful run."
            );
        }

        return DetectionTensorView::contiguous(
            static_cast<const float*>(host_output_float32_.data()),
            kModelMaximumDetections
        );
    }

    [[nodiscard]] const InferenceRuntimeInfo& runtime_info()
        const noexcept override {
        return runtime_info_;
    }

private:
    void inspect_tensor_names_and_types() {
        const int tensor_count = engine_->getNbIOTensors();

        if (tensor_count != 2) {
            throw std::runtime_error(
                "The project requires exactly two TensorRT I/O tensors."
            );
        }

        for (int index = 0; index < tensor_count; ++index) {
            const char* const tensor_name =
                engine_->getIOTensorName(index);

            if (tensor_name == nullptr) {
                throw std::runtime_error(
                    "TensorRT returned a null I/O tensor name."
                );
            }

            const nvinfer1::TensorIOMode mode =
                engine_->getTensorIOMode(tensor_name);

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                if (!input_name_.empty()) {
                    throw std::runtime_error(
                        "TensorRT engine exposes multiple inputs."
                    );
                }

                input_name_ = tensor_name;
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                if (!output_name_.empty()) {
                    throw std::runtime_error(
                        "TensorRT engine exposes multiple outputs."
                    );
                }

                output_name_ = tensor_name;
            } else {
                throw std::runtime_error(
                    "TensorRT engine contains an I/O tensor with no mode."
                );
            }
        }

        if (
            input_name_ != config_.expected_input_name
            || output_name_ != config_.expected_output_name
        ) {
            throw std::runtime_error(
                "TensorRT tensor-name mismatch. Expected input '"
                + config_.expected_input_name
                + "' and output '"
                + config_.expected_output_name
                + "', received input '"
                + input_name_
                + "' and output '"
                + output_name_
                + "'."
            );
        }

        if (
            engine_->getTensorLocation(input_name_.c_str())
                != nvinfer1::TensorLocation::kDEVICE
            || engine_->getTensorLocation(output_name_.c_str())
                != nvinfer1::TensorLocation::kDEVICE
        ) {
            throw std::runtime_error(
                "The native TensorRT path currently requires device-resident "
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
                "The native TensorRT path requires linear input/output "
                "tensor formats."
            );
        }

        runtime_info_.model.input_name = input_name_;
        runtime_info_.model.output_name = output_name_;

        runtime_info_.model.input_element_type =
            tensor_type_from_tensorrt(
                engine_->getTensorDataType(input_name_.c_str()),
                "TensorRT input"
            );

        runtime_info_.model.backend_output_element_type =
            tensor_type_from_tensorrt(
                engine_->getTensorDataType(output_name_.c_str()),
                "TensorRT output"
            );

        runtime_info_.model.host_output_element_type =
            TensorElementType::Float32;
    }

    void resolve_and_validate_shapes() {
        const nvinfer1::Dims engine_input_shape =
            engine_->getTensorShape(input_name_.c_str());

        if (contains_dynamic_dimension(engine_input_shape)) {
            const nvinfer1::Dims fixed_input =
                shape_to_tensorrt_dims(kModelInputShape);

            if (
                !context_->setInputShape(
                    input_name_.c_str(),
                    fixed_input
                )
            ) {
                throw std::runtime_error(
                    "TensorRT rejected the fixed [1,3,640,640] input shape."
                );
            }
        }

        runtime_info_.model.input_shape =
            fixed_shape_from_tensorrt<4U>(
                context_->getTensorShape(input_name_.c_str()),
                "TensorRT input"
            );

        runtime_info_.model.output_shape =
            fixed_shape_from_tensorrt<3U>(
                context_->getTensorShape(output_name_.c_str()),
                "TensorRT output"
            );

        require_exact_shape(
            runtime_info_.model.input_shape,
            kModelInputShape,
            "TensorRT input"
        );

        require_exact_shape(
            runtime_info_.model.output_shape,
            kModelOutputShape,
            "TensorRT output"
        );

        runtime_info_.model.input_element_count =
            kModelInputElementCount;

        runtime_info_.model.input_byte_count =
            kModelInputElementCount
            * tensor_element_size(
                runtime_info_.model.input_element_type
            );

        runtime_info_.model.output_element_count =
            kModelOutputElementCount;

        runtime_info_.model.backend_output_byte_count =
            kModelOutputElementCount
            * tensor_element_size(
                runtime_info_.model.backend_output_element_type
            );

        runtime_info_.model.host_output_byte_count =
            kModelOutputElementCount * sizeof(float);
    }

    void allocate_and_bind_buffers() {
        const InferenceModelInfo& model = runtime_info_.model;

        host_input_.allocate(model.input_byte_count);
        host_output_float32_.allocate(
            model.host_output_byte_count
        );

        if (
            model.backend_output_element_type
            == TensorElementType::Float16
        ) {
            host_output_backend_.allocate(
                model.backend_output_byte_count
            );
        }

        device_input_.allocate(model.input_byte_count);
        device_output_.allocate(model.backend_output_byte_count);

        if (
            !context_->setInputTensorAddress(
                input_name_.c_str(),
                device_input_.data()
            )
        ) {
            throw std::runtime_error(
                "TensorRT rejected the persistent input tensor address."
            );
        }

        if (
            !context_->setOutputTensorAddress(
                output_name_.c_str(),
                device_output_.data()
            )
        ) {
            throw std::runtime_error(
                "TensorRT rejected the persistent output tensor address."
            );
        }
    }

    void enqueue_pipeline() {
        check_cuda(
            cudaMemcpyAsync(
                device_input_.data(),
                host_input_.data(),
                runtime_info_.model.input_byte_count,
                cudaMemcpyHostToDevice,
                cuda_stream_.get()
            ),
            "cudaMemcpyAsync native TensorRT input"
        );

        if (!context_->enqueueV3(cuda_stream_.get())) {
            throw std::runtime_error(
                "TensorRT enqueueV3 returned false."
            );
        }

        void* host_destination =
            runtime_info_.model.backend_output_element_type
                    == TensorElementType::Float32
                ? host_output_float32_.data()
                : host_output_backend_.data();

        check_cuda(
            cudaMemcpyAsync(
                host_destination,
                device_output_.data(),
                runtime_info_.model.backend_output_byte_count,
                cudaMemcpyDeviceToHost,
                cuda_stream_.get()
            ),
            "cudaMemcpyAsync native TensorRT output"
        );
    }

    void publish_host_output() {
        if (
            runtime_info_.model.backend_output_element_type
            == TensorElementType::Float32
        ) {
            return;
        }

        const auto* const source =
            static_cast<const std::uint16_t*>(
                host_output_backend_.data()
            );

        auto* const destination =
            static_cast<float*>(host_output_float32_.data());

        for (
            std::size_t index = 0U;
            index < kModelOutputElementCount;
            ++index
        ) {
            destination[index] =
                Ort::Float16_t::FromBits(source[index]).ToFloat();
        }
    }

    void populate_runtime_info() {
        runtime_info_.backend = InferenceBackend::NativeTensorRt;
        runtime_info_.artifact_precision =
            config_.artifact_precision;

        runtime_info_.runtime_name = "TensorRT";
        runtime_info_.runtime_version =
            tensorrt_runtime_version();

        runtime_info_.active_execution_providers.clear();

        runtime_info_.buffers.host_input_bytes =
            runtime_info_.model.input_byte_count;

        runtime_info_.buffers.host_output_bytes =
            runtime_info_.model.host_output_byte_count;

        if (
            runtime_info_.model.backend_output_element_type
            == TensorElementType::Float16
        ) {
            runtime_info_.buffers.host_output_bytes +=
                runtime_info_.model.backend_output_byte_count;
        }

        runtime_info_.buffers.device_input_bytes =
            runtime_info_.model.input_byte_count;

        runtime_info_.buffers.device_output_bytes =
            runtime_info_.model.backend_output_byte_count;

        runtime_info_.buffers.device_workspace_bytes =
            context_workspace_bytes_;

        runtime_info_.buffers.host_input_is_pinned = true;
        runtime_info_.buffers.host_output_is_pinned = true;
    }

    InferenceEngineConfig config_{};
    InferenceRuntimeInfo runtime_info_{};

    // Destruction order is deliberate: the CUDA stream is declared first so
    // every object that may depend on it is released before the stream itself.
    CudaStream cuda_stream_{};

    PinnedHostBuffer host_input_{};
    PinnedHostBuffer host_output_float32_{};
    PinnedHostBuffer host_output_backend_{};

    DeviceBuffer device_input_{};
    DeviceBuffer device_output_{};

    std::unique_ptr<nvinfer1::IRuntime> runtime_{};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{};

    // Declared before context_ so reverse destruction releases the context
    // before its user-managed activation/workspace allocation.
    DeviceBuffer context_device_memory_{};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{};

    std::size_t context_workspace_bytes_ = 0U;

    std::string input_name_{};
    std::string output_name_{};

    // Declared last among CUDA/TensorRT resources so reverse destruction
    // releases captured nodes before their context, buffers, and stream.
    CudaGraph cuda_graph_{};

    bool graph_warmup_complete_ = false;
    bool has_output_ = false;
};

#endif

}  // namespace

/**
 * @brief Validate ONNX Runtime session-level configuration.
 */
void OrtSessionConfig::validate() const {
    if (!is_supported_graph_optimization(graph_optimization)) {
        throw std::invalid_argument(
            "OrtSessionConfig.graph_optimization is invalid."
        );
    }

    if (intra_op_thread_count < 0) {
        throw std::invalid_argument(
            "OrtSessionConfig.intra_op_thread_count must be non-negative."
        );
    }

    if (inter_op_thread_count < 0) {
        throw std::invalid_argument(
            "OrtSessionConfig.inter_op_thread_count must be non-negative."
        );
    }

    if (enable_profiling && profiling_output.empty()) {
        throw std::invalid_argument(
            "OrtSessionConfig.profiling_output is required when profiling "
            "is enabled."
        );
    }
}

/**
 * @brief Validate CUDA-backed execution configuration.
 */
void GpuExecutionConfig::validate() const {
    if (device_id < 0) {
        throw std::invalid_argument(
            "GpuExecutionConfig.device_id must be non-negative."
        );
    }

    if (!is_supported_cuda_arena_strategy(arena_extend_strategy)) {
        throw std::invalid_argument(
            "GpuExecutionConfig.arena_extend_strategy is invalid."
        );
    }

    if (!is_supported_cudnn_search(cudnn_conv_algorithm_search)) {
        throw std::invalid_argument(
            "GpuExecutionConfig.cudnn_conv_algorithm_search is invalid."
        );
    }

    if (disable_provider_synchronization) {
        throw std::invalid_argument(
            "GpuExecutionConfig.disable_provider_synchronization must remain "
            "false because InferenceEngine::run() publishes synchronous host "
            "output."
        );
    }
}

/**
 * @brief Validate TensorRT Execution Provider configuration.
 */
void OrtTensorRtConfig::validate() const {
    constexpr std::size_t maximum_provider_integer =
        static_cast<std::size_t>(
            std::numeric_limits<int>::max()
        );

    if (
        minimum_subgraph_size == 0U
        || minimum_subgraph_size > maximum_provider_integer
    ) {
        throw std::invalid_argument(
            "OrtTensorRtConfig.minimum_subgraph_size must be inside "
            "[1, INT_MAX]."
        );
    }

    if (
        maximum_partition_iterations == 0U
        || maximum_partition_iterations > maximum_provider_integer
    ) {
        throw std::invalid_argument(
            "OrtTensorRtConfig.maximum_partition_iterations must be inside "
            "[1, INT_MAX]."
        );
    }

    if (
        builder_optimization_level < 0
        || builder_optimization_level > 5
    ) {
        throw std::invalid_argument(
            "OrtTensorRtConfig.builder_optimization_level must be inside "
            "[0, 5]."
        );
    }

    if (auxiliary_streams < -1) {
        throw std::invalid_argument(
            "OrtTensorRtConfig.auxiliary_streams must be -1 or non-negative."
        );
    }

    if (enable_engine_cache && engine_cache_directory.empty()) {
        throw std::invalid_argument(
            "OrtTensorRtConfig.engine_cache_directory is required when "
            "engine caching is enabled."
        );
    }

    if (enable_timing_cache && timing_cache_directory.empty()) {
        throw std::invalid_argument(
            "OrtTensorRtConfig.timing_cache_directory is required when "
            "timing caching is enabled."
        );
    }
}

/**
 * @brief Validate backend, artifact, precision, and nested configuration.
 */
void InferenceEngineConfig::validate() const {
    if (artifact_path.empty()) {
        throw std::invalid_argument(
            "InferenceEngineConfig.artifact_path must not be empty."
        );
    }

    if (!is_supported_backend(backend)) {
        throw std::invalid_argument(
            "InferenceEngineConfig.backend is invalid."
        );
    }

    if (!is_supported_precision(artifact_precision)) {
        throw std::invalid_argument(
            "InferenceEngineConfig.artifact_precision is invalid."
        );
    }

    if (!is_valid_tensor_name(expected_input_name)) {
        throw std::invalid_argument(
            "InferenceEngineConfig.expected_input_name is invalid."
        );
    }

    if (!is_valid_tensor_name(expected_output_name)) {
        throw std::invalid_argument(
            "InferenceEngineConfig.expected_output_name is invalid."
        );
    }

    ort.validate();
    gpu.validate();
    ort_tensorrt.validate();

    const std::string extension = ascii_lowercase(
        artifact_path.extension().string()
    );

    if (backend == InferenceBackend::NativeTensorRt) {
        if (extension != ".engine") {
            throw std::invalid_argument(
                "Native TensorRT requires a .engine artifact."
            );
        }
    } else if (extension != ".onnx") {
        throw std::invalid_argument(
            "ONNX Runtime backends require a .onnx artifact."
        );
    }

    if (
        gpu.enable_cuda_graph
        && backend != InferenceBackend::NativeTensorRt
    ) {
        throw std::invalid_argument(
            "GpuExecutionConfig.enable_cuda_graph is supported only by the "
            "native TensorRT backend. The project's ONNX Runtime GPU package "
            "uses CUDA 13, while native TensorRT is linked through CUDA 12.8; "
            "their streams and device allocations are intentionally isolated."
        );
    }

    if (
        backend == InferenceBackend::OrtTensorRt
        && artifact_precision == ArtifactPrecision::Int8
    ) {
        throw std::invalid_argument(
            "This project does not route the QDQ INT8 ONNX artifact through "
            "ONNX Runtime TensorRT EP. Use ORT CUDA for that ONNX model or "
            "use the native TensorRT INT8 .engine artifact."
        );
    }
}

class InferenceEngine::Impl final {
public:
    explicit Impl(
        InferenceEngineConfig config
    )
        : config_(std::move(config)) {
        config_.validate();
        validate_artifact_file(config_.artifact_path);

        switch (config_.backend) {
            case InferenceBackend::OrtCpu:
            case InferenceBackend::OrtCuda:
            case InferenceBackend::OrtTensorRt:
                runner_ = std::make_unique<OrtBackendRunner>(
                    config_
                );
                break;

            case InferenceBackend::NativeTensorRt:
#if defined(EDGE_WITH_TENSORRT) && EDGE_WITH_TENSORRT
                runner_ = std::make_unique<NativeTensorRtRunner>(
                    config_
                );
#else
                throw std::runtime_error(
                    "Native TensorRT support is not compiled into this "
                    "binary. Reconfigure with EDGE_ENABLE_TENSORRT=ON."
                );
#endif
                break;
        }
    }

    [[nodiscard]] const InferenceEngineConfig& config()
        const noexcept {
        return config_;
    }

    [[nodiscard]] const InferenceRuntimeInfo& runtime_info()
        const noexcept {
        return runner_->runtime_info();
    }

    [[nodiscard]] MutableTensorView input_tensor() {
        return runner_->input_tensor();
    }

    [[nodiscard]] DetectionTensorView run() {
        has_output_ = false;

        const DetectionTensorView output = runner_->run();

        has_output_ = true;
        ++completed_run_count_;
        return output;
    }

    [[nodiscard]] DetectionTensorView output_tensor() const {
        if (!has_output_) {
            throw std::logic_error(
                "Inference output requested before a successful run."
            );
        }

        return runner_->output_tensor();
    }

    [[nodiscard]] bool has_output() const noexcept {
        return has_output_;
    }

    [[nodiscard]] std::uint64_t completed_run_count() const noexcept {
        return completed_run_count_;
    }

private:
    InferenceEngineConfig config_{};
    std::unique_ptr<BackendRunner> runner_{};

    bool has_output_ = false;
    std::uint64_t completed_run_count_ = 0U;
};

/**
 * @brief Construct one validated backend implementation.
 */
InferenceEngine::InferenceEngine(
    InferenceEngineConfig config
)
    : implementation_(
          std::make_unique<Impl>(std::move(config))
      ) {}

InferenceEngine::~InferenceEngine() = default;

InferenceEngine::InferenceEngine(
    InferenceEngine&& other
) noexcept = default;

InferenceEngine& InferenceEngine::operator=(
    InferenceEngine&& other
) noexcept = default;

/**
 * @brief Return the validated engine configuration.
 */
const InferenceEngineConfig& InferenceEngine::config() const {
    return implementation().config();
}

/**
 * @brief Return discovered model, runtime, provider, and buffer information.
 */
const InferenceRuntimeInfo& InferenceEngine::runtime_info() const {
    return implementation().runtime_info();
}

/**
 * @brief Return the external model-input element type.
 */
TensorElementType InferenceEngine::input_element_type() const {
    return runtime_info().model.input_element_type;
}

/**
 * @brief Expose persistent CPU-writable model-input storage.
 */
MutableTensorView InferenceEngine::input_tensor() {
    return implementation().input_tensor();
}

/**
 * @brief Execute one synchronous backend inference.
 */
DetectionTensorView InferenceEngine::run() {
    return implementation().run();
}

/**
 * @brief Return the most recent successful inference output.
 */
DetectionTensorView InferenceEngine::output_tensor() const {
    return implementation().output_tensor();
}

/**
 * @brief Return whether a successful output is currently available.
 */
bool InferenceEngine::has_output() const noexcept {
    return implementation_ != nullptr
        && implementation_->has_output();
}

/**
 * @brief Return the number of successful runs, or zero after a move.
 */
std::uint64_t InferenceEngine::completed_run_count() const noexcept {
    return implementation_ != nullptr
        ? implementation_->completed_run_count()
        : 0U;
}

/**
 * @brief Return a live implementation or reject use after move.
 */
InferenceEngine::Impl& InferenceEngine::implementation() {
    if (implementation_ == nullptr) {
        throw std::logic_error(
            "InferenceEngine is in a moved-from state."
        );
    }

    return *implementation_;
}

/**
 * @brief Return a live implementation or reject use after move.
 */
const InferenceEngine::Impl& InferenceEngine::implementation() const {
    if (implementation_ == nullptr) {
        throw std::logic_error(
            "InferenceEngine is in a moved-from state."
        );
    }

    return *implementation_;
}

}  // namespace edge
