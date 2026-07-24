#include "benchmark.hpp"
#include "gpu_pipeline.hpp"
#include "image_processor.hpp"
#include "inference_engine.hpp"
#include "postprocess.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cctype>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef EDGE_PROJECT_VERSION
#define EDGE_PROJECT_VERSION "unknown"
#endif

namespace edge {
namespace {

namespace fs = std::filesystem;

std::atomic<bool> g_stop_requested{false};

void handle_interrupt(int) noexcept {
    g_stop_requested.store(true, std::memory_order_relaxed);
}

enum class ApplicationMode : std::uint8_t {
    Inference,
    Benchmark
};

enum class BackendSelection : std::uint8_t {
    Auto,
    OrtCpu,
    OrtCuda,
    OrtTensorRt,
    NativeTensorRt,
    FusedTensorRt
};

enum class SourceKind : std::uint8_t {
    Image,
    ImageDirectory,
    Video,
    Camera
};

struct SourceDescriptor {
    SourceKind kind = SourceKind::Image;
    fs::path path{};
    int camera_index = -1;
};

struct ApplicationOptions {
    ApplicationMode mode = ApplicationMode::Inference;
    BackendSelection backend = BackendSelection::Auto;
    ArtifactPrecision precision = ArtifactPrecision::Float32;

    std::string source_argument{};
    fs::path model_path{};
    fs::path output_directory{"outputs/cpp"};
    fs::path ort_cache_directory{"models/ort_trt_cache"};

    int device_id = 0;
    int intra_op_threads = 0;
    int inter_op_threads = 0;

    std::size_t pipeline_depth = 2U;
    ImageSize maximum_host_image_size{1920, 1080};

    float confidence_threshold = 0.25F;
    std::size_t maximum_detections = kModelMaximumDetections;

    bool save_output = true;
    bool print_detections = false;
    bool debug_parity = false;
    bool enable_cuda_graph = true;
    bool require_cuda_graph = false;
    bool ort_cuda_fallback = true;

    std::uint64_t maximum_stream_frames = 0U;

    BenchmarkMode benchmark_mode = BenchmarkMode::SynchronousLatency;
    BenchmarkTimingScope benchmark_timing_scope =
        BenchmarkTimingScope::EndToEnd;

    std::uint64_t warmup_iterations = 10U;
    std::uint64_t measured_iterations = 100U;
    std::size_t benchmark_source_frames = 16U;
    bool collect_stage_timings = true;
    bool retain_raw_samples = false;
    BenchmarkFailurePolicy failure_policy =
        BenchmarkFailurePolicy::StopOnFirstFailure;
    std::size_t maximum_recorded_failures = 16U;

    bool show_help = false;
    bool show_version = false;
};

struct InferenceFrameResult {
    DetectionBuffer detections{};
    std::optional<bool> used_cuda_graph{};
};

struct PendingFusedFrame {
    GpuPipelineTicket ticket{};
    cv::Mat frame{};
    std::uint64_t frame_index = 0U;
    std::string source_label{};
};

struct BackendAttemptRecord {
    InferenceBackend backend = InferenceBackend::OrtCpu;
    bool succeeded = false;
    std::string error{};
};

struct EngineSelectionResult {
    std::unique_ptr<InferenceEngine> engine{};
    BackendSelection requested_backend = BackendSelection::Auto;
    InferenceBackend selected_backend = InferenceBackend::OrtCpu;
    std::vector<std::string> available_ort_providers{};
    std::vector<BackendAttemptRecord> attempts{};
    bool automatic = false;
};

[[nodiscard]] std::string lowercase(std::string value) {
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

[[nodiscard]] std::string path_to_utf8(const fs::path& path) {
#if defined(_WIN32)
    return path.u8string();
#else
    return path.string();
#endif
}

[[nodiscard]] bool starts_with(
    std::string_view value,
    std::string_view prefix
) noexcept {
    return value.size() >= prefix.size()
        && value.substr(0U, prefix.size()) == prefix;
}

template <typename Integer>
[[nodiscard]] Integer parse_integer(
    const std::string& text,
    std::string_view option_name
) {
    static_assert(std::numeric_limits<Integer>::is_integer);

    std::size_t parsed_count = 0U;

    try {
        if constexpr (std::numeric_limits<Integer>::is_signed) {
            const long long value = std::stoll(text, &parsed_count, 10);

            if (
                value < static_cast<long long>(
                    std::numeric_limits<Integer>::min()
                )
                || value > static_cast<long long>(
                    std::numeric_limits<Integer>::max()
                )
            ) {
                throw std::out_of_range{"integer range"};
            }

            if (parsed_count != text.size()) {
                throw std::invalid_argument{"trailing characters"};
            }

            return static_cast<Integer>(value);
        } else {
            if (!text.empty() && text.front() == '-') {
                throw std::out_of_range{"negative unsigned value"};
            }

            const unsigned long long value =
                std::stoull(text, &parsed_count, 10);

            if (
                value > static_cast<unsigned long long>(
                    std::numeric_limits<Integer>::max()
                )
            ) {
                throw std::out_of_range{"integer range"};
            }

            if (parsed_count != text.size()) {
                throw std::invalid_argument{"trailing characters"};
            }

            return static_cast<Integer>(value);
        }
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "Invalid value for "
            + std::string{option_name}
            + ": '"
            + text
            + "'."
        );
    }
}

[[nodiscard]] double parse_double(
    const std::string& text,
    std::string_view option_name
) {
    std::size_t parsed_count = 0U;

    try {
        const double value = std::stod(text, &parsed_count);

        if (parsed_count != text.size() || !std::isfinite(value)) {
            throw std::invalid_argument{"invalid floating-point value"};
        }

        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "Invalid value for "
            + std::string{option_name}
            + ": '"
            + text
            + "'."
        );
    }
}

[[nodiscard]] const char* require_option_value(
    int argc,
    char** argv,
    int& index,
    std::string_view option_name
) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(
            "Missing value after " + std::string{option_name} + "."
        );
    }

    ++index;
    return argv[index];
}

[[nodiscard]] ApplicationMode parse_application_mode(
    const std::string& value
) {
    const std::string normalized = lowercase(value);

    if (normalized == "infer" || normalized == "inference") {
        return ApplicationMode::Inference;
    }

    if (normalized == "benchmark" || normalized == "bench") {
        return ApplicationMode::Benchmark;
    }

    throw std::invalid_argument(
        "Unsupported --mode value: '" + value + "'."
    );
}

[[nodiscard]] BackendSelection parse_backend(
    const std::string& value
) {
    const std::string normalized = lowercase(value);

    if (normalized == "auto") {
        return BackendSelection::Auto;
    }

    if (normalized == "ort-cpu" || normalized == "ort_cpu") {
        return BackendSelection::OrtCpu;
    }

    if (normalized == "ort-cuda" || normalized == "ort_cuda") {
        return BackendSelection::OrtCuda;
    }

    if (
        normalized == "ort-trt"
        || normalized == "ort-tensorrt"
        || normalized == "ort_tensorrt"
    ) {
        return BackendSelection::OrtTensorRt;
    }

    if (
        normalized == "native-trt"
        || normalized == "native-tensorrt"
        || normalized == "native_tensorrt"
    ) {
        return BackendSelection::NativeTensorRt;
    }

    if (
        normalized == "fused-trt"
        || normalized == "fused-tensorrt"
        || normalized == "fused_tensorrt"
    ) {
        return BackendSelection::FusedTensorRt;
    }

    throw std::invalid_argument(
        "Unsupported --backend value: '" + value + "'."
    );
}

[[nodiscard]] ArtifactPrecision parse_precision(
    const std::string& value
) {
    const std::string normalized = lowercase(value);

    if (normalized == "fp32" || normalized == "float32") {
        return ArtifactPrecision::Float32;
    }

    if (normalized == "fp16" || normalized == "float16") {
        return ArtifactPrecision::Float16;
    }

    if (normalized == "int8") {
        return ArtifactPrecision::Int8;
    }

    throw std::invalid_argument(
        "Unsupported --precision value: '" + value + "'."
    );
}

[[nodiscard]] BenchmarkMode parse_benchmark_mode(
    const std::string& value
) {
    const std::string normalized = lowercase(value);

    if (
        normalized == "latency"
        || normalized == "synchronous"
        || normalized == "synchronous-latency"
    ) {
        return BenchmarkMode::SynchronousLatency;
    }

    if (
        normalized == "throughput"
        || normalized == "streaming"
        || normalized == "streaming-throughput"
    ) {
        return BenchmarkMode::StreamingThroughput;
    }

    throw std::invalid_argument(
        "Unsupported --benchmark-mode value: '" + value + "'."
    );
}

[[nodiscard]] BenchmarkTimingScope parse_timing_scope(
    const std::string& value
) {
    const std::string normalized = lowercase(value);

    if (
        normalized == "backend"
        || normalized == "backend-execution"
    ) {
        return BenchmarkTimingScope::BackendExecution;
    }

    if (
        normalized == "end-to-end"
        || normalized == "e2e"
    ) {
        return BenchmarkTimingScope::EndToEnd;
    }

    throw std::invalid_argument(
        "Unsupported --timing-scope value: '" + value + "'."
    );
}

[[nodiscard]] BenchmarkFailurePolicy parse_failure_policy(
    const std::string& value
) {
    const std::string normalized = lowercase(value);

    if (normalized == "stop" || normalized == "stop-on-first") {
        return BenchmarkFailurePolicy::StopOnFirstFailure;
    }

    if (normalized == "continue" || normalized == "continue-after-failure") {
        return BenchmarkFailurePolicy::ContinueAfterFailure;
    }

    throw std::invalid_argument(
        "Unsupported --failure-policy value: '" + value + "'."
    );
}

[[nodiscard]] std::string_view backend_name(
    BackendSelection backend
) noexcept {
    switch (backend) {
        case BackendSelection::Auto:
            return "auto";

        case BackendSelection::OrtCpu:
            return "ort_cpu";

        case BackendSelection::OrtCuda:
            return "ort_cuda";

        case BackendSelection::OrtTensorRt:
            return "ort_tensorrt";

        case BackendSelection::NativeTensorRt:
            return "native_tensorrt";

        case BackendSelection::FusedTensorRt:
            return "fused_tensorrt";
    }

    return "unknown";
}

[[nodiscard]] InferenceBackend generic_backend(
    BackendSelection backend
) {
    switch (backend) {
        case BackendSelection::Auto:
            break;

        case BackendSelection::OrtCpu:
            return InferenceBackend::OrtCpu;

        case BackendSelection::OrtCuda:
            return InferenceBackend::OrtCuda;

        case BackendSelection::OrtTensorRt:
            return InferenceBackend::OrtTensorRt;

        case BackendSelection::NativeTensorRt:
            return InferenceBackend::NativeTensorRt;

        case BackendSelection::FusedTensorRt:
            break;
    }

    throw std::invalid_argument(
        "Automatic or fused TensorRT selection is not a direct "
        "InferenceEngine backend."
    );
}

[[nodiscard]] bool is_image_extension(
    const fs::path& path
) {
    const std::string extension = lowercase(path.extension().string());

    return extension == ".jpg"
        || extension == ".jpeg"
        || extension == ".png"
        || extension == ".bmp"
        || extension == ".ppm"
        || extension == ".tif"
        || extension == ".tiff"
        || extension == ".webp";
}

[[nodiscard]] bool is_video_extension(
    const fs::path& path
) {
    const std::string extension = lowercase(path.extension().string());

    return extension == ".mp4"
        || extension == ".avi"
        || extension == ".mkv"
        || extension == ".mov"
        || extension == ".m4v"
        || extension == ".webm";
}

[[nodiscard]] SourceDescriptor describe_source(
    const std::string& source_argument
) {
    constexpr std::string_view camera_prefix{"camera:"};

    if (starts_with(source_argument, camera_prefix)) {
        const std::string index_text{
            source_argument.substr(camera_prefix.size())
        };

        const int camera_index = parse_integer<int>(
            index_text,
            "--source camera index"
        );

        if (camera_index < 0) {
            throw std::invalid_argument(
                "Camera index must be non-negative."
            );
        }

        return SourceDescriptor{
            SourceKind::Camera,
            {},
            camera_index
        };
    }

    const fs::path source_path{source_argument};
    std::error_code error;

    if (fs::is_directory(source_path, error)) {
        return SourceDescriptor{
            SourceKind::ImageDirectory,
            source_path,
            -1
        };
    }

    if (error) {
        throw std::runtime_error(
            "Failed to inspect source path '"
            + path_to_utf8(source_path)
            + "': "
            + error.message()
        );
    }

    if (!fs::is_regular_file(source_path, error)) {
        if (error) {
            throw std::runtime_error(
                "Failed to inspect source path '"
                + path_to_utf8(source_path)
                + "': "
                + error.message()
            );
        }

        throw std::invalid_argument(
            "Source is not a regular file, directory, or camera:<index>: "
            + path_to_utf8(source_path)
        );
    }

    if (is_image_extension(source_path)) {
        return SourceDescriptor{
            SourceKind::Image,
            source_path,
            -1
        };
    }

    if (is_video_extension(source_path)) {
        return SourceDescriptor{
            SourceKind::Video,
            source_path,
            -1
        };
    }

    throw std::invalid_argument(
        "Unsupported source extension: "
        + path_to_utf8(source_path.extension())
    );
}

void print_usage(std::ostream& output) {
    output
        << "edge-perception-cpp " << EDGE_PROJECT_VERSION << "\n\n"
        << "Usage:\n"
        << "  edge_perception --mode infer --source <path|camera:N> "
           "--backend <name> --model <artifact> [options]\n"
        << "  edge_perception --mode benchmark --source <path> "
           "--backend <name> --model <artifact> [options]\n\n"
        << "Core options:\n"
        << "  --mode infer|benchmark               Operation mode (default: infer)\n"
        << "  --source <path|camera:N>             Image, image directory, video, or camera\n"
        << "  --backend <name>                     auto, ort-cpu, ort-cuda, ort-trt,\n"
        << "                                       native-trt, or fused-trt\n"
        << "                                       default: auto\n"
        << "  --model <path>                       .onnx or .engine artifact\n"
        << "  --precision fp32|fp16|int8           Artifact precision (default: fp32)\n"
        << "  --device <N>                         CUDA device index (default: 0)\n"
        << "  --confidence <0..1>                  Confidence threshold (default: 0.25)\n"
        << "  --max-detections <N>                 Final detection limit (default: 300)\n"
        << "  --output-dir <path>                  Output root (default: outputs/cpp)\n"
        << "  --pipeline-depth <1..8>              Fused/streaming depth (default: 2)\n"
        << "  --debug-parity                       Fused CPU-postprocess parity path\n"
        << "  --no-cuda-graph                      Disable supported CUDA graph paths\n"
        << "  --require-cuda-graph                 Fail fused execution if capture fails\n"
        << "  --intra-op-threads <N>               ORT intra-op threads; 0 = runtime default\n"
        << "  --inter-op-threads <N>               ORT inter-op threads; 0 = runtime default\n"
        << "  --ort-cache-dir <path>               ORT TensorRT engine/timing cache\n"
        << "  --no-ort-cuda-fallback               Disable CUDA fallback behind ORT TensorRT\n\n"
        << "Inference options:\n"
        << "  --no-save                            Do not write annotated media or JSONL\n"
        << "  --print-detections                   Print every detection row\n"
        << "  --max-frames <N>                     Stream frame limit; 0 = until end/Ctrl+C\n"
        << "  --max-host-width <N>                 Fused pinned host capacity width\n"
        << "  --max-host-height <N>                Fused pinned host capacity height\n\n"
        << "Benchmark options:\n"
        << "  --benchmark-mode latency|throughput  Default: latency\n"
        << "  --timing-scope backend|end-to-end    Default: end-to-end\n"
        << "  --warmup <N>                         Untimed warm-up frames (default: 10)\n"
        << "  --iterations <N>                     Measured frames (default: 100)\n"
        << "  --benchmark-source-frames <N>        Preloaded source frames (default: 16)\n"
        << "  --no-stage-timings                   Minimize host timing instrumentation\n"
        << "  --retain-samples                     Write per-frame samples CSV\n"
        << "  --failure-policy stop|continue       Default: stop\n"
        << "  --max-recorded-failures <N>          Detailed failure cap (default: 16)\n\n"
        << "Other:\n"
        << "  --help, -h                           Show this message\n"
        << "  --version                            Show version\n\n"
        << "Examples:\n"
        << "  edge_perception --mode infer --source frame.jpg "
           "--backend auto --model models/onnx/yolo26m_bdd100k_fp32.onnx\n"
        << "  edge_perception --mode benchmark --source frame.jpg "
           "--backend fused-trt --model models/engine/yolo26m_bdd100k_fp16.engine "
           "--precision fp16 --pipeline-depth 1 --benchmark-mode latency\n"
        << "  edge_perception --mode benchmark --source clips/sample.mp4 "
           "--backend fused-trt --model models/engine/yolo26m_bdd100k_fp16.engine "
           "--precision fp16 --pipeline-depth 2 --benchmark-mode throughput\n";
}

[[nodiscard]] ApplicationOptions parse_arguments(
    int argc,
    char** argv
) {
    ApplicationOptions options{};

    for (int index = 1; index < argc; ++index) {
        const std::string argument{argv[index]};

        if (argument == "--help" || argument == "-h") {
            options.show_help = true;
            continue;
        }

        if (argument == "--version") {
            options.show_version = true;
            continue;
        }

        if (argument == "--mode") {
            options.mode = parse_application_mode(
                require_option_value(argc, argv, index, argument)
            );
            continue;
        }

        if (argument == "--source") {
            options.source_argument =
                require_option_value(argc, argv, index, argument);
            continue;
        }

        if (argument == "--backend") {
            options.backend = parse_backend(
                require_option_value(argc, argv, index, argument)
            );
            continue;
        }

        if (argument == "--model") {
            options.model_path =
                require_option_value(argc, argv, index, argument);
            continue;
        }

        if (argument == "--precision") {
            options.precision = parse_precision(
                require_option_value(argc, argv, index, argument)
            );
            continue;
        }

        if (argument == "--device") {
            options.device_id = parse_integer<int>(
                require_option_value(argc, argv, index, argument),
                argument
            );
            continue;
        }

        if (argument == "--pipeline-depth") {
            options.pipeline_depth = parse_integer<std::size_t>(
                require_option_value(argc, argv, index, argument),
                argument
            );
            continue;
        }

        if (argument == "--confidence") {
            options.confidence_threshold = static_cast<float>(
                parse_double(
                    require_option_value(argc, argv, index, argument),
                    argument
                )
            );
            continue;
        }

        if (argument == "--max-detections") {
            options.maximum_detections = parse_integer<std::size_t>(
                require_option_value(argc, argv, index, argument),
                argument
            );
            continue;
        }

        if (argument == "--output-dir") {
            options.output_directory =
                require_option_value(argc, argv, index, argument);
            continue;
        }

        if (argument == "--no-save") {
            options.save_output = false;
            continue;
        }

        if (argument == "--print-detections") {
            options.print_detections = true;
            continue;
        }

        if (argument == "--debug-parity") {
            options.debug_parity = true;
            continue;
        }

        if (argument == "--no-cuda-graph") {
            options.enable_cuda_graph = false;
            continue;
        }

        if (argument == "--require-cuda-graph") {
            options.require_cuda_graph = true;
            continue;
        }

        if (argument == "--max-frames") {
            options.maximum_stream_frames = parse_integer<std::uint64_t>(
                require_option_value(argc, argv, index, argument),
                argument
            );
            continue;
        }

        if (argument == "--max-host-width") {
            options.maximum_host_image_size.width = parse_integer<int>(
                require_option_value(argc, argv, index, argument),
                argument
            );
            continue;
        }

        if (argument == "--max-host-height") {
            options.maximum_host_image_size.height = parse_integer<int>(
                require_option_value(argc, argv, index, argument),
                argument
            );
            continue;
        }

        if (argument == "--intra-op-threads") {
            options.intra_op_threads = parse_integer<int>(
                require_option_value(argc, argv, index, argument),
                argument
            );
            continue;
        }

        if (argument == "--inter-op-threads") {
            options.inter_op_threads = parse_integer<int>(
                require_option_value(argc, argv, index, argument),
                argument
            );
            continue;
        }

        if (argument == "--ort-cache-dir") {
            options.ort_cache_directory =
                require_option_value(argc, argv, index, argument);
            continue;
        }

        if (argument == "--no-ort-cuda-fallback") {
            options.ort_cuda_fallback = false;
            continue;
        }

        if (argument == "--benchmark-mode") {
            options.benchmark_mode = parse_benchmark_mode(
                require_option_value(argc, argv, index, argument)
            );
            continue;
        }

        if (argument == "--timing-scope") {
            options.benchmark_timing_scope = parse_timing_scope(
                require_option_value(argc, argv, index, argument)
            );
            continue;
        }

        if (argument == "--warmup") {
            options.warmup_iterations = parse_integer<std::uint64_t>(
                require_option_value(argc, argv, index, argument),
                argument
            );
            continue;
        }

        if (argument == "--iterations") {
            options.measured_iterations = parse_integer<std::uint64_t>(
                require_option_value(argc, argv, index, argument),
                argument
            );
            continue;
        }

        if (argument == "--benchmark-source-frames") {
            options.benchmark_source_frames = parse_integer<std::size_t>(
                require_option_value(argc, argv, index, argument),
                argument
            );
            continue;
        }

        if (argument == "--no-stage-timings") {
            options.collect_stage_timings = false;
            continue;
        }

        if (argument == "--retain-samples") {
            options.retain_raw_samples = true;
            continue;
        }

        if (argument == "--failure-policy") {
            options.failure_policy = parse_failure_policy(
                require_option_value(argc, argv, index, argument)
            );
            continue;
        }

        if (argument == "--max-recorded-failures") {
            options.maximum_recorded_failures = parse_integer<std::size_t>(
                require_option_value(argc, argv, index, argument),
                argument
            );
            continue;
        }

        throw std::invalid_argument(
            "Unknown command-line option: '" + argument + "'."
        );
    }

    return options;
}

void validate_options(const ApplicationOptions& options) {
    if (options.show_help || options.show_version) {
        return;
    }

    if (options.source_argument.empty()) {
        throw std::invalid_argument("--source is required.");
    }

    if (options.model_path.empty()) {
        throw std::invalid_argument("--model is required.");
    }

    if (!fs::is_regular_file(options.model_path)) {
        throw std::invalid_argument(
            "Model artifact does not exist: "
            + path_to_utf8(options.model_path)
        );
    }

    const std::string model_extension =
        lowercase(options.model_path.extension().string());

    const bool requires_onnx =
        options.backend == BackendSelection::Auto
        || options.backend == BackendSelection::OrtCpu
        || options.backend == BackendSelection::OrtCuda
        || options.backend == BackendSelection::OrtTensorRt;

    const bool requires_engine =
        options.backend == BackendSelection::NativeTensorRt
        || options.backend == BackendSelection::FusedTensorRt;

    if (requires_onnx && model_extension != ".onnx") {
        throw std::invalid_argument(
            "The selected backend requires an .onnx model artifact."
        );
    }

    if (requires_engine && model_extension != ".engine") {
        throw std::invalid_argument(
            "The selected backend requires a TensorRT .engine artifact."
        );
    }

    if (
        options.backend == BackendSelection::Auto
        && options.require_cuda_graph
    ) {
        throw std::invalid_argument(
            "--require-cuda-graph requires an explicit GPU backend. "
            "Automatic selection may legitimately fall back to ORT CPU."
        );
    }

    if (options.device_id < 0) {
        throw std::invalid_argument("--device must be non-negative.");
    }

    if (options.intra_op_threads < 0 || options.inter_op_threads < 0) {
        throw std::invalid_argument(
            "ORT thread counts must be non-negative."
        );
    }

    if (options.pipeline_depth < 1U || options.pipeline_depth > 8U) {
        throw std::invalid_argument(
            "--pipeline-depth must be inside [1, 8]."
        );
    }

    if (!options.maximum_host_image_size.valid()) {
        throw std::invalid_argument(
            "Maximum fused host-image dimensions must be positive."
        );
    }

    if (
        !std::isfinite(options.confidence_threshold)
        || options.confidence_threshold < 0.0F
        || options.confidence_threshold > 1.0F
    ) {
        throw std::invalid_argument(
            "--confidence must be finite and inside [0, 1]."
        );
    }

    if (
        options.maximum_detections == 0U
        || options.maximum_detections > kModelMaximumDetections
    ) {
        throw std::invalid_argument(
            "--max-detections must be inside [1, 300]."
        );
    }

    if (options.require_cuda_graph && !options.enable_cuda_graph) {
        throw std::invalid_argument(
            "--require-cuda-graph cannot be combined with --no-cuda-graph."
        );
    }

    if (
        options.debug_parity
        && options.backend != BackendSelection::FusedTensorRt
    ) {
        throw std::invalid_argument(
            "--debug-parity applies only to --backend fused-trt."
        );
    }

    if (
        options.backend == BackendSelection::FusedTensorRt
        && options.mode == ApplicationMode::Benchmark
        && options.benchmark_timing_scope
            != BenchmarkTimingScope::EndToEnd
    ) {
        throw std::invalid_argument(
            "fused-trt exposes only end-to-end benchmark timing."
        );
    }

    if (
        options.mode == ApplicationMode::Benchmark
        && options.measured_iterations == 0U
    ) {
        throw std::invalid_argument(
            "--iterations must be positive."
        );
    }

    if (
        options.mode == ApplicationMode::Benchmark
        && options.benchmark_source_frames == 0U
    ) {
        throw std::invalid_argument(
            "--benchmark-source-frames must be positive."
        );
    }

    if (
        (options.save_output || options.mode == ApplicationMode::Benchmark)
        && options.output_directory.empty()
    ) {
        throw std::invalid_argument(
            "--output-dir must not be empty when output is enabled."
        );
    }
}

[[nodiscard]] std::vector<fs::path> collect_image_paths(
    const fs::path& directory
) {
    std::vector<fs::path> images{};
    std::error_code error;

    const fs::recursive_directory_iterator end{};

    for (
        fs::recursive_directory_iterator iterator{
            directory,
            fs::directory_options::skip_permission_denied,
            error
        };
        iterator != end;
        iterator.increment(error)
    ) {
        if (error) {
            throw std::runtime_error(
                "Failed while traversing image directory '"
                + path_to_utf8(directory)
                + "': "
                + error.message()
            );
        }

        if (
            iterator->is_regular_file(error)
            && !error
            && is_image_extension(iterator->path())
        ) {
            images.push_back(iterator->path());
        }

        if (error) {
            throw std::runtime_error(
                "Failed while inspecting image directory entry: "
                + error.message()
            );
        }
    }

    std::sort(images.begin(), images.end());

    if (images.empty()) {
        throw std::invalid_argument(
            "No supported images were found under: "
            + path_to_utf8(directory)
        );
    }

    return images;
}

void validate_frame(
    const cv::Mat& frame,
    std::string_view source_label
) {
    if (frame.empty()) {
        throw std::runtime_error(
            "Received an empty frame from " + std::string{source_label} + "."
        );
    }

    if (frame.dims != 2 || frame.type() != CV_8UC3) {
        throw std::runtime_error(
            "Frame from "
            + std::string{source_label}
            + " is not a two-dimensional CV_8UC3 BGR image."
        );
    }
}

[[nodiscard]] cv::VideoCapture open_capture(
    const SourceDescriptor& source
) {
    cv::VideoCapture capture;

    if (source.kind == SourceKind::Camera) {
        capture.open(source.camera_index);
    } else {
        capture.open(path_to_utf8(source.path));
    }

    if (!capture.isOpened()) {
        if (source.kind == SourceKind::Camera) {
            throw std::runtime_error(
                "Failed to open camera index "
                + std::to_string(source.camera_index)
                + "."
            );
        }

        throw std::runtime_error(
            "Failed to open video source: " + path_to_utf8(source.path)
        );
    }

    return capture;
}

[[nodiscard]] PreprocessConfig make_preprocess_config() {
    PreprocessConfig config{};
    config.validate();
    return config;
}

[[nodiscard]] PostprocessConfig make_postprocess_config(
    const ApplicationOptions& options
) {
    PostprocessConfig config{};
    config.confidence_threshold = options.confidence_threshold;
    config.maximum_detections = options.maximum_detections;
    config.validate();
    return config;
}

[[nodiscard]] InferenceEngineConfig make_engine_config(
    const ApplicationOptions& options,
    InferenceBackend backend
) {
    InferenceEngineConfig config{};

    config.artifact_path = options.model_path;
    config.backend = backend;
    config.artifact_precision = options.precision;

    config.ort.intra_op_thread_count = options.intra_op_threads;
    config.ort.inter_op_thread_count = options.inter_op_threads;

    config.gpu.device_id = options.device_id;

    // InferenceEngine::run() exposes a synchronous CPU-output contract.
    // Keep normal ONNX Runtime provider synchronization enabled so CUDA
    // execution and the bound D2H output copy are complete before postprocessing.
    config.gpu.disable_provider_synchronization = false;

    // The generic GpuExecutionConfig CUDA-graph path belongs only to the
    // native TensorRT InferenceEngine backend. ORT CPU/CUDA/TRT execution
    // owns its own provider streams and allocations.
    config.gpu.enable_cuda_graph =
        options.enable_cuda_graph
        && config.backend == InferenceBackend::NativeTensorRt;

    config.ort_tensorrt.enable_cuda_fallback =
        options.ort_cuda_fallback;

    config.ort_tensorrt.engine_cache_directory =
        options.ort_cache_directory;

    config.ort_tensorrt.timing_cache_directory =
        options.ort_cache_directory;

    // ORT TensorRT EP has a separate provider-controlled CUDA-graph option.
    config.ort_tensorrt.enable_cuda_graph =
        options.enable_cuda_graph
        && config.backend == InferenceBackend::OrtTensorRt;

    config.validate();
    return config;
}

[[nodiscard]] std::string_view required_provider_name(
    InferenceBackend backend
) noexcept {
    switch (backend) {
        case InferenceBackend::OrtCpu:
            return "CPUExecutionProvider";

        case InferenceBackend::OrtCuda:
            return "CUDAExecutionProvider";

        case InferenceBackend::OrtTensorRt:
            return "TensorrtExecutionProvider";

        case InferenceBackend::NativeTensorRt:
            return {};
    }

    return {};
}

[[nodiscard]] bool contains_string(
    const std::vector<std::string>& values,
    std::string_view expected
) {
    return std::find(
        values.begin(),
        values.end(),
        expected
    ) != values.end();
}

[[nodiscard]] EngineSelectionResult select_engine(
    const ApplicationOptions& options
) {
    EngineSelectionResult result{};
    result.requested_backend = options.backend;
    result.automatic = options.backend == BackendSelection::Auto;

    try {
        result.available_ort_providers =
            InferenceEngine::available_onnxruntime_providers();
    } catch (const std::exception& exception) {
        std::cerr
            << "ONNX Runtime provider discovery failed: "
            << exception.what()
            << '\n';
    }

    if (!result.automatic) {
        const InferenceBackend backend = generic_backend(options.backend);

        result.engine = std::make_unique<InferenceEngine>(
            make_engine_config(options, backend)
        );
        result.selected_backend = backend;
        result.attempts.push_back(
            BackendAttemptRecord{backend, true, {}}
        );
        return result;
    }

    constexpr std::array<InferenceBackend, 3> candidates{
        InferenceBackend::OrtTensorRt,
        InferenceBackend::OrtCuda,
        InferenceBackend::OrtCpu
    };

    for (const InferenceBackend backend : candidates) {
        const std::string_view provider = required_provider_name(backend);

        if (
            !provider.empty()
            && !result.available_ort_providers.empty()
            && !contains_string(result.available_ort_providers, provider)
        ) {
            result.attempts.push_back(
                BackendAttemptRecord{
                    backend,
                    false,
                    "ONNX Runtime does not advertise "
                        + std::string{provider}
                }
            );
            continue;
        }

        try {
            result.engine = std::make_unique<InferenceEngine>(
                make_engine_config(options, backend)
            );
            result.selected_backend = backend;
            result.attempts.push_back(
                BackendAttemptRecord{backend, true, {}}
            );
            return result;
        } catch (const std::exception& exception) {
            result.attempts.push_back(
                BackendAttemptRecord{
                    backend,
                    false,
                    exception.what()
                }
            );
        }
    }

    std::ostringstream message;
    message
        << "Automatic backend selection failed. "
        << "Every portable ONNX backend was unavailable.";

    for (const BackendAttemptRecord& attempt : result.attempts) {
        message
            << "\n  "
            << inference_backend_name(attempt.backend)
            << ": "
            << (attempt.error.empty() ? "unknown failure" : attempt.error);
    }

    throw std::runtime_error(message.str());
}

void print_backend_selection(
    const EngineSelectionResult& selection
) {
    std::cout
        << "Requested backend: "
        << backend_name(selection.requested_backend)
        << '\n';

    if (!selection.available_ort_providers.empty()) {
        std::cout << "Available ORT providers:";

        for (const std::string& provider : selection.available_ort_providers) {
            std::cout << ' ' << provider;
        }

        std::cout << '\n';
    }

    for (const BackendAttemptRecord& attempt : selection.attempts) {
        std::cout
            << "Backend attempt "
            << inference_backend_name(attempt.backend)
            << ": "
            << (attempt.succeeded ? "selected" : "unavailable");

        if (!attempt.error.empty()) {
            std::cout << " (" << attempt.error << ')';
        }

        std::cout << '\n';
    }

    std::cout
        << "Selected backend: "
        << inference_backend_name(selection.selected_backend)
        << '\n';
}

[[nodiscard]] NativeTensorRtPipelineConfig make_pipeline_config(
    const ApplicationOptions& options
) {
    NativeTensorRtPipelineConfig config{};
    config.engine_path = options.model_path;
    config.device_id = options.device_id;
    config.pipeline_depth = options.pipeline_depth;
    config.maximum_host_image_size = options.maximum_host_image_size;
    config.preprocess = make_preprocess_config();
    config.postprocess = make_postprocess_config(options);

    config.enable_cuda_graph =
        options.enable_cuda_graph && !options.debug_parity;

    config.require_cuda_graph =
        options.require_cuda_graph && config.enable_cuda_graph;

    config.enable_gpu_postprocess = !options.debug_parity;
    config.use_high_priority_streams = true;
    config.use_engine_auxiliary_streams = true;
    config.minimize_runtime_instrumentation = true;
    config.capture_external_sources = false;

    config.validate();
    return config;
}

[[nodiscard]] BenchmarkConfig make_benchmark_config(
    const ApplicationOptions& options
) {
    BenchmarkConfig config{};
    config.mode = options.benchmark_mode;
    config.timing_scope = options.benchmark_timing_scope;
    config.warmup_iterations = options.warmup_iterations;
    config.measured_iterations = options.measured_iterations;
    config.pipeline_depth = options.pipeline_depth;
    config.collect_stage_timings = options.collect_stage_timings;
    config.retain_raw_samples = options.retain_raw_samples;
    config.failure_policy = options.failure_policy;
    config.maximum_recorded_failures =
        options.maximum_recorded_failures;

    config.validate();
    return config;
}

void print_engine_runtime_info(const InferenceEngine& engine) {
    const InferenceRuntimeInfo& info = engine.runtime_info();

    std::cout
        << "Backend: "
        << inference_backend_name(info.backend)
        << "\nPrecision: "
        << artifact_precision_name(info.artifact_precision)
        << "\nRuntime: "
        << info.runtime_name
        << ' '
        << info.runtime_version
        << "\nInput: ["
        << info.model.input_shape[0]
        << ", "
        << info.model.input_shape[1]
        << ", "
        << info.model.input_shape[2]
        << ", "
        << info.model.input_shape[3]
        << "]\nOutput: ["
        << info.model.output_shape[0]
        << ", "
        << info.model.output_shape[1]
        << ", "
        << info.model.output_shape[2]
        << "]\nHost input pinned: "
        << (info.buffers.host_input_is_pinned ? "yes" : "no")
        << "\nHost output pinned: "
        << (info.buffers.host_output_is_pinned ? "yes" : "no")
        << '\n';

    if (!info.active_execution_providers.empty()) {
        std::cout << "Execution providers:";

        for (const std::string& provider : info.active_execution_providers) {
            std::cout << ' ' << provider;
        }

        std::cout << '\n';
    }
}

void print_pipeline_runtime_info(
    const NativeTensorRtPipeline& pipeline
) {
    const GpuPipelineRuntimeInfo& info = pipeline.runtime_info();

    std::cout
        << "Backend: fused_tensorrt\n"
        << "Runtime: "
        << info.runtime_name
        << ' '
        << info.runtime_version
        << "\nPipeline depth: "
        << pipeline.pipeline_depth()
        << "\nGPU postprocessing: "
        << (info.gpu_postprocess_enabled ? "enabled" : "disabled")
        << "\nCUDA graphs: "
        << (info.cuda_graph_enabled ? "enabled" : "disabled")
        << "\nInput: ["
        << info.input_shape[0]
        << ", "
        << info.input_shape[1]
        << ", "
        << info.input_shape[2]
        << ", "
        << info.input_shape[3]
        << "]\nOutput: ["
        << info.output_shape[0]
        << ", "
        << info.output_shape[1]
        << ", "
        << info.output_shape[2]
        << "]\nPinned host memory: "
        << info.memory.total_pinned_host_bytes
        << " bytes\nDevice memory: "
        << info.memory.total_device_bytes
        << " bytes\n";
}

[[nodiscard]] InferenceFrameResult run_generic_frame(
    InferenceEngine& engine,
    ImageProcessor& processor,
    const Postprocessor& postprocessor,
    DetectionBuffer& detection_workspace,
    const cv::Mat& frame
) {
    const PreprocessResult preprocessing = processor.preprocess(
        frame,
        engine.input_tensor()
    );

    const DetectionTensorView output = engine.run();

    static_cast<void>(
        postprocessor.process(
            output,
            preprocessing.transform,
            detection_workspace
        )
    );

    return InferenceFrameResult{
        detection_workspace,
        std::nullopt
    };
}

[[nodiscard]] InferenceFrameResult run_fused_frame(
    NativeTensorRtPipeline& pipeline,
    const cv::Mat& frame
) {
    GpuPipelineResult result = pipeline.run(frame);

    return InferenceFrameResult{
        std::move(result.detections),
        result.used_cuda_graph
    };
}

[[nodiscard]] cv::Scalar class_color(
    std::int32_t class_id
) noexcept {
    static constexpr int colors[][3]{
        {255, 96, 96},
        {96, 255, 96},
        {96, 96, 255},
        {255, 192, 96},
        {192, 96, 255},
        {96, 255, 255},
        {255, 96, 192},
        {192, 255, 96},
        {96, 192, 255},
        {224, 224, 224}
    };

    const std::size_t index =
        class_id >= 0
            ? static_cast<std::size_t>(class_id)
                % (sizeof(colors) / sizeof(colors[0]))
            : 0U;

    return cv::Scalar{
        static_cast<double>(colors[index][0]),
        static_cast<double>(colors[index][1]),
        static_cast<double>(colors[index][2])
    };
}

[[nodiscard]] cv::Mat draw_detections(
    const cv::Mat& frame,
    const DetectionBuffer& detections
) {
    cv::Mat annotated = frame.clone();

    for (const Detection& detection : detections) {
        const cv::Scalar color = class_color(detection.class_id);

        const int x_min = std::clamp(
            static_cast<int>(std::lround(detection.box.x_min)),
            0,
            annotated.cols
        );

        const int y_min = std::clamp(
            static_cast<int>(std::lround(detection.box.y_min)),
            0,
            annotated.rows
        );

        const int x_max = std::clamp(
            static_cast<int>(std::lround(detection.box.x_max)),
            0,
            annotated.cols
        );

        const int y_max = std::clamp(
            static_cast<int>(std::lround(detection.box.y_max)),
            0,
            annotated.rows
        );

        if (x_max <= x_min || y_max <= y_min) {
            continue;
        }

        cv::rectangle(
            annotated,
            cv::Point{x_min, y_min},
            cv::Point{x_max, y_max},
            color,
            2,
            cv::LINE_AA
        );

        std::ostringstream label;
        label
            << bdd100k_class_name(detection.class_id)
            << ' '
            << std::fixed
            << std::setprecision(2)
            << detection.confidence;

        const int label_y = std::max(16, y_min - 6);

        cv::putText(
            annotated,
            label.str(),
            cv::Point{x_min, label_y},
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv::LINE_AA
        );
    }

    return annotated;
}

[[nodiscard]] std::string json_escape(
    std::string_view value
) {
    std::ostringstream output;

    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"':
                output << "\\\"";
                break;

            case '\\':
                output << "\\\\";
                break;

            case '\b':
                output << "\\b";
                break;

            case '\f':
                output << "\\f";
                break;

            case '\n':
                output << "\\n";
                break;

            case '\r':
                output << "\\r";
                break;

            case '\t':
                output << "\\t";
                break;

            default:
                if (character < 0x20U) {
                    output
                        << "\\u"
                        << std::hex
                        << std::setw(4)
                        << std::setfill('0')
                        << static_cast<unsigned int>(character)
                        << std::dec
                        << std::setfill(' ');
                } else {
                    output << static_cast<char>(character);
                }
                break;
        }
    }

    return output.str();
}

void write_backend_selection_metadata(
    const ApplicationOptions& options,
    const EngineSelectionResult& selection
) {
    if (
        !options.save_output
        && options.mode != ApplicationMode::Benchmark
    ) {
        return;
    }

    fs::create_directories(options.output_directory);

    const fs::path output_path =
        options.output_directory / "backend_selection.json";

    std::ofstream output(
        output_path,
        std::ios::binary | std::ios::trunc
    );

    if (!output.is_open()) {
        throw std::runtime_error(
            "Failed to open backend-selection metadata: "
            + path_to_utf8(output_path)
        );
    }

    output
        << "{\n"
        << "  \"requested_backend\": \""
        << json_escape(backend_name(selection.requested_backend))
        << "\",\n"
        << "  \"selected_backend\": \""
        << json_escape(
            inference_backend_name(selection.selected_backend)
        )
        << "\",\n"
        << "  \"automatic\": "
        << (selection.automatic ? "true" : "false")
        << ",\n"
        << "  \"model\": \""
        << json_escape(path_to_utf8(options.model_path))
        << "\",\n"
        << "  \"available_onnxruntime_providers\": [";

    for (
        std::size_t index = 0U;
        index < selection.available_ort_providers.size();
        ++index
    ) {
        if (index != 0U) {
            output << ", ";
        }

        output
            << '"'
            << json_escape(selection.available_ort_providers[index])
            << '"';
    }

    output << "],\n  \"attempts\": [";

    for (
        std::size_t index = 0U;
        index < selection.attempts.size();
        ++index
    ) {
        if (index != 0U) {
            output << ',';
        }

        const BackendAttemptRecord& attempt =
            selection.attempts[index];

        output
            << "\n    {\"backend\": \""
            << json_escape(inference_backend_name(attempt.backend))
            << "\", \"succeeded\": "
            << (attempt.succeeded ? "true" : "false")
            << ", \"error\": ";

        if (attempt.error.empty()) {
            output << "null";
        } else {
            output
                << '"'
                << json_escape(attempt.error)
                << '"';
        }

        output << '}';
    }

    if (!selection.attempts.empty()) {
        output << '\n';
    }

    output << "  ]\n}\n";

    if (!output) {
        throw std::runtime_error(
            "Failed while writing backend-selection metadata."
        );
    }
}

class DetectionJsonlWriter final {
public:
    DetectionJsonlWriter() = default;

    explicit DetectionJsonlWriter(
        const fs::path& output_path
    ) {
        open(output_path);
    }

    void open(
        const fs::path& output_path
    ) {
        if (output_path.empty()) {
            throw std::invalid_argument(
                "Detection JSONL output path must not be empty."
            );
        }

        const fs::path parent = output_path.parent_path();

        if (!parent.empty()) {
            fs::create_directories(parent);
        }

        output_.open(output_path, std::ios::binary | std::ios::trunc);

        if (!output_.is_open()) {
            throw std::runtime_error(
                "Failed to open detection JSONL output: "
                + path_to_utf8(output_path)
            );
        }
    }

    void write(
        std::uint64_t frame_index,
        std::string_view source_label,
        const InferenceFrameResult& result
    ) {
        if (!output_.is_open()) {
            return;
        }

        output_
            << "{\"frame_index\":"
            << frame_index
            << ",\"source\":\""
            << json_escape(source_label)
            << "\",\"used_cuda_graph\":";

        if (result.used_cuda_graph.has_value()) {
            output_ << (*result.used_cuda_graph ? "true" : "false");
        } else {
            output_ << "null";
        }

        output_
            << ",\"detections\":[";

        for (
            std::size_t index = 0U;
            index < result.detections.size();
            ++index
        ) {
            if (index != 0U) {
                output_ << ',';
            }

            const Detection& detection = result.detections[index];

            output_
                << "{\"class_id\":"
                << detection.class_id
                << ",\"class_name\":\""
                << json_escape(bdd100k_class_name(detection.class_id))
                << "\",\"confidence\":"
                << std::setprecision(9)
                << detection.confidence
                << ",\"box\":["
                << detection.box.x_min
                << ','
                << detection.box.y_min
                << ','
                << detection.box.x_max
                << ','
                << detection.box.y_max
                << "]}";
        }

        output_ << "]}\n";

        if (!output_) {
            throw std::runtime_error(
                "Failed while writing detection JSONL output."
            );
        }
    }

private:
    std::ofstream output_{};
};

void print_frame_result(
    std::uint64_t frame_index,
    std::string_view source_label,
    const InferenceFrameResult& result,
    bool print_every_detection
) {
    std::cout
        << "Frame "
        << frame_index
        << " ["
        << source_label
        << "]: "
        << result.detections.size()
        << " detections";

    if (result.used_cuda_graph.has_value()) {
        std::cout
            << ", CUDA graph="
            << (*result.used_cuda_graph ? "yes" : "no");
    }

    std::cout << '\n';

    if (!print_every_detection) {
        return;
    }

    for (
        std::size_t index = 0U;
        index < result.detections.size();
        ++index
    ) {
        const Detection& detection = result.detections[index];

        std::cout
            << "  ["
            << index
            << "] "
            << bdd100k_class_name(detection.class_id)
            << " class="
            << detection.class_id
            << " confidence="
            << std::fixed
            << std::setprecision(4)
            << detection.confidence
            << " box=["
            << detection.box.x_min
            << ", "
            << detection.box.y_min
            << ", "
            << detection.box.x_max
            << ", "
            << detection.box.y_max
            << "]\n";
    }
}

[[nodiscard]] fs::path image_output_path(
    const fs::path& output_directory,
    const fs::path& source_path
) {
    std::string stem = source_path.stem().string();

    if (stem.empty()) {
        stem = "frame";
    }

    return output_directory / (stem + "_detections.jpg");
}

void save_annotated_image(
    const fs::path& output_path,
    const cv::Mat& frame,
    const DetectionBuffer& detections
) {
    fs::create_directories(output_path.parent_path());

    const cv::Mat annotated = draw_detections(frame, detections);

    if (!cv::imwrite(path_to_utf8(output_path), annotated)) {
        throw std::runtime_error(
            "OpenCV failed to write annotated image: "
            + path_to_utf8(output_path)
        );
    }
}

class VideoOutput final {
public:
    VideoOutput() = default;

    void open(
        const fs::path& output_path,
        cv::Size frame_size,
        double frames_per_second
    ) {
        fs::create_directories(output_path.parent_path());

        const double safe_fps =
            std::isfinite(frames_per_second)
                && frames_per_second > 0.0
                    ? frames_per_second
                    : 30.0;

        writer_.open(
            path_to_utf8(output_path),
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            safe_fps,
            frame_size,
            true
        );

        if (!writer_.isOpened()) {
            throw std::runtime_error(
                "Failed to open annotated video output: "
                + path_to_utf8(output_path)
            );
        }
    }

    void write(
        const cv::Mat& frame
    ) {
        if (!writer_.isOpened()) {
            return;
        }

        writer_.write(frame);
    }

    [[nodiscard]] bool is_open() const noexcept {
        return writer_.isOpened();
    }

private:
    cv::VideoWriter writer_{};
};

[[nodiscard]] fs::path stream_output_path(
    const ApplicationOptions& options,
    const SourceDescriptor& source
) {
    std::string stem;

    if (source.kind == SourceKind::Camera) {
        stem = "camera_" + std::to_string(source.camera_index);
    } else {
        stem = source.path.stem().string();
    }

    if (stem.empty()) {
        stem = "stream";
    }

    return options.output_directory / (stem + "_detections.mp4");
}

[[nodiscard]] std::string source_display_name(
    const SourceDescriptor& source
) {
    if (source.kind == SourceKind::Camera) {
        return "camera:" + std::to_string(source.camera_index);
    }

    return path_to_utf8(source.path);
}

template <typename RunFrame>
int run_image_inference(
    const ApplicationOptions& options,
    const SourceDescriptor& source,
    RunFrame&& run_frame
) {
    const cv::Mat frame = ImageProcessor::load_bgr_image(source.path);
    validate_frame(frame, path_to_utf8(source.path));

    InferenceFrameResult result = run_frame(frame);

    print_frame_result(
        0U,
        path_to_utf8(source.path),
        result,
        true
    );

    if (options.save_output) {
        fs::create_directories(options.output_directory);

        const fs::path annotated_path = image_output_path(
            options.output_directory,
            source.path
        );

        save_annotated_image(
            annotated_path,
            frame,
            result.detections
        );

        DetectionJsonlWriter jsonl{
            options.output_directory / "detections.jsonl"
        };

        jsonl.write(
            0U,
            path_to_utf8(source.path),
            result
        );

        std::cout
            << "Annotated image: "
            << path_to_utf8(annotated_path)
            << "\nDetection JSONL: "
            << path_to_utf8(
                options.output_directory / "detections.jsonl"
            )
            << '\n';
    }

    return 0;
}

template <typename RunFrame>
int run_directory_inference(
    const ApplicationOptions& options,
    const SourceDescriptor& source,
    RunFrame&& run_frame
) {
    const std::vector<fs::path> images =
        collect_image_paths(source.path);

    std::optional<DetectionJsonlWriter> jsonl{};

    if (options.save_output) {
        fs::create_directories(options.output_directory);
        jsonl.emplace(
            options.output_directory / "detections.jsonl"
        );
    }

    std::uint64_t processed = 0U;

    for (const fs::path& image_path : images) {
        if (g_stop_requested.load(std::memory_order_relaxed)) {
            break;
        }

        const cv::Mat frame = ImageProcessor::load_bgr_image(image_path);
        validate_frame(frame, path_to_utf8(image_path));

        InferenceFrameResult result = run_frame(frame);

        print_frame_result(
            processed,
            path_to_utf8(image_path),
            result,
            options.print_detections
        );

        if (options.save_output) {
            const fs::path relative = fs::relative(
                image_path,
                source.path
            );

            const fs::path parent =
                options.output_directory / relative.parent_path();

            save_annotated_image(
                image_output_path(parent, image_path),
                frame,
                result.detections
            );

            jsonl->write(
                processed,
                path_to_utf8(image_path),
                result
            );
        }

        ++processed;
    }

    std::cout << "Processed images: " << processed << '\n';
    return processed == 0U ? 2 : 0;
}

template <typename RunFrame>
int run_generic_stream_inference(
    const ApplicationOptions& options,
    const SourceDescriptor& source,
    RunFrame&& run_frame
) {
    cv::VideoCapture capture = open_capture(source);

    const double input_fps = capture.get(cv::CAP_PROP_FPS);
    VideoOutput video_output{};
    DetectionJsonlWriter jsonl{};

    if (options.save_output) {
        fs::create_directories(options.output_directory);

        jsonl.open(
            options.output_directory / "detections.jsonl"
        );
    }

    std::uint64_t frame_index = 0U;
    const std::string label = source_display_name(source);

    while (!g_stop_requested.load(std::memory_order_relaxed)) {
        if (
            options.maximum_stream_frames != 0U
            && frame_index >= options.maximum_stream_frames
        ) {
            break;
        }

        cv::Mat frame;

        if (!capture.read(frame)) {
            break;
        }

        validate_frame(frame, label);

        InferenceFrameResult result = run_frame(frame);

        if (options.print_detections || frame_index % 30U == 0U) {
            print_frame_result(
                frame_index,
                label,
                result,
                options.print_detections
            );
        }

        if (options.save_output) {
            const cv::Mat annotated = draw_detections(
                frame,
                result.detections
            );

            if (!video_output.is_open()) {
                video_output.open(
                    stream_output_path(options, source),
                    annotated.size(),
                    input_fps
                );
            }

            video_output.write(annotated);
            jsonl.write(frame_index, label, result);
        }

        ++frame_index;
    }

    std::cout << "Processed stream frames: " << frame_index << '\n';
    return frame_index == 0U ? 2 : 0;
}

void publish_fused_stream_result(
    const ApplicationOptions& options,
    const SourceDescriptor& source,
    PendingFusedFrame pending,
    GpuPipelineResult collected,
    VideoOutput& video_output,
    DetectionJsonlWriter& jsonl,
    double input_fps
) {
    InferenceFrameResult result{
        std::move(collected.detections),
        collected.used_cuda_graph
    };

    if (
        options.print_detections
        || pending.frame_index % 30U == 0U
    ) {
        print_frame_result(
            pending.frame_index,
            pending.source_label,
            result,
            options.print_detections
        );
    }

    if (!options.save_output) {
        return;
    }

    const cv::Mat annotated = draw_detections(
        pending.frame,
        result.detections
    );

    if (!video_output.is_open()) {
        video_output.open(
            stream_output_path(options, source),
            annotated.size(),
            input_fps
        );
    }

    video_output.write(annotated);

    jsonl.write(
        pending.frame_index,
        pending.source_label,
        result
    );
}

int run_fused_stream_inference(
    const ApplicationOptions& options,
    const SourceDescriptor& source,
    NativeTensorRtPipeline& pipeline
) {
    cv::VideoCapture capture = open_capture(source);

    const double input_fps = capture.get(cv::CAP_PROP_FPS);
    VideoOutput video_output{};
    DetectionJsonlWriter jsonl{};

    if (options.save_output) {
        fs::create_directories(options.output_directory);

        jsonl.open(
            options.output_directory / "detections.jsonl"
        );
    }

    const std::string label = source_display_name(source);
    std::deque<PendingFusedFrame> pending{};
    std::uint64_t submitted_frames = 0U;
    std::uint64_t completed_frames = 0U;

    const auto collect_oldest = [&]() {
        PendingFusedFrame oldest = std::move(pending.front());
        pending.pop_front();

        GpuPipelineResult collected =
            pipeline.collect(oldest.ticket);

        publish_fused_stream_result(
            options,
            source,
            std::move(oldest),
            std::move(collected),
            video_output,
            jsonl,
            input_fps
        );

        ++completed_frames;
    };

    while (!g_stop_requested.load(std::memory_order_relaxed)) {
        if (
            options.maximum_stream_frames != 0U
            && submitted_frames >= options.maximum_stream_frames
        ) {
            break;
        }

        while (pending.size() >= options.pipeline_depth) {
            collect_oldest();
        }

        cv::Mat frame;

        if (!capture.read(frame)) {
            break;
        }

        validate_frame(frame, label);

        const GpuPipelineTicket ticket = pipeline.submit(frame);

        pending.push_back(
            PendingFusedFrame{
                ticket,
                std::move(frame),
                submitted_frames,
                label
            }
        );

        ++submitted_frames;
    }

    while (!pending.empty()) {
        collect_oldest();
    }

    std::cout
        << "Submitted stream frames: "
        << submitted_frames
        << "\nCompleted stream frames: "
        << completed_frames
        << '\n';

    return completed_frames == 0U ? 2 : 0;
}

[[nodiscard]] std::vector<cv::Mat> load_benchmark_frames(
    const ApplicationOptions& options,
    const SourceDescriptor& source
) {
    if (source.kind == SourceKind::Camera) {
        throw std::invalid_argument(
            "Camera sources are not accepted for reproducible benchmarks."
        );
    }

    std::vector<cv::Mat> frames{};
    frames.reserve(options.benchmark_source_frames);

    if (source.kind == SourceKind::Image) {
        cv::Mat frame = ImageProcessor::load_bgr_image(source.path);
        validate_frame(frame, path_to_utf8(source.path));
        frames.push_back(std::move(frame));
        return frames;
    }

    if (source.kind == SourceKind::ImageDirectory) {
        const std::vector<fs::path> paths =
            collect_image_paths(source.path);

        const std::size_t count = std::min(
            options.benchmark_source_frames,
            paths.size()
        );

        for (std::size_t index = 0U; index < count; ++index) {
            cv::Mat frame =
                ImageProcessor::load_bgr_image(paths[index]);

            validate_frame(frame, path_to_utf8(paths[index]));
            frames.push_back(std::move(frame));
        }

        return frames;
    }

    cv::VideoCapture capture = open_capture(source);

    while (frames.size() < options.benchmark_source_frames) {
        cv::Mat frame;

        if (!capture.read(frame)) {
            break;
        }

        validate_frame(frame, path_to_utf8(source.path));
        frames.push_back(std::move(frame));
    }

    if (frames.empty()) {
        throw std::runtime_error(
            "No frames could be decoded from benchmark video: "
            + path_to_utf8(source.path)
        );
    }

    return frames;
}

[[nodiscard]] fs::path benchmark_run_directory(
    const ApplicationOptions& options
) {
    std::ostringstream name;

    name
        << "benchmark_"
        << backend_name(options.backend)
        << '_'
        << artifact_precision_name(options.precision)
        << '_'
        << benchmark_mode_name(options.benchmark_mode)
        << "_depth"
        << options.pipeline_depth;

    return options.output_directory / name.str();
}

int publish_benchmark_result(
    const ApplicationOptions& options,
    const BenchmarkResult& result
) {
    const std::string summary = format_benchmark_summary(result);
    std::cout << summary << '\n';

    const fs::path run_directory =
        benchmark_run_directory(options);

    fs::create_directories(run_directory);

    const fs::path summary_path = run_directory / "summary.txt";
    std::ofstream summary_file(
        summary_path,
        std::ios::binary | std::ios::trunc
    );

    if (!summary_file.is_open()) {
        throw std::runtime_error(
            "Failed to open benchmark summary output: "
            + path_to_utf8(summary_path)
        );
    }

    summary_file << summary;

    if (!summary_file) {
        throw std::runtime_error(
            "Failed while writing benchmark summary output."
        );
    }

    write_benchmark_json(
        result,
        run_directory / "benchmark.json"
    );

    write_benchmark_summary_csv(
        std::vector<BenchmarkResult>{result},
        run_directory / "summary.csv"
    );

    if (options.retain_raw_samples) {
        write_benchmark_samples_csv(
            result,
            run_directory / "samples.csv"
        );
    }

    std::cout
        << "Benchmark outputs: "
        << path_to_utf8(run_directory)
        << '\n';

    return result.successful() ? 0 : 2;
}

int run_benchmark_mode(
    const ApplicationOptions& options,
    const SourceDescriptor& source
) {
    std::vector<cv::Mat> frames =
        load_benchmark_frames(options, source);

    const BenchmarkConfig benchmark_config =
        make_benchmark_config(options);

    std::cout
        << "Preloaded benchmark frames: "
        << frames.size()
        << '\n';

    if (options.backend == BackendSelection::FusedTensorRt) {
        if (!NativeTensorRtPipeline::is_compiled()) {
            throw std::runtime_error(
                "The fused CUDA/TensorRT pipeline was not compiled."
            );
        }

        NativeTensorRtPipeline pipeline{
            make_pipeline_config(options)
        };

        print_pipeline_runtime_info(pipeline);

        const BenchmarkResult result = run_benchmark(
            pipeline,
            frames,
            benchmark_config,
            BenchmarkControl{}
        );

        return publish_benchmark_result(options, result);
    }

    EngineSelectionResult selection = select_engine(options);
    print_backend_selection(selection);
    write_backend_selection_metadata(options, selection);

    InferenceEngine& engine = *selection.engine;
    ImageProcessor processor{make_preprocess_config()};
    Postprocessor postprocessor{
        make_postprocess_config(options)
    };

    print_engine_runtime_info(engine);

    const BenchmarkResult result = run_benchmark(
        engine,
        processor,
        postprocessor,
        frames,
        benchmark_config
    );

    return publish_benchmark_result(options, result);
}

int run_inference_mode(
    const ApplicationOptions& options,
    const SourceDescriptor& source
) {
    if (options.backend == BackendSelection::FusedTensorRt) {
        if (!NativeTensorRtPipeline::is_compiled()) {
            throw std::runtime_error(
                "The fused CUDA/TensorRT pipeline was not compiled."
            );
        }

        NativeTensorRtPipeline pipeline{
            make_pipeline_config(options)
        };

        print_pipeline_runtime_info(pipeline);

        if (
            source.kind == SourceKind::Video
            || source.kind == SourceKind::Camera
        ) {
            return run_fused_stream_inference(
                options,
                source,
                pipeline
            );
        }

        const auto run_frame = [&pipeline](
            const cv::Mat& frame
        ) {
            return run_fused_frame(pipeline, frame);
        };

        if (source.kind == SourceKind::Image) {
            return run_image_inference(
                options,
                source,
                run_frame
            );
        }

        return run_directory_inference(
            options,
            source,
            run_frame
        );
    }

    EngineSelectionResult selection = select_engine(options);
    print_backend_selection(selection);
    write_backend_selection_metadata(options, selection);

    InferenceEngine& engine = *selection.engine;
    ImageProcessor processor{make_preprocess_config()};
    Postprocessor postprocessor{
        make_postprocess_config(options)
    };
    DetectionBuffer detection_workspace{};

    print_engine_runtime_info(engine);

    const auto run_frame = [
        &engine,
        &processor,
        &postprocessor,
        &detection_workspace
    ](const cv::Mat& frame) {
        return run_generic_frame(
            engine,
            processor,
            postprocessor,
            detection_workspace,
            frame
        );
    };

    if (source.kind == SourceKind::Image) {
        return run_image_inference(
            options,
            source,
            run_frame
        );
    }

    if (source.kind == SourceKind::ImageDirectory) {
        return run_directory_inference(
            options,
            source,
            run_frame
        );
    }

    return run_generic_stream_inference(
        options,
        source,
        run_frame
    );
}

}  // namespace
}  // namespace edge

int main(
    int argc,
    char** argv
) {
    std::signal(SIGINT, edge::handle_interrupt);

#if defined(SIGTERM)
    std::signal(SIGTERM, edge::handle_interrupt);
#endif

    try {
        const edge::ApplicationOptions options =
            edge::parse_arguments(argc, argv);

        if (options.show_help) {
            edge::print_usage(std::cout);
            return 0;
        }

        if (options.show_version) {
            std::cout
                << "edge-perception-cpp "
                << EDGE_PROJECT_VERSION
                << '\n';
            return 0;
        }

        edge::validate_options(options);

        const edge::SourceDescriptor source =
            edge::describe_source(options.source_argument);

        if (options.mode == edge::ApplicationMode::Benchmark) {
            return edge::run_benchmark_mode(options, source);
        }

        return edge::run_inference_mode(options, source);
    } catch (const std::exception& exception) {
        std::cerr
            << "edge_perception failed: "
            << exception.what()
            << '\n';

        return 1;
    }
}
