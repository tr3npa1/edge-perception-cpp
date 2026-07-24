#include "benchmark.hpp"
#include "image_processor.hpp"
#include "inference_engine.hpp"
#include "postprocess.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

class TestFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw TestFailure{std::string{message}};
    }
}

void require_near(
    double actual,
    double expected,
    double tolerance,
    std::string_view message
) {
    if (
        !std::isfinite(actual)
        || !std::isfinite(expected)
        || std::abs(actual - expected) > tolerance
    ) {
        throw TestFailure{
            std::string{message}
            + ": expected "
            + std::to_string(expected)
            + ", got "
            + std::to_string(actual)
        };
    }
}

template <typename Callable>
void require_throws(
    Callable&& callable,
    std::string_view message
) {
    try {
        std::forward<Callable>(callable)();
    } catch (const std::exception&) {
        return;
    }

    throw TestFailure{std::string{message}};
}

void test_letterbox_geometry() {
    edge::PreprocessConfig config{};
    config.network_size = edge::ImageSize{640, 640};
    config.allow_scale_up = true;
    config.center_letterbox = true;

    const edge::LetterboxTransform transform =
        edge::ImageProcessor::calculate_letterbox_transform(
            edge::ImageSize{1280, 720},
            config
        );

    require(transform.valid(), "Letterbox transform must be valid.");
    require(
        transform.resized_size.width == 640
            && transform.resized_size.height == 360,
        "Landscape image must resize to 640x360."
    );
    require(
        transform.pad_left == 0
            && transform.pad_right == 0
            && transform.pad_top == 140
            && transform.pad_bottom == 140,
        "Landscape image must receive centered vertical padding."
    );
    require_near(transform.scale_x, 0.5, 1.0e-6, "scale_x mismatch");
    require_near(transform.scale_y, 0.5, 1.0e-6, "scale_y mismatch");

    const float original_x = 321.5F;
    const float original_y = 210.25F;

    require_near(
        transform.network_to_original_x(
            transform.original_to_network_x(original_x)
        ),
        original_x,
        1.0e-4,
        "X coordinate round-trip mismatch"
    );
    require_near(
        transform.network_to_original_y(
            transform.original_to_network_y(original_y)
        ),
        original_y,
        1.0e-4,
        "Y coordinate round-trip mismatch"
    );
}

void test_preprocess_rgb_nchw() {
    edge::PreprocessConfig config{};
    config.network_size = edge::ImageSize{2, 2};
    config.execution_policy =
        edge::PreprocessExecutionPolicy::Serial;

    edge::ImageProcessor processor{config};

    cv::Mat image(2, 2, CV_8UC3);
    image.at<cv::Vec3b>(0, 0) = cv::Vec3b{10U, 20U, 30U};
    image.at<cv::Vec3b>(0, 1) = cv::Vec3b{40U, 50U, 60U};
    image.at<cv::Vec3b>(1, 0) = cv::Vec3b{70U, 80U, 90U};
    image.at<cv::Vec3b>(1, 1) = cv::Vec3b{100U, 110U, 120U};

    std::vector<float> tensor(processor.required_element_count(), -1.0F);

    const edge::PreprocessResult result = processor.preprocess(
        image,
        edge::MutableTensorView::float32(
            tensor.data(),
            tensor.size()
        )
    );

    require(
        result.tensor_shape
            == std::array<std::int64_t, 4>{1, 3, 2, 2},
        "Unexpected preprocessing tensor shape."
    );
    require(result.transform.valid(), "Preprocess transform must be valid.");

    const std::vector<float> expected{
        30.0F / 255.0F,
        60.0F / 255.0F,
        90.0F / 255.0F,
        120.0F / 255.0F,
        20.0F / 255.0F,
        50.0F / 255.0F,
        80.0F / 255.0F,
        110.0F / 255.0F,
        10.0F / 255.0F,
        40.0F / 255.0F,
        70.0F / 255.0F,
        100.0F / 255.0F
    };

    require(tensor.size() == expected.size(), "Tensor size mismatch.");

    for (std::size_t index = 0U; index < tensor.size(); ++index) {
        require_near(
            tensor[index],
            expected[index],
            1.0e-6,
            "NCHW channel conversion mismatch"
        );
    }
}

void test_postprocess_validation_and_restoration() {
    edge::PreprocessConfig preprocess{};
    const edge::LetterboxTransform transform =
        edge::ImageProcessor::calculate_letterbox_transform(
            edge::ImageSize{1280, 720},
            preprocess
        );

    edge::PostprocessConfig config{};
    config.confidence_threshold = 0.25F;
    config.output_sorted_by_confidence = true;
    config.malformed_candidate_policy =
        edge::MalformedCandidatePolicy::Discard;

    edge::Postprocessor postprocessor{config};

    const std::vector<float> output{
        // Valid: original box [100, 120, 300, 320].
        50.0F, 200.0F, 150.0F, 300.0F, 0.95F, 2.0F,
        // Invalid class ID.
        10.0F, 10.0F, 40.0F, 40.0F, 0.90F, 99.0F,
        // Degenerate after restoration.
        100.0F, 100.0F, 100.0F, 200.0F, 0.85F, 1.0F,
        // Valid but deliberately outside image bounds; must be clipped.
        -100.0F, -100.0F, 800.0F, 800.0F, 0.80F, 0.0F,
        // First low-confidence row; sorted output allows early termination.
        0.0F, 0.0F, 0.0F, 0.0F, 0.10F, 0.0F
    };

    edge::DetectionBuffer detections{};
    const edge::PostprocessSummary summary = postprocessor.process(
        edge::DetectionTensorView::contiguous(output.data(), 5U),
        transform,
        detections
    );

    require(detections.size() == 2U, "Expected two valid detections.");
    require(
        summary.rejected_as_malformed == 1U,
        "Expected one malformed candidate."
    );
    require(
        summary.rejected_as_degenerate == 1U,
        "Expected one degenerate candidate."
    );
    require(
        summary.rejected_by_confidence == 1U,
        "Expected one confidence rejection."
    );

    const edge::Detection& first = detections[0U];
    require(first.class_id == 2, "First class ID mismatch.");
    require_near(first.confidence, 0.95, 1.0e-6, "First confidence mismatch");
    require_near(first.box.x_min, 100.0, 1.0e-4, "x_min restore mismatch");
    require_near(first.box.y_min, 120.0, 1.0e-4, "y_min restore mismatch");
    require_near(first.box.x_max, 300.0, 1.0e-4, "x_max restore mismatch");
    require_near(first.box.y_max, 320.0, 1.0e-4, "y_max restore mismatch");

    const edge::Detection& second = detections[1U];
    require(second.class_id == 0, "Second class ID mismatch.");
    require_near(second.box.x_min, 0.0, 1.0e-6, "Clipped x_min mismatch");
    require_near(second.box.y_min, 0.0, 1.0e-6, "Clipped y_min mismatch");
    require_near(second.box.x_max, 1280.0, 1.0e-6, "Clipped x_max mismatch");
    require_near(second.box.y_max, 720.0, 1.0e-6, "Clipped y_max mismatch");

    require_throws(
        [&postprocessor, &transform]() {
            edge::DetectionBuffer buffer{};
            postprocessor.process(
                edge::DetectionTensorView{
                    nullptr,
                    6U,
                    1U,
                    edge::kDetectionValueCount
                },
                transform,
                buffer
            );
        },
        "Null non-empty output view must throw."
    );
}

void test_benchmark_statistics() {
    const std::vector<edge::BenchmarkDuration> samples{
        edge::BenchmarkDuration{1.0},
        edge::BenchmarkDuration{2.0},
        edge::BenchmarkDuration{3.0},
        edge::BenchmarkDuration{4.0}
    };

    const auto statistics =
        edge::calculate_latency_statistics(samples);

    require(statistics.has_value(), "Statistics must exist.");
    require(statistics->sample_count == 4U, "Sample count mismatch.");
    require_near(statistics->minimum.count(), 1.0, 1.0e-12, "Minimum mismatch");
    require_near(statistics->maximum.count(), 4.0, 1.0e-12, "Maximum mismatch");
    require_near(statistics->mean.count(), 2.5, 1.0e-12, "Mean mismatch");
    require_near(statistics->median.count(), 2.5, 1.0e-12, "Median mismatch");
    require_near(statistics->p90.count(), 3.7, 1.0e-12, "p90 mismatch");
    require_near(statistics->p95.count(), 3.85, 1.0e-12, "p95 mismatch");
    require_near(statistics->p99.count(), 3.97, 1.0e-12, "p99 mismatch");

    require(
        !edge::calculate_latency_statistics({}).has_value(),
        "Empty duration series must return nullopt."
    );

    require_throws(
        []() {
            const std::vector<edge::BenchmarkDuration> invalid{
                edge::BenchmarkDuration{-1.0}
            };
            static_cast<void>(
                edge::calculate_latency_statistics(invalid)
            );
        },
        "Negative benchmark duration must throw."
    );
}

void test_provider_discovery() {
    const std::vector<std::string> providers =
        edge::InferenceEngine::available_onnxruntime_providers();

    require(!providers.empty(), "ORT provider list must not be empty.");
    require(
        edge::InferenceEngine::onnxruntime_provider_is_available(
            "CPUExecutionProvider"
        ),
        "CPUExecutionProvider must be available."
    );
}

void test_ort_cpu_end_to_end(const fs::path& model_path) {
    require(
        fs::is_regular_file(model_path),
        "Generated ONNX integration model is missing."
    );

    edge::InferenceEngineConfig engine_config{};
    engine_config.artifact_path = model_path;
    engine_config.backend = edge::InferenceBackend::OrtCpu;
    engine_config.artifact_precision =
        edge::ArtifactPrecision::Float32;
    engine_config.validate();

    edge::InferenceEngine engine{engine_config};
    edge::ImageProcessor processor{};
    edge::Postprocessor postprocessor{};

    cv::Mat image(640, 640, CV_8UC3, cv::Scalar{0, 0, 0});

    const edge::PreprocessResult preprocess_result =
        processor.preprocess(image, engine.input_tensor());

    const edge::DetectionTensorView output = engine.run();

    require(engine.has_output(), "Engine must report a completed output.");
    require(
        engine.completed_run_count() == 1U,
        "Completed-run counter mismatch."
    );
    require(
        output.candidate_count == edge::kModelMaximumDetections,
        "Integration output candidate count mismatch."
    );

    edge::DetectionBuffer detections{};
    const edge::PostprocessSummary summary = postprocessor.process(
        output,
        preprocess_result.transform,
        detections
    );

    require(
        summary.detections_written == 2U,
        "Generated model must produce two detections."
    );
    require(detections.size() == 2U, "Detection count mismatch.");

    require(detections[0U].class_id == 2, "First integration class mismatch.");
    require_near(
        detections[0U].confidence,
        0.90,
        1.0e-6,
        "First integration confidence mismatch"
    );
    require_near(
        detections[0U].box.x_min,
        100.0,
        1.0e-4,
        "First integration box mismatch"
    );

    require(detections[1U].class_id == 0, "Second integration class mismatch.");
    require_near(
        detections[1U].confidence,
        0.80,
        1.0e-6,
        "Second integration confidence mismatch"
    );
}

struct Options {
    bool run_unit = false;
    bool run_integration = false;
    fs::path model_path{};
};

Options parse_options(int argc, char** argv) {
    Options options{};

    for (int index = 1; index < argc; ++index) {
        const std::string argument{argv[index]};

        if (argument == "--unit") {
            options.run_unit = true;
            continue;
        }

        if (argument == "--integration") {
            options.run_integration = true;
            continue;
        }

        if (argument == "--model") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--model requires a path.");
            }

            options.model_path = argv[++index];
            continue;
        }

        throw std::invalid_argument(
            "Unknown test option: " + argument
        );
    }

    if (!options.run_unit && !options.run_integration) {
        options.run_unit = true;
        options.run_integration = true;
    }

    if (options.run_integration && options.model_path.empty()) {
        throw std::invalid_argument(
            "--integration requires --model <generated.onnx>."
        );
    }

    return options;
}

using TestFunction = std::function<void()>;

int run_test(std::string_view name, const TestFunction& function) {
    try {
        function();
        std::cout << "[PASS] " << name << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "[FAIL] "
            << name
            << ": "
            << exception.what()
            << '\n';
        return 1;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int failures = 0;

        if (options.run_unit) {
            failures += run_test(
                "letterbox_geometry",
                test_letterbox_geometry
            );
            failures += run_test(
                "preprocess_rgb_nchw",
                test_preprocess_rgb_nchw
            );
            failures += run_test(
                "postprocess_validation_and_restoration",
                test_postprocess_validation_and_restoration
            );
            failures += run_test(
                "benchmark_statistics",
                test_benchmark_statistics
            );
            failures += run_test(
                "onnxruntime_provider_discovery",
                test_provider_discovery
            );
        }

        if (options.run_integration) {
            failures += run_test(
                "ort_cpu_end_to_end",
                [&options]() {
                    test_ort_cpu_end_to_end(options.model_path);
                }
            );
        }

        if (failures != 0) {
            std::cerr << failures << " test(s) failed.\n";
            return 1;
        }

        std::cout << "All selected tests passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "Test runner configuration failed: "
            << exception.what()
            << '\n';
        return 2;
    }
}
