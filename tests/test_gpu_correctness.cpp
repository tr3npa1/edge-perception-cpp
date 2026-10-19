#include "gpu_pipeline.hpp"
#include "gpu_pipeline_test_support.hpp"
#include "benchmark.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(float actual, float expected, float tolerance,
          const std::string& label) {
    require(std::isfinite(actual) && std::isfinite(expected)
        && std::abs(actual - expected) <= tolerance,
        label + ": expected " + std::to_string(expected)
            + ", got " + std::to_string(actual));
}

template <typename Callable>
void must_throw(Callable&& callable, const std::string& label) {
    try {
        callable();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(label);
}

void compare_detections(const edge::DetectionBuffer& actual,
                        const edge::DetectionBuffer& expected,
                        float coordinate_tolerance = 0.002F,
                        float confidence_tolerance = 0.00001F) {
    require(actual.size() == expected.size(),
        "Detection count mismatch: " + std::to_string(actual.size())
            + " versus " + std::to_string(expected.size()));
    for (std::size_t row = 0U; row < actual.size(); ++row) {
        const auto& a = actual[row];
        const auto& e = expected[row];
        const std::string label = "Detection " + std::to_string(row);
        require(a.class_id == e.class_id, label + " class/order mismatch");
        near(a.confidence, e.confidence, confidence_tolerance, label + " score");
        near(a.box.x_min, e.box.x_min, coordinate_tolerance, label + " x_min");
        near(a.box.y_min, e.box.y_min, coordinate_tolerance, label + " y_min");
        near(a.box.x_max, e.box.x_max, coordinate_tolerance, label + " x_max");
        near(a.box.y_max, e.box.y_max, coordinate_tolerance, label + " y_max");
    }
}

void compare_summary(const edge::PostprocessSummary& a,
                     const edge::PostprocessSummary& e) {
    require(a.candidates_examined == e.candidates_examined
        && a.rejected_by_confidence == e.rejected_by_confidence
        && a.rejected_as_malformed == e.rejected_as_malformed
        && a.rejected_as_degenerate == e.rejected_as_degenerate
        && a.detections_written == e.detections_written
        && a.maximum_reached == e.maximum_reached,
        "CPU/CUDA postprocessing diagnostic counters differ.");
}

void compare_transform(const edge::LetterboxTransform& a,
                       const edge::LetterboxTransform& e) {
    require(a.valid() && e.valid()
        && a.original_size.width == e.original_size.width
        && a.original_size.height == e.original_size.height
        && a.resized_size.width == e.resized_size.width
        && a.resized_size.height == e.resized_size.height
        && a.pad_left == e.pad_left && a.pad_right == e.pad_right
        && a.pad_top == e.pad_top && a.pad_bottom == e.pad_bottom
        && a.scale_x == e.scale_x && a.scale_y == e.scale_y,
        "CPU/CUDA letterbox transform mismatch.");
}

cv::Mat patterned_image(int width, int height) {
    // Deliberately return a non-contiguous ROI to exercise source row pitch.
    cv::Mat storage(height + 2, width + 6, CV_8UC3, cv::Scalar{251, 253, 255});
    cv::Mat image = storage(cv::Rect{2, 1, width, height});
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image.at<cv::Vec3b>(y, x) = cv::Vec3b{
                static_cast<std::uint8_t>((x * 37 + y * 17) % 256),
                static_cast<std::uint8_t>((x * 11 + y * 53 + 67) % 256),
                static_cast<std::uint8_t>((x * 71 + y * 7 + 129) % 256)
            };
        }
    }
    return image;
}

void compare_preprocess(const cv::Mat& image,
                        const edge::PreprocessConfig& config) {
    edge::ImageProcessor processor{config};
    const std::size_t count = processor.required_element_count();
    std::vector<float> cpu(count), gpu(count);
    std::vector<std::uint16_t> cpu_half(count), gpu_half(count);
    const auto cpu_result = processor.preprocess(
        image, edge::MutableTensorView::float32(cpu.data(), count)
    );
    const auto transform = edge::testing::preprocess_bgr(
        image, config, edge::MutableTensorView::float32(gpu.data(), count)
    );
    compare_transform(transform, cpu_result.transform);
    static_cast<void>(processor.preprocess(
        image, edge::MutableTensorView::float16(cpu_half.data(), count)
    ));
    static_cast<void>(edge::testing::preprocess_bgr(
        image, config, edge::MutableTensorView::float16(gpu_half.data(), count)
    ));
    for (std::size_t index = 0U; index < count; ++index) {
        if (!std::isfinite(gpu[index])
            || std::abs(gpu[index] - cpu[index]) > 5.0e-7F) {
            near(gpu[index], cpu[index], 5.0e-7F,
                "Preprocess " + std::to_string(image.cols) + "x"
                    + std::to_string(image.rows) + " element " + std::to_string(index));
        }
        if (gpu_half[index] != cpu_half[index]) {
            require(gpu_half[index] == cpu_half[index],
                "FP16 preprocessing bit mismatch at element " + std::to_string(index)
                    + " for " + std::to_string(image.cols) + "x"
                    + std::to_string(image.rows));
        }
    }
}

void test_preprocess_kernels() {
    edge::PreprocessConfig config{};
    config.execution_policy = edge::PreprocessExecutionPolicy::Serial;
    const std::array<edge::ImageSize, 6> sizes{{
        {1280, 720}, {1279, 719}, {641, 479}, {317, 199}, {640, 640}, {1, 1}
    }};
    for (const auto size : sizes) {
        compare_preprocess(patterned_image(size.width, size.height), config);
    }

    config.output_channel_order = edge::ChannelOrder::BGR;
    config.padding_bgr = {3U, 57U, 201U};
    config.center_letterbox = false;
    config.allow_scale_up = false;
    config.pixel_scale = 0.125F;
    config.mean = {1.0F, 2.0F, 4.0F};
    config.stddev = {2.0F, 4.0F, 8.0F};
    compare_preprocess(patterned_image(317, 199), config);

    // Black pixels turn normalization into exact halfway inputs to the
    // hardware binary16 conversion, including subnormal-to-normal carry.
    config = {};
    config.mean = {-1.00048828125F, -1.00146484375F, -0x1.ffcp-15F};
    compare_preprocess(cv::Mat(640, 640, CV_8UC3, cv::Scalar{0, 0, 0}), config);
}

void test_postprocess_kernels() {
    const auto transform = edge::ImageProcessor::calculate_letterbox_transform(
        {1279, 719}, edge::PreprocessConfig{}
    );
    std::vector<float> raw(edge::kModelOutputElementCount, 0.0F);
    const std::array<std::array<float, 6>, 9> rows{{
        {{64, 180, 576, 450, 0.875F, 2}},
        {{64, 180, 576, 450, 0.75F, 2}}, // No second NMS.
        {{-32, -32, 800, 800, 0.625F, 0}}, // Clip.
        {{64, 180, 64, 450, 0.5F, 1}}, // Degenerate.
        {{64, 180, 576, 450, 0.5F, 99}}, // Invalid class.
        {{64, 180, 576, 450, 0.5F, 1.5F}}, // Nonintegral class.
        {{std::numeric_limits<float>::infinity(), 180, 576, 450, 0.5F, 1}},
        {{64, 180, 576, 450, 0.25F, 2}}, // Threshold equality.
        {{64, 180, 576, 450, 0.5F, 3}} // Only read when unsorted.
    }};
    for (std::size_t row = 0U; row < rows.size(); ++row) {
        std::copy(rows[row].begin(), rows[row].end(),
            raw.begin() + static_cast<std::ptrdiff_t>(row * edge::kDetectionValueCount));
    }
    for (const bool sorted : {false, true}) {
        edge::PostprocessConfig config{};
        config.output_sorted_by_confidence = sorted;
        edge::Postprocessor processor{config};
        edge::DetectionBuffer expected{};
        const auto summary = processor.process(
            edge::DetectionTensorView::contiguous(raw.data(), edge::kModelMaximumDetections),
            transform, expected
        );
        require(expected.size() == (sorted ? 3U : 4U),
            "Synthetic decode fixture has an unexpected accepted-row count.");
        for (const auto type : {edge::TensorElementType::Float32,
                                edge::TensorElementType::Float16}) {
            edge::DetectionBuffer actual{};
            const auto actual_summary = edge::testing::postprocess(
                raw, type, transform, config, actual
            );
            compare_detections(actual, expected);
            compare_summary(actual_summary, summary);
        }
    }
}

struct Reference {
    std::vector<float> raw;
    edge::DetectionBuffer detections{};
    edge::PostprocessSummary summary{};
    edge::LetterboxTransform transform{};
};

edge::ArtifactPrecision test_precision = edge::ArtifactPrecision::Float16;

std::array<Reference, 2> test_native_repeats(
    const std::filesystem::path& engine_path,
    const std::array<cv::Mat, 2>& images
) {
    std::array<Reference, 2> references{};
    for (const bool graph : {false, true}) {
        edge::InferenceEngineConfig config{};
        config.artifact_path = engine_path;
        config.backend = edge::InferenceBackend::NativeTensorRt;
        config.artifact_precision = test_precision;
        config.gpu.enable_cuda_graph = graph;
        edge::InferenceEngine engine{config};
        edge::ImageProcessor processor{};
        edge::Postprocessor postprocessor{};
        must_throw([&]() { static_cast<void>(engine.output_tensor()); },
            "Output must not be published before the first run.");
        for (std::size_t iteration = 0U; iteration < 6U; ++iteration) {
            const std::size_t frame = iteration % images.size();
            const auto prep = processor.preprocess(images[frame], engine.input_tensor());
            const auto output = engine.run();
            require(output.element_count == edge::kModelOutputElementCount,
                "Native output shape mismatch.");
            require(engine.has_output() && engine.completed_run_count() == iteration + 1U,
                "Native completion state mismatch.");
            // Reading immediately is intentional: run() must synchronize D2H.
            std::vector<float> copied(output.data, output.data + output.element_count);
            for (const float value : copied) {
                require(std::isfinite(value), "Native inference emitted non-finite output.");
            }
            if (!graph && iteration < images.size()) {
                references[frame].raw = copied;
                references[frame].transform = prep.transform;
                references[frame].summary = postprocessor.process(
                    output, prep.transform, references[frame].detections
                );
                require(!references[frame].detections.empty(),
                    "Choose a real fixture image with detections above 0.25.");
            } else {
                require(copied == references[frame].raw,
                    "Repeated native inference or CUDA graph returned changed/stale output.");
            }
        }
    }
    require(references[0].raw != references[1].raw,
        "The original/flipped images must produce different outputs to detect stale frames.");
    return references;
}

void test_pipeline_modes(const std::filesystem::path& engine_path,
                         const std::array<cv::Mat, 2>& images,
                         const std::array<Reference, 2>& references) {
    for (const bool gpu_postprocess : {false, true}) {
        for (const bool graph : {false, true}) {
            edge::NativeTensorRtPipelineConfig config{};
            config.engine_path = engine_path;
            config.pipeline_depth = 2U;
            config.maximum_host_image_size = {images[0].cols, images[0].rows};
            config.enable_gpu_postprocess = gpu_postprocess;
            config.enable_cuda_graph = graph;
            config.require_cuda_graph = graph;
            edge::NativeTensorRtPipeline pipeline{config};
            std::array<edge::GpuPipelineResult, 2> initial{};
            std::array<edge::GpuPipelineTicket, 2> previous_tickets{};
            for (std::size_t round = 0U; round < 3U; ++round) {
                const std::size_t first_frame = round % images.size();
                const std::size_t second_frame = 1U - first_frame;
                // Alternate the content assigned to each persistent slot so
                // graph replay cannot silently return that slot's old frame.
                cv::Mat mutable_source = images[first_frame].clone();
                const auto first = pipeline.submit(mutable_source);
                // Ordinary submit() must already own its source bytes.
                mutable_source.setTo(cv::Scalar{255, 0, 255});
                const auto second = pipeline.submit(images[second_frame]);
                require(first.valid() && second.valid()
                    && first.slot_index != second.slot_index
                    && pipeline.in_flight_count() == 2U,
                    "Two submissions must occupy distinct live slots.");
                static_cast<void>(pipeline.ready(first));
                if (round != 0U) {
                    for (const auto stale : previous_tickets) {
                        must_throw([&]() { static_cast<void>(pipeline.ready(stale)); },
                            "A recycled slot must reject its previous generation's ticket.");
                    }
                }
                must_throw([&]() { static_cast<void>(pipeline.submit(images[0])); },
                    "Submitting beyond pipeline depth must reject before overwriting a slot.");

                // Out-of-order collection must wait for the right frame and
                // leave the other slot intact, without requiring ready().
                const auto second_result = pipeline.collect(second);
                require(pipeline.in_flight_count() == 1U, "Collection must release one slot.");
                const auto first_result = pipeline.collect(first);
                require(pipeline.in_flight_count() == 0U, "All slots must be released.");
                require(first_result.sequence_number < second_result.sequence_number,
                    "Collection order must not rewrite submission sequence numbers.");
                must_throw([&]() { static_cast<void>(pipeline.collect(first)); },
                    "A collected ticket must be rejected.");
                must_throw([&]() { static_cast<void>(pipeline.ready(second)); },
                    "Readiness must reject a stale ticket.");

                const std::array<edge::GpuPipelineResult, 2> results{first_result, second_result};
                for (std::size_t position = 0U; position < results.size(); ++position) {
                    const std::size_t frame = (round + position) % images.size();
                    const auto& result = results[position];
                    compare_transform(result.transform, references[frame].transform);
                    compare_detections(result.detections, references[frame].detections);
                    compare_summary(result.summary, references[frame].summary);
                    require(!result.source_was_device_resident,
                        "Host submissions must be identified correctly.");
                    if (round == 0U) {
                        initial[frame] = result;
                    } else {
                        compare_detections(result.detections, initial[frame].detections, 0, 0);
                        require(result.used_cuda_graph == graph,
                            "Warm slots must execute the requested graph/non-graph path.");
                    }
                }
                previous_tickets = {first, second};
            }
            std::cout << "[PASS] pipeline repeats, submit/collect: graph=" << graph
                      << " gpu_postprocess=" << gpu_postprocess << '\n';
        }
    }
}

void test_native_concurrent_workers(const std::filesystem::path& engine_path,
                                   const std::array<cv::Mat, 2>& images,
                                   const std::array<Reference, 2>& references) {
    for (const bool graph : {false, true}) {
        edge::InferenceEngineConfig config{};
        config.artifact_path = engine_path;
        config.backend = edge::InferenceBackend::NativeTensorRt;
        config.artifact_precision = test_precision;
        config.gpu.enable_cuda_graph = graph;
        std::array<std::unique_ptr<edge::InferenceEngine>, 3> engines;
        // Allocate every independent engine before any thread can capture.
        for (auto& engine : engines) engine = std::make_unique<edge::InferenceEngine>(config);
        std::promise<void> release;
        const auto start = release.get_future().share();
        std::array<std::future<void>, 3> workers;
        for (std::size_t slot = 0; slot < workers.size(); ++slot) {
            workers[slot] = std::async(std::launch::async, [&, slot, start] {
                edge::ImageProcessor processor{};
                edge::Postprocessor postprocessor{};
                start.wait();
                for (std::size_t iteration = 0; iteration < 6U; ++iteration) {
                    const auto frame = (slot + iteration) % images.size();
                    auto& engine = *engines[slot];
                    const auto prep = processor.preprocess(images[frame], engine.input_tensor());
                    const auto output = engine.run();
                    const std::vector<float> raw(output.data, output.data + output.element_count);
                    require(raw == references[frame].raw,
                        "Concurrent native engine returned stale or changed raw output.");
                    edge::DetectionBuffer detections{};
                    static_cast<void>(postprocessor.process(output, prep.transform, detections));
                    compare_detections(detections, references[frame].detections, 0, 0);
                }
            });
        }
        release.set_value();
        for (auto& worker : workers) worker.get();
        std::cout << "[PASS] three concurrent native engines: exact reused-frame outputs, graph="
                  << graph << '\n';
    }
}

void test_native_benchmark_scheduler(const std::filesystem::path& engine_path,
                                    const std::array<cv::Mat, 2>& images,
                                    const std::array<Reference, 2>& references) {
    const std::vector<cv::Mat> frames{images.begin(), images.end()};
    for (const bool graph : {false, true}) {
        edge::InferenceEngineConfig engine_config{};
        engine_config.artifact_path = engine_path;
        engine_config.backend = edge::InferenceBackend::NativeTensorRt;
        engine_config.artifact_precision = test_precision;
        engine_config.gpu.enable_cuda_graph = graph;
        edge::InferenceEngine engine{engine_config};
        edge::ImageProcessor processor{};
        edge::Postprocessor postprocessor{};
        edge::BenchmarkConfig config{};
        config.mode = edge::BenchmarkMode::StreamingThroughput;
        config.timing_scope = edge::BenchmarkTimingScope::EndToEnd;
        config.warmup_iterations = 6U;
        config.measured_iterations = 12U;
        config.retain_raw_samples = true;
        config.collect_stage_timings = false;
        for (const std::size_t depth : {1U, 3U}) {
            config.pipeline_depth = depth;
            const auto result = edge::run_benchmark(engine, processor, postprocessor, frames, config);
            require(result.successful() && result.effective_pipeline_depth == depth
                && result.execution_policy == "native_per_slot_workers"
                && result.maximum_active_workers == depth,
                "Native benchmark must execute the requested real worker depth.");
            require(result.samples.size() == config.measured_iterations,
                "Native scheduler lost measured frames.");
            for (std::size_t index = 0; index < result.samples.size(); ++index) {
                const auto& sample = result.samples[index];
                require(sample.completed && sample.submitted
                    && sample.iteration_index == index
                    && sample.source_frame_index == index % images.size()
                    && sample.pipeline_slot_index == index % depth
                    && sample.detection_count == references[index % images.size()].detections.size(),
                    "Native scheduler returned mismatched frame/slot/result metadata.");
            }
        }
        bool cancel = false;
        edge::BenchmarkControl control{};
        control.should_cancel = [&] { return cancel; };
        control.on_progress = [&](const edge::BenchmarkProgress& progress) {
            if (progress.phase == edge::BenchmarkPhase::Measurement && progress.in_flight == 3U)
                cancel = true;
        };
        const auto cancelled = edge::run_benchmark(engine, processor, postprocessor, frames, config, control);
        require(cancelled.status == edge::BenchmarkCompletionStatus::Cancelled
            && cancelled.counts.measurement.attempted == 3U
            && cancelled.counts.measurement.completed == 3U,
            "Cancellation must drain all accepted native jobs without scheduling more.");
        control.should_cancel = {};
        control.on_progress = [](const edge::BenchmarkProgress& progress) {
            if (progress.phase == edge::BenchmarkPhase::Measurement && progress.in_flight == 3U)
                throw std::runtime_error("Intentional callback exception.");
        };
        must_throw([&] { static_cast<void>(edge::run_benchmark(
            engine, processor, postprocessor, frames, config, control)); },
            "Callback exceptions must propagate after worker cleanup.");
        const auto output = engine.run();
        require(std::vector<float>(output.data, output.data + output.element_count) == references[0].raw,
            "Callback cleanup did not safely finish/release the borrowed engine.");
        std::cout << "[PASS] native benchmark depths 1/3, cancellation and callback cleanup, graph="
                  << graph << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path engine_path, image_path;
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--int8") {
                test_precision = edge::ArtifactPrecision::Int8;
                continue;
            }
            if ((option == "--engine" || option == "--image") && index + 1 < argc) {
                (option == "--engine" ? engine_path : image_path) = argv[++index];
            } else {
                throw std::invalid_argument("Expected --engine <fp16.engine> --image <fixture>.");
            }
        }
        require(std::filesystem::is_regular_file(engine_path)
            && std::filesystem::is_regular_file(image_path),
            "GPU correctness tests require explicit existing --engine and --image artifacts.");
        std::array<cv::Mat, 2> images{};
        images[0] = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        require(!images[0].empty(), "Could not decode the fixture image.");
        cv::flip(images[0], images[1], 1);
        test_preprocess_kernels();
        std::cout << "[PASS] CPU/CUDA preprocessing, interpolation, layout and half rounding\n";
        test_postprocess_kernels();
        std::cout << "[PASS] FP32/FP16 CUDA decoding and coordinate restoration\n";
        const auto references = test_native_repeats(engine_path, images);
        std::cout << "[PASS] native synchronous output and deterministic graph/non-graph repeats\n";
        test_pipeline_modes(engine_path, images, references);
        test_native_concurrent_workers(engine_path, images, references);
        test_native_benchmark_scheduler(engine_path, images, references);
        std::cout << "All GPU correctness regressions passed. Tiny scheduler smoke timings are not benchmark results.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] GPU correctness: " << error.what() << '\n';
        return 1;
    }
}
