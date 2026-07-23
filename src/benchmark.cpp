#include "benchmark.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>

namespace edge {
namespace {

using BenchmarkClock = std::chrono::steady_clock;
using BenchmarkTimePoint = BenchmarkClock::time_point;

constexpr std::size_t kMinimumPipelineDepth = 1U;
constexpr std::size_t kMaximumPipelineDepth = 8U;

[[nodiscard]] bool supported_benchmark_mode(
    BenchmarkMode mode
) noexcept {
    switch (mode) {
        case BenchmarkMode::SynchronousLatency:
        case BenchmarkMode::StreamingThroughput:
            return true;
    }

    return false;
}

[[nodiscard]] bool supported_timing_scope(
    BenchmarkTimingScope scope
) noexcept {
    switch (scope) {
        case BenchmarkTimingScope::BackendExecution:
        case BenchmarkTimingScope::EndToEnd:
            return true;
    }

    return false;
}

[[nodiscard]] bool supported_failure_policy(
    BenchmarkFailurePolicy policy
) noexcept {
    switch (policy) {
        case BenchmarkFailurePolicy::StopOnFirstFailure:
        case BenchmarkFailurePolicy::ContinueAfterFailure:
            return true;
    }

    return false;
}

[[nodiscard]] BenchmarkDuration elapsed(
    BenchmarkTimePoint start,
    BenchmarkTimePoint end
) noexcept {
    return std::chrono::duration_cast<BenchmarkDuration>(end - start);
}

[[nodiscard]] bool finite_non_negative(
    BenchmarkDuration duration
) noexcept {
    return std::isfinite(duration.count()) && duration.count() >= 0.0;
}

void validate_source_frames(
    const std::vector<cv::Mat>& frames
) {
    if (frames.empty()) {
        throw std::invalid_argument(
            "Benchmark requires at least one preloaded source frame."
        );
    }

    for (std::size_t index = 0U; index < frames.size(); ++index) {
        const cv::Mat& frame = frames[index];

        if (
            frame.empty()
            || frame.dims != 2
            || frame.type() != CV_8UC3
            || frame.data == nullptr
        ) {
            throw std::invalid_argument(
                "Benchmark frame " + std::to_string(index)
                + " must be a non-empty two-dimensional CV_8UC3 BGR image."
            );
        }
    }
}

[[nodiscard]] const cv::Mat& select_frame(
    const std::vector<cv::Mat>& frames,
    std::uint64_t iteration,
    std::size_t& source_index
) noexcept {
    source_index = static_cast<std::size_t>(
        iteration % static_cast<std::uint64_t>(frames.size())
    );

    return frames[source_index];
}

[[nodiscard]] bool cancellation_requested(
    const BenchmarkControl& control
) {
    return control.should_cancel
        ? control.should_cancel()
        : false;
}

void report_progress(
    const BenchmarkControl& control,
    BenchmarkPhase phase,
    const BenchmarkPhaseCounts& counts,
    std::size_t in_flight,
    std::size_t effective_pipeline_depth
) {
    if (!control.on_progress) {
        return;
    }

    control.on_progress(
        BenchmarkProgress{
            phase,
            counts,
            in_flight,
            effective_pipeline_depth
        }
    );
}

void record_failure(
    BenchmarkResult& result,
    BenchmarkPhase phase,
    std::uint64_t iteration_index,
    std::optional<std::size_t> source_frame_index,
    std::optional<std::uint64_t> target_sequence_number,
    std::string operation,
    std::string message
) {
    if (
        result.failures.size()
        < result.config.maximum_recorded_failures
    ) {
        result.failures.push_back(
            BenchmarkFailure{
                phase,
                iteration_index,
                source_frame_index,
                target_sequence_number,
                std::move(operation),
                std::move(message)
            }
        );
    } else {
        ++result.omitted_failure_count;
    }
}

void add_warning_once(
    BenchmarkResult& result,
    std::string warning
) {
    if (
        std::find(
            result.warnings.begin(),
            result.warnings.end(),
            warning
        ) == result.warnings.end()
    ) {
        result.warnings.push_back(std::move(warning));
    }
}

[[nodiscard]] std::string exception_message(
    const std::exception& exception
) {
    const char* const message = exception.what();

    return message != nullptr && message[0] != '\0'
        ? std::string{message}
        : std::string{"Unknown standard-library exception."};
}

struct DurationSeries final {
    std::vector<BenchmarkDuration> primary{};
    std::vector<BenchmarkDuration> queue_waiting{};
    std::vector<BenchmarkDuration> submission{};
    std::vector<BenchmarkDuration> preprocessing{};
    std::vector<BenchmarkDuration> backend_execution{};
    std::vector<BenchmarkDuration> postprocessing{};
    std::vector<BenchmarkDuration> host_to_device{};
    std::vector<BenchmarkDuration> device_execution{};
    std::vector<BenchmarkDuration> device_to_host{};
    std::vector<BenchmarkDuration> completion_wait{};
    std::vector<BenchmarkDuration> end_to_end{};

    void reserve(std::size_t sample_count) {
        primary.reserve(sample_count);
        queue_waiting.reserve(sample_count);
        submission.reserve(sample_count);
        preprocessing.reserve(sample_count);
        backend_execution.reserve(sample_count);
        postprocessing.reserve(sample_count);
        host_to_device.reserve(sample_count);
        device_execution.reserve(sample_count);
        device_to_host.reserve(sample_count);
        completion_wait.reserve(sample_count);
        end_to_end.reserve(sample_count);
    }
};

void append_optional_duration(
    std::vector<BenchmarkDuration>& destination,
    const std::optional<BenchmarkDuration>& value
) {
    if (value.has_value()) {
        destination.push_back(*value);
    }
}

void append_completed_sample(
    DurationSeries& series,
    const BenchmarkSample& sample,
    BenchmarkTimingScope primary_scope
) {
    if (!sample.completed) {
        return;
    }

    const std::optional<BenchmarkDuration>& primary =
        primary_scope == BenchmarkTimingScope::BackendExecution
            ? sample.timings.backend_execution
            : sample.timings.end_to_end;

    append_optional_duration(series.primary, primary);
    append_optional_duration(
        series.queue_waiting,
        sample.timings.queue_waiting
    );
    append_optional_duration(series.submission, sample.timings.submission);
    append_optional_duration(
        series.preprocessing,
        sample.timings.preprocessing
    );
    append_optional_duration(
        series.backend_execution,
        sample.timings.backend_execution
    );
    append_optional_duration(
        series.postprocessing,
        sample.timings.postprocessing
    );
    append_optional_duration(
        series.host_to_device,
        sample.timings.host_to_device
    );
    append_optional_duration(
        series.device_execution,
        sample.timings.device_execution
    );
    append_optional_duration(
        series.device_to_host,
        sample.timings.device_to_host
    );
    append_optional_duration(
        series.completion_wait,
        sample.timings.completion_wait
    );
    append_optional_duration(series.end_to_end, sample.timings.end_to_end);
}

void finalize_statistics(
    BenchmarkResult& result,
    const DurationSeries& series
) {
    result.primary_latency = calculate_latency_statistics(series.primary);

    result.stage_statistics.queue_waiting =
        calculate_latency_statistics(series.queue_waiting);
    result.stage_statistics.submission =
        calculate_latency_statistics(series.submission);
    result.stage_statistics.preprocessing =
        calculate_latency_statistics(series.preprocessing);
    result.stage_statistics.backend_execution =
        calculate_latency_statistics(series.backend_execution);
    result.stage_statistics.postprocessing =
        calculate_latency_statistics(series.postprocessing);
    result.stage_statistics.host_to_device =
        calculate_latency_statistics(series.host_to_device);
    result.stage_statistics.device_execution =
        calculate_latency_statistics(series.device_execution);
    result.stage_statistics.device_to_host =
        calculate_latency_statistics(series.device_to_host);
    result.stage_statistics.completion_wait =
        calculate_latency_statistics(series.completion_wait);
    result.stage_statistics.end_to_end =
        calculate_latency_statistics(series.end_to_end);
}

void finalize_throughput(
    BenchmarkResult& result
) {
    if (
        !result.phase_durations.measurement.has_value()
        || result.counts.measurement.completed == 0U
    ) {
        result.throughput.reset();
        return;
    }

    const BenchmarkDuration wall_time =
        *result.phase_durations.measurement;

    if (!finite_non_negative(wall_time) || wall_time.count() <= 0.0) {
        add_warning_once(
            result,
            "Measured wall time was non-positive; throughput is unavailable."
        );
        result.throughput.reset();
        return;
    }

    const double seconds = wall_time.count() / 1000.0;
    const double frames_per_second =
        static_cast<double>(result.counts.measurement.completed) / seconds;

    if (!std::isfinite(frames_per_second)) {
        add_warning_once(
            result,
            "Calculated throughput was non-finite and was not published."
        );
        result.throughput.reset();
        return;
    }

    result.throughput = ThroughputStatistics{
        result.counts.measurement.completed,
        wall_time,
        frames_per_second
    };
}

[[nodiscard]] BenchmarkResult make_generic_result(
    InferenceEngine& engine,
    ImageProcessor& image_processor,
    const Postprocessor& postprocessor,
    const BenchmarkConfig& config,
    const BenchmarkControl& control
) {
    BenchmarkResult result{};
    result.config = config;
    result.target = InferenceEngineBenchmarkTargetInfo{
        engine.config(),
        engine.runtime_info(),
        image_processor.config(),
        postprocessor.config()
    };
    result.effective_pipeline_depth = 1U;
    result.counts.warmup.requested = config.warmup_iterations;
    result.counts.measurement.requested = config.measured_iterations;

    if (
        config.mode == BenchmarkMode::StreamingThroughput
        && config.pipeline_depth != 1U
    ) {
        add_warning_once(
            result,
            "The generic InferenceEngine is synchronous and not safe for "
            "concurrent run() calls; effective pipeline depth is one."
        );
    }

    if (
        inference_backend_uses_gpu(engine.runtime_info().backend)
        && config.collect_stage_timings
    ) {
        add_warning_once(
            result,
            "Generic GPU backend timing exposes only the synchronous public "
            "run() boundary; separate H2D, device execution, and D2H metrics "
            "are unavailable."
        );
    }

    if (control.on_progress || control.should_cancel) {
        add_warning_once(
            result,
            "Benchmark callbacks execute on the benchmark thread and may "
            "perturb publishable latency or throughput measurements."
        );
    }

    return result;
}

void validate_generic_target_contract(
    const InferenceEngine& engine,
    const ImageProcessor& image_processor
) {
    const InferenceRuntimeInfo& runtime_info = engine.runtime_info();

    if (image_processor.tensor_shape() != runtime_info.model.input_shape) {
        throw std::invalid_argument(
            "ImageProcessor tensor shape does not match the loaded model input."
        );
    }

    if (
        image_processor.required_element_count()
        != runtime_info.model.input_element_count
    ) {
        throw std::invalid_argument(
            "ImageProcessor element count does not match the loaded model input."
        );
    }

    if (
        image_processor.required_byte_count(engine.input_element_type())
        != runtime_info.model.input_byte_count
    ) {
        throw std::invalid_argument(
            "ImageProcessor byte count does not match the loaded model input."
        );
    }
}

struct GenericFrameOutcome final {
    BenchmarkSample sample{};
    std::string operation{};
    std::string failure_message{};
};

[[nodiscard]] GenericFrameOutcome execute_generic_frame(
    InferenceEngine& engine,
    ImageProcessor& image_processor,
    const Postprocessor& postprocessor,
    const cv::Mat& frame,
    std::uint64_t iteration_index,
    std::size_t source_frame_index,
    const BenchmarkConfig& config,
    bool measure
) {
    GenericFrameOutcome outcome{};
    outcome.sample.iteration_index = iteration_index;
    outcome.sample.source_frame_index = source_frame_index;

    DetectionBuffer detections{};
    PreprocessResult preprocess_result{};
    const bool collect_stages = measure && config.collect_stage_timings;
    const BenchmarkTimePoint end_to_end_start = measure
        ? BenchmarkClock::now()
        : BenchmarkTimePoint{};

    try {
        outcome.operation = "preprocess";

        const BenchmarkTimePoint preprocess_start = collect_stages
            ? BenchmarkClock::now()
            : BenchmarkTimePoint{};

        preprocess_result = image_processor.preprocess(
            frame,
            engine.input_tensor()
        );

        const BenchmarkTimePoint preprocess_end = collect_stages
            ? BenchmarkClock::now()
            : BenchmarkTimePoint{};

        if (collect_stages) {
            outcome.sample.timings.preprocessing = elapsed(
                preprocess_start,
                preprocess_end
            );
        }

        outcome.operation = "inference";

        const BenchmarkTimePoint backend_start = measure
            ? BenchmarkClock::now()
            : BenchmarkTimePoint{};

        const DetectionTensorView output = engine.run();
        outcome.sample.submitted = true;

        const BenchmarkTimePoint backend_end = measure
            ? BenchmarkClock::now()
            : BenchmarkTimePoint{};

        if (
            measure
            && (
                config.timing_scope
                    == BenchmarkTimingScope::BackendExecution
                || collect_stages
            )
        ) {
            outcome.sample.timings.backend_execution = elapsed(
                backend_start,
                backend_end
            );
        }

        outcome.operation = "postprocess";

        const BenchmarkTimePoint postprocess_start = collect_stages
            ? BenchmarkClock::now()
            : BenchmarkTimePoint{};

        const PostprocessSummary summary = postprocessor.process(
            output,
            preprocess_result.transform,
            detections
        );

        const BenchmarkTimePoint complete_time = measure
            ? BenchmarkClock::now()
            : BenchmarkTimePoint{};

        if (summary.detections_written != detections.size()) {
            throw std::runtime_error(
                "Postprocess summary and DetectionBuffer size disagree."
            );
        }

        if (collect_stages) {
            outcome.sample.timings.postprocessing = elapsed(
                postprocess_start,
                complete_time
            );
        }

        if (
            measure
            && (
                config.timing_scope == BenchmarkTimingScope::EndToEnd
                || collect_stages
            )
        ) {
            outcome.sample.timings.end_to_end = elapsed(
                end_to_end_start,
                complete_time
            );
        }

        outcome.sample.detection_count = static_cast<std::uint32_t>(
            detections.size()
        );
        outcome.sample.completed = true;
        outcome.operation.clear();
    } catch (const std::exception& exception) {
        outcome.failure_message = exception_message(exception);
    } catch (...) {
        outcome.failure_message = "Unknown non-standard exception.";
    }

    return outcome;
}

[[nodiscard]] bool process_generic_phase(
    InferenceEngine& engine,
    ImageProcessor& image_processor,
    const Postprocessor& postprocessor,
    const std::vector<cv::Mat>& frames,
    BenchmarkPhase phase,
    BenchmarkPhaseCounts& counts,
    BenchmarkResult& result,
    DurationSeries& duration_series,
    const BenchmarkControl& control,
    bool measured,
    BenchmarkCompletionStatus& stop_status
) {
    std::optional<BenchmarkTimePoint> phase_start{};
    std::optional<BenchmarkTimePoint> last_terminal_time{};

    report_progress(
        control,
        phase,
        counts,
        0U,
        result.effective_pipeline_depth
    );

    for (std::uint64_t iteration = 0U;
         iteration < counts.requested;
         ++iteration) {
        if (cancellation_requested(control)) {
            stop_status = BenchmarkCompletionStatus::Cancelled;
            break;
        }

        if (!phase_start.has_value()) {
            phase_start = BenchmarkClock::now();
        }

        ++counts.attempted;

        std::size_t source_index = 0U;
        const cv::Mat& frame = select_frame(
            frames,
            iteration,
            source_index
        );

        GenericFrameOutcome outcome = execute_generic_frame(
            engine,
            image_processor,
            postprocessor,
            frame,
            iteration,
            source_index,
            result.config,
            measured
        );

        last_terminal_time = BenchmarkClock::now();

        if (outcome.sample.submitted) {
            ++counts.submitted;
        }

        if (outcome.sample.completed) {
            ++counts.completed;

            if (measured) {
                append_completed_sample(
                    duration_series,
                    outcome.sample,
                    result.config.timing_scope
                );
            }
        } else {
            ++counts.failed;

            record_failure(
                result,
                phase,
                iteration,
                source_index,
                std::nullopt,
                outcome.operation.empty()
                    ? std::string{"frame"}
                    : outcome.operation,
                outcome.failure_message.empty()
                    ? std::string{"Frame operation failed."}
                    : outcome.failure_message
            );
        }

        const bool frame_completed = outcome.sample.completed;

        if (measured && result.config.retain_raw_samples) {
            result.samples.push_back(std::move(outcome.sample));
        }

        report_progress(
            control,
            phase,
            counts,
            0U,
            result.effective_pipeline_depth
        );

        if (
            !frame_completed
            && result.config.failure_policy
                == BenchmarkFailurePolicy::StopOnFirstFailure
        ) {
            stop_status = BenchmarkCompletionStatus::StoppedOnFailure;
            break;
        }
    }

    if (phase_start.has_value() && last_terminal_time.has_value()) {
        const BenchmarkDuration phase_duration = elapsed(
            *phase_start,
            *last_terminal_time
        );

        if (measured) {
            result.phase_durations.measurement = phase_duration;
        } else {
            result.phase_durations.warmup = phase_duration;
        }
    }

    return stop_status == BenchmarkCompletionStatus::NotStarted;
}

[[nodiscard]] BenchmarkResult make_fused_result(
    NativeTensorRtPipeline& pipeline,
    const BenchmarkConfig& config,
    const BenchmarkControl& control
) {
    BenchmarkResult result{};
    result.config = config;
    result.target = NativeTensorRtPipelineBenchmarkTargetInfo{
        pipeline.config(),
        pipeline.runtime_info()
    };
    result.effective_pipeline_depth =
        config.mode == BenchmarkMode::SynchronousLatency
            ? 1U
            : config.pipeline_depth;
    result.counts.warmup.requested = config.warmup_iterations;
    result.counts.measurement.requested = config.measured_iterations;

    if (config.collect_stage_timings) {
        add_warning_once(
            result,
            "The fused pipeline exposes end-to-end completion but not separate "
            "H2D, preprocessing, TensorRT-kernel, postprocessing, or D2H event "
            "boundaries through its public API."
        );
    }

    if (control.on_progress || control.should_cancel) {
        add_warning_once(
            result,
            "Benchmark callbacks execute on the benchmark thread and may "
            "perturb publishable latency or throughput measurements."
        );
    }

    const NativeTensorRtPipelineConfig& pipeline_config = pipeline.config();

    if (pipeline_config.enable_cuda_graph) {
        const std::uint64_t physical_depth = static_cast<std::uint64_t>(
            pipeline.pipeline_depth()
        );

        if (
            physical_depth
                <= std::numeric_limits<std::uint64_t>::max() / 2U
            && config.warmup_iterations < physical_depth * 2U
        ) {
            add_warning_once(
                result,
                "CUDA graphs are enabled, but warmup_iterations is less than "
                "twice the physical pipeline depth; measured samples may "
                "include per-slot graph capture."
            );
        }
    }

    if (
        !pipeline_config.enable_gpu_postprocess
        || !pipeline_config.enable_cuda_graph
    ) {
        add_warning_once(
            result,
            "The supplied fused pipeline is not using the normal fully "
            "optimized configuration because GPU postprocessing or CUDA graph "
            "execution is disabled."
        );
    }

    return result;
}

void validate_fused_target_contract(
    const NativeTensorRtPipeline& pipeline,
    const BenchmarkConfig& config
) {
    if (config.timing_scope != BenchmarkTimingScope::EndToEnd) {
        throw std::invalid_argument(
            "NativeTensorRtPipeline supports only EndToEnd benchmark timing."
        );
    }

    if (pipeline.in_flight_count() != 0U) {
        throw std::invalid_argument(
            "NativeTensorRtPipeline must have no outstanding tickets before "
            "benchmarking begins."
        );
    }

    if (
        config.mode == BenchmarkMode::StreamingThroughput
        && config.pipeline_depth > pipeline.pipeline_depth()
    ) {
        throw std::invalid_argument(
            "Requested benchmark pipeline depth exceeds the constructed "
            "NativeTensorRtPipeline depth."
        );
    }
}

struct FusedSynchronousOutcome final {
    BenchmarkSample sample{};
    std::string operation{};
    std::string failure_message{};
};

[[nodiscard]] FusedSynchronousOutcome execute_fused_synchronous_frame(
    NativeTensorRtPipeline& pipeline,
    const cv::Mat& frame,
    std::uint64_t iteration_index,
    std::size_t source_frame_index,
    bool measure
) {
    FusedSynchronousOutcome outcome{};
    outcome.sample.iteration_index = iteration_index;
    outcome.sample.source_frame_index = source_frame_index;
    outcome.operation = "pipeline_run";

    const BenchmarkTimePoint start = measure
        ? BenchmarkClock::now()
        : BenchmarkTimePoint{};

    try {
        const GpuPipelineResult pipeline_result = pipeline.run(frame);
        outcome.sample.submitted = true;

        const BenchmarkTimePoint end = measure
            ? BenchmarkClock::now()
            : BenchmarkTimePoint{};

        if (
            pipeline_result.summary.detections_written
            != pipeline_result.detections.size()
        ) {
            throw std::runtime_error(
                "GpuPipelineResult summary and DetectionBuffer size disagree."
            );
        }

        if (measure) {
            outcome.sample.timings.end_to_end = elapsed(start, end);
        }

        outcome.sample.target_sequence_number =
            pipeline_result.sequence_number;
        outcome.sample.detection_count = static_cast<std::uint32_t>(
            pipeline_result.detections.size()
        );
        outcome.sample.used_cuda_graph =
            pipeline_result.used_cuda_graph;
        outcome.sample.completed = true;
        outcome.operation.clear();
    } catch (const std::exception& exception) {
        outcome.failure_message = exception_message(exception);
    } catch (...) {
        outcome.failure_message = "Unknown non-standard exception.";
    }

    return outcome;
}

[[nodiscard]] bool process_fused_synchronous_phase(
    NativeTensorRtPipeline& pipeline,
    const std::vector<cv::Mat>& frames,
    BenchmarkPhase phase,
    BenchmarkPhaseCounts& counts,
    BenchmarkResult& result,
    DurationSeries& duration_series,
    const BenchmarkControl& control,
    bool measured,
    BenchmarkCompletionStatus& stop_status
) {
    std::optional<BenchmarkTimePoint> phase_start{};
    std::optional<BenchmarkTimePoint> last_terminal_time{};

    report_progress(
        control,
        phase,
        counts,
        0U,
        result.effective_pipeline_depth
    );

    for (std::uint64_t iteration = 0U;
         iteration < counts.requested;
         ++iteration) {
        if (cancellation_requested(control)) {
            stop_status = BenchmarkCompletionStatus::Cancelled;
            break;
        }

        if (!phase_start.has_value()) {
            phase_start = BenchmarkClock::now();
        }

        ++counts.attempted;

        std::size_t source_index = 0U;
        const cv::Mat& frame = select_frame(
            frames,
            iteration,
            source_index
        );

        FusedSynchronousOutcome outcome =
            execute_fused_synchronous_frame(
                pipeline,
                frame,
                iteration,
                source_index,
                measured
            );

        last_terminal_time = BenchmarkClock::now();

        if (outcome.sample.submitted) {
            ++counts.submitted;
        }

        if (outcome.sample.completed) {
            ++counts.completed;

            if (measured) {
                append_completed_sample(
                    duration_series,
                    outcome.sample,
                    result.config.timing_scope
                );
            }
        } else {
            ++counts.failed;

            record_failure(
                result,
                phase,
                iteration,
                source_index,
                outcome.sample.target_sequence_number,
                outcome.operation.empty()
                    ? std::string{"pipeline_run"}
                    : outcome.operation,
                outcome.failure_message.empty()
                    ? std::string{"Fused pipeline frame failed."}
                    : outcome.failure_message
            );
        }

        const bool frame_completed = outcome.sample.completed;

        if (measured && result.config.retain_raw_samples) {
            result.samples.push_back(std::move(outcome.sample));
        }

        report_progress(
            control,
            phase,
            counts,
            0U,
            result.effective_pipeline_depth
        );

        if (
            !frame_completed
            && result.config.failure_policy
                == BenchmarkFailurePolicy::StopOnFirstFailure
        ) {
            stop_status = BenchmarkCompletionStatus::StoppedOnFailure;
            break;
        }
    }

    if (phase_start.has_value() && last_terminal_time.has_value()) {
        const BenchmarkDuration phase_duration = elapsed(
            *phase_start,
            *last_terminal_time
        );

        if (measured) {
            result.phase_durations.measurement = phase_duration;
        } else {
            result.phase_durations.warmup = phase_duration;
        }
    }

    return stop_status == BenchmarkCompletionStatus::NotStarted;
}

struct PendingFusedFrame final {
    std::uint64_t iteration_index = 0U;
    std::size_t source_frame_index = 0U;
    BenchmarkTimePoint arrival_time{};
};

struct InFlightFusedFrame final {
    GpuPipelineTicket ticket{};
    BenchmarkSample sample{};
    BenchmarkTimePoint arrival_time{};
    BenchmarkTimePoint submission_end{};
};

[[nodiscard]] bool process_fused_streaming_phase(
    NativeTensorRtPipeline& pipeline,
    const std::vector<cv::Mat>& frames,
    BenchmarkPhase phase,
    BenchmarkPhaseCounts& counts,
    BenchmarkResult& result,
    DurationSeries& duration_series,
    const BenchmarkControl& control,
    bool measured,
    BenchmarkCompletionStatus& stop_status
) {
    std::vector<InFlightFusedFrame> in_flight{};
    in_flight.reserve(result.effective_pipeline_depth);

    std::optional<PendingFusedFrame> pending{};

    std::uint64_t next_iteration = 0U;
    bool stop_submitting = false;

    std::optional<BenchmarkTimePoint> phase_start{};
    std::optional<BenchmarkTimePoint> last_terminal_time{};
    std::optional<BenchmarkTimePoint> drain_start{};

    report_progress(
        control,
        phase,
        counts,
        0U,
        result.effective_pipeline_depth
    );

    const auto drain_after_callback_exception = [&]() noexcept {
        pending.reset();

        while (!in_flight.empty()) {
            const GpuPipelineTicket ticket = in_flight.front().ticket;
            in_flight.erase(in_flight.begin());

            try {
                static_cast<void>(pipeline.collect(ticket));
            } catch (...) {
                // The original callback exception remains the exception that
                // must propagate. Best-effort cleanup must never replace it.
            }
        }
    };

    try {
        while (
            next_iteration < counts.requested
            || pending.has_value()
            || !in_flight.empty()
        ) {
            if (!stop_submitting && cancellation_requested(control)) {
                stop_status = BenchmarkCompletionStatus::Cancelled;
                stop_submitting = true;
                pending.reset();
            }

            if (
                !stop_submitting
                && !pending.has_value()
                && next_iteration < counts.requested
            ) {
                std::size_t source_index = 0U;
                static_cast<void>(select_frame(
                    frames,
                    next_iteration,
                    source_index
                ));

                const BenchmarkTimePoint arrival_time =
                    BenchmarkClock::now();

                if (!phase_start.has_value()) {
                    phase_start = arrival_time;
                }

                pending = PendingFusedFrame{
                    next_iteration,
                    source_index,
                    arrival_time
                };
            }

            if (
                !stop_submitting
                && pending.has_value()
                && in_flight.size() < result.effective_pipeline_depth
            ) {
                const PendingFusedFrame current = *pending;
                pending.reset();

                ++counts.attempted;

                BenchmarkSample sample{};
                sample.iteration_index = current.iteration_index;
                sample.source_frame_index = current.source_frame_index;

                const cv::Mat& frame = frames[current.source_frame_index];
                const BenchmarkTimePoint submission_start =
                    BenchmarkClock::now();

                if (measured && result.config.collect_stage_timings) {
                    sample.timings.queue_waiting = elapsed(
                        current.arrival_time,
                        submission_start
                    );
                }

                try {
                    const GpuPipelineTicket ticket = pipeline.submit(frame);
                    const BenchmarkTimePoint submission_end =
                        BenchmarkClock::now();

                    sample.submitted = true;
                    sample.pipeline_slot_index = ticket.slot_index;

                    if (measured && result.config.collect_stage_timings) {
                        sample.timings.submission = elapsed(
                            submission_start,
                            submission_end
                        );
                    }

                    ++counts.submitted;
                    ++next_iteration;

                    in_flight.push_back(
                        InFlightFusedFrame{
                            ticket,
                            std::move(sample),
                            current.arrival_time,
                            submission_end
                        }
                    );
                } catch (const std::exception& exception) {
                    const BenchmarkTimePoint failure_time =
                        BenchmarkClock::now();
                    last_terminal_time = failure_time;

                    if (measured && result.config.collect_stage_timings) {
                        sample.timings.submission = elapsed(
                            submission_start,
                            failure_time
                        );
                    }

                    ++counts.failed;
                    ++next_iteration;

                    record_failure(
                        result,
                        phase,
                        current.iteration_index,
                        current.source_frame_index,
                        std::nullopt,
                        "pipeline_submit",
                        exception_message(exception)
                    );

                    if (measured && result.config.retain_raw_samples) {
                        result.samples.push_back(std::move(sample));
                    }

                    if (
                        result.config.failure_policy
                        == BenchmarkFailurePolicy::StopOnFirstFailure
                    ) {
                        stop_status =
                            BenchmarkCompletionStatus::StoppedOnFailure;
                        stop_submitting = true;
                        pending.reset();
                    }
                } catch (...) {
                    const BenchmarkTimePoint failure_time =
                        BenchmarkClock::now();
                    last_terminal_time = failure_time;

                    if (measured && result.config.collect_stage_timings) {
                        sample.timings.submission = elapsed(
                            submission_start,
                            failure_time
                        );
                    }

                    ++counts.failed;
                    ++next_iteration;

                    record_failure(
                        result,
                        phase,
                        current.iteration_index,
                        current.source_frame_index,
                        std::nullopt,
                        "pipeline_submit",
                        "Unknown non-standard exception."
                    );

                    if (measured && result.config.retain_raw_samples) {
                        result.samples.push_back(std::move(sample));
                    }

                    if (
                        result.config.failure_policy
                        == BenchmarkFailurePolicy::StopOnFirstFailure
                    ) {
                        stop_status =
                            BenchmarkCompletionStatus::StoppedOnFailure;
                        stop_submitting = true;
                        pending.reset();
                    }
                }

                report_progress(
                    control,
                    phase,
                    counts,
                    in_flight.size(),
                    result.effective_pipeline_depth
                );

                continue;
            }

            if (!in_flight.empty()) {
                if (
                    !drain_start.has_value()
                    && (
                        stop_submitting
                        || next_iteration >= counts.requested
                    )
                ) {
                    drain_start = BenchmarkClock::now();
                }

                InFlightFusedFrame current =
                    std::move(in_flight.front());
                in_flight.erase(in_flight.begin());

                try {
                    const GpuPipelineResult pipeline_result =
                        pipeline.collect(current.ticket);

                    const BenchmarkTimePoint complete_time =
                        BenchmarkClock::now();
                    last_terminal_time = complete_time;

                    if (
                        pipeline_result.summary.detections_written
                        != pipeline_result.detections.size()
                    ) {
                        throw std::runtime_error(
                            "GpuPipelineResult summary and DetectionBuffer "
                            "size disagree."
                        );
                    }

                    current.sample.target_sequence_number =
                        pipeline_result.sequence_number;
                    current.sample.detection_count =
                        static_cast<std::uint32_t>(
                            pipeline_result.detections.size()
                        );
                    current.sample.used_cuda_graph =
                        pipeline_result.used_cuda_graph;
                    current.sample.completed = true;

                    if (measured && result.config.collect_stage_timings) {
                        current.sample.timings.completion_wait = elapsed(
                            current.submission_end,
                            complete_time
                        );
                    }

                    if (measured) {
                        current.sample.timings.end_to_end = elapsed(
                            current.arrival_time,
                            complete_time
                        );
                    }

                    ++counts.completed;

                    if (measured) {
                        append_completed_sample(
                            duration_series,
                            current.sample,
                            result.config.timing_scope
                        );
                    }
                } catch (const std::exception& exception) {
                    const BenchmarkTimePoint failure_time =
                        BenchmarkClock::now();
                    last_terminal_time = failure_time;
                    ++counts.failed;

                    if (measured && result.config.collect_stage_timings) {
                        current.sample.timings.completion_wait = elapsed(
                            current.submission_end,
                            failure_time
                        );
                    }

                    record_failure(
                        result,
                        drain_start.has_value()
                            ? BenchmarkPhase::Drain
                            : phase,
                        current.sample.iteration_index,
                        current.sample.source_frame_index,
                        current.sample.target_sequence_number,
                        "pipeline_collect",
                        exception_message(exception)
                    );

                    if (
                        result.config.failure_policy
                        == BenchmarkFailurePolicy::StopOnFirstFailure
                    ) {
                        stop_status =
                            BenchmarkCompletionStatus::StoppedOnFailure;
                        stop_submitting = true;
                        pending.reset();
                    }
                } catch (...) {
                    const BenchmarkTimePoint failure_time =
                        BenchmarkClock::now();
                    last_terminal_time = failure_time;
                    ++counts.failed;

                    if (measured && result.config.collect_stage_timings) {
                        current.sample.timings.completion_wait = elapsed(
                            current.submission_end,
                            failure_time
                        );
                    }

                    record_failure(
                        result,
                        drain_start.has_value()
                            ? BenchmarkPhase::Drain
                            : phase,
                        current.sample.iteration_index,
                        current.sample.source_frame_index,
                        current.sample.target_sequence_number,
                        "pipeline_collect",
                        "Unknown non-standard exception."
                    );

                    if (
                        result.config.failure_policy
                        == BenchmarkFailurePolicy::StopOnFirstFailure
                    ) {
                        stop_status =
                            BenchmarkCompletionStatus::StoppedOnFailure;
                        stop_submitting = true;
                        pending.reset();
                    }
                }

                if (measured && result.config.retain_raw_samples) {
                    result.samples.push_back(std::move(current.sample));
                }

                report_progress(
                    control,
                    drain_start.has_value()
                        ? BenchmarkPhase::Drain
                        : phase,
                    counts,
                    in_flight.size(),
                    result.effective_pipeline_depth
                );

                continue;
            }

            if (stop_submitting) {
                break;
            }

            // A pending frame with an empty queue and a positive effective
            // depth should always have been submitted by the branch above.
            throw std::logic_error(
                "Fused benchmark scheduler reached an impossible state."
            );
        }
    } catch (...) {
        drain_after_callback_exception();
        throw;
    }

    if (phase_start.has_value() && last_terminal_time.has_value()) {
        const BenchmarkDuration phase_duration = elapsed(
            *phase_start,
            *last_terminal_time
        );

        if (measured) {
            result.phase_durations.measurement = phase_duration;
        } else {
            result.phase_durations.warmup = phase_duration;
        }
    }

    if (
        measured
        && drain_start.has_value()
        && last_terminal_time.has_value()
    ) {
        result.phase_durations.drain = elapsed(
            *drain_start,
            *last_terminal_time
        );
    }

    if (pipeline.in_flight_count() != 0U) {
        add_warning_once(
            result,
            "The fused pipeline still reports outstanding work after benchmark "
            "drain; the target should be reconstructed before reuse."
        );
    }

    return stop_status == BenchmarkCompletionStatus::NotStarted;
}

[[nodiscard]] std::string tensor_element_type_name(
    TensorElementType type
) {
    switch (type) {
        case TensorElementType::Float32:
            return "float32";

        case TensorElementType::Float16:
            return "float16";
    }

    return "unknown";
}

[[nodiscard]] std::string channel_order_name(
    ChannelOrder order
) {
    switch (order) {
        case ChannelOrder::RGB:
            return "rgb";

        case ChannelOrder::BGR:
            return "bgr";
    }

    return "unknown";
}

[[nodiscard]] std::string resize_interpolation_name(
    ResizeInterpolation interpolation
) {
    switch (interpolation) {
        case ResizeInterpolation::Linear:
            return "linear";

        case ResizeInterpolation::Automatic:
            return "automatic";

        case ResizeInterpolation::Nearest:
            return "nearest";

        case ResizeInterpolation::Area:
            return "area";

        case ResizeInterpolation::Cubic:
            return "cubic";
    }

    return "unknown";
}

[[nodiscard]] std::string preprocess_execution_policy_name(
    PreprocessExecutionPolicy policy
) {
    switch (policy) {
        case PreprocessExecutionPolicy::Automatic:
            return "automatic";

        case PreprocessExecutionPolicy::Serial:
            return "serial";

        case PreprocessExecutionPolicy::OpenCvParallel:
            return "opencv_parallel";
    }

    return "unknown";
}

[[nodiscard]] std::string malformed_candidate_policy_name(
    MalformedCandidatePolicy policy
) {
    switch (policy) {
        case MalformedCandidatePolicy::Discard:
            return "discard";

        case MalformedCandidatePolicy::Throw:
            return "throw";
    }

    return "unknown";
}

[[nodiscard]] std::string ort_graph_optimization_name(
    OrtGraphOptimization optimization
) {
    switch (optimization) {
        case OrtGraphOptimization::Disabled:
            return "disabled";

        case OrtGraphOptimization::Basic:
            return "basic";

        case OrtGraphOptimization::Extended:
            return "extended";

        case OrtGraphOptimization::All:
            return "all";
    }

    return "unknown";
}

[[nodiscard]] std::string ort_cuda_arena_strategy_name_for_report(
    OrtCudaArenaExtendStrategy strategy
) {
    switch (strategy) {
        case OrtCudaArenaExtendStrategy::NextPowerOfTwo:
            return "next_power_of_two";

        case OrtCudaArenaExtendStrategy::SameAsRequested:
            return "same_as_requested";
    }

    return "unknown";
}

[[nodiscard]] std::string ort_cudnn_search_name_for_report(
    OrtCudnnConvAlgorithmSearch search
) {
    switch (search) {
        case OrtCudnnConvAlgorithmSearch::Exhaustive:
            return "exhaustive";

        case OrtCudnnConvAlgorithmSearch::Heuristic:
            return "heuristic";

        case OrtCudnnConvAlgorithmSearch::Default:
            return "default";
    }

    return "unknown";
}

[[nodiscard]] std::string path_to_utf8(
    const std::filesystem::path& path
) {
    const auto text = path.generic_u8string();
    return std::string{text.begin(), text.end()};
}

void write_json_string(
    std::ostream& output,
    std::string_view value
) {
    output.put('"');

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
                    const std::ios_base::fmtflags flags = output.flags();
                    const char fill = output.fill();

                    output << "\\u"
                           << std::hex
                           << std::uppercase
                           << std::setw(4)
                           << std::setfill('0')
                           << static_cast<unsigned int>(character);

                    output.flags(flags);
                    output.fill(fill);
                } else {
                    output.put(static_cast<char>(character));
                }
                break;
        }
    }

    output.put('"');
}

void write_json_key(
    std::ostream& output,
    std::string_view key
) {
    write_json_string(output, key);
    output.put(':');
}

void write_json_bool(
    std::ostream& output,
    bool value
) {
    output << (value ? "true" : "false");
}

void write_json_double(
    std::ostream& output,
    double value
) {
    if (!std::isfinite(value)) {
        output << "null";
        return;
    }

    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
}

void write_json_duration(
    std::ostream& output,
    BenchmarkDuration duration
) {
    write_json_double(output, duration.count());
}

void write_json_optional_duration(
    std::ostream& output,
    const std::optional<BenchmarkDuration>& duration
) {
    if (!duration.has_value()) {
        output << "null";
        return;
    }

    write_json_duration(output, *duration);
}

void write_json_optional_bool(
    std::ostream& output,
    const std::optional<bool>& value
) {
    if (!value.has_value()) {
        output << "null";
        return;
    }

    write_json_bool(output, *value);
}

template <typename Integer>
void write_json_optional_integer(
    std::ostream& output,
    const std::optional<Integer>& value
) {
    if (!value.has_value()) {
        output << "null";
        return;
    }

    output << *value;
}

template <typename Value, std::size_t Size>
void write_json_numeric_array(
    std::ostream& output,
    const std::array<Value, Size>& values
) {
    output.put('[');

    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) {
            output.put(',');
        }

        if constexpr (std::is_floating_point_v<Value>) {
            write_json_double(output, static_cast<double>(values[index]));
        } else {
            output << +values[index];
        }
    }

    output.put(']');
}

void write_json_string_array(
    std::ostream& output,
    const std::vector<std::string>& values
) {
    output.put('[');

    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) {
            output.put(',');
        }

        write_json_string(output, values[index]);
    }

    output.put(']');
}

void write_preprocess_config_json(
    std::ostream& output,
    const PreprocessConfig& config
) {
    output.put('{');
    write_json_key(output, "network_width");
    output << config.network_size.width << ',';
    write_json_key(output, "network_height");
    output << config.network_size.height << ',';
    write_json_key(output, "allow_scale_up");
    write_json_bool(output, config.allow_scale_up);
    output.put(',');
    write_json_key(output, "center_letterbox");
    write_json_bool(output, config.center_letterbox);
    output.put(',');
    write_json_key(output, "padding_bgr");
    write_json_numeric_array(output, config.padding_bgr);
    output.put(',');
    write_json_key(output, "output_channel_order");
    write_json_string(output, channel_order_name(config.output_channel_order));
    output.put(',');
    write_json_key(output, "interpolation");
    write_json_string(output, resize_interpolation_name(config.interpolation));
    output.put(',');
    write_json_key(output, "execution_policy");
    write_json_string(
        output,
        preprocess_execution_policy_name(config.execution_policy)
    );
    output.put(',');
    write_json_key(output, "parallel_minimum_pixels");
    output << config.parallel_minimum_pixels << ',';
    write_json_key(output, "pixel_scale");
    write_json_double(output, static_cast<double>(config.pixel_scale));
    output.put(',');
    write_json_key(output, "mean");
    write_json_numeric_array(output, config.mean);
    output.put(',');
    write_json_key(output, "stddev");
    write_json_numeric_array(output, config.stddev);
    output.put('}');
}

void write_postprocess_config_json(
    std::ostream& output,
    const PostprocessConfig& config
) {
    output.put('{');
    write_json_key(output, "confidence_threshold");
    write_json_double(output, static_cast<double>(config.confidence_threshold));
    output.put(',');
    write_json_key(output, "maximum_detections");
    output << config.maximum_detections << ',';
    write_json_key(output, "class_count");
    output << config.class_count << ',';
    write_json_key(output, "clip_boxes");
    write_json_bool(output, config.clip_boxes);
    output.put(',');
    write_json_key(output, "discard_degenerate_boxes");
    write_json_bool(output, config.discard_degenerate_boxes);
    output.put(',');
    write_json_key(output, "output_sorted_by_confidence");
    write_json_bool(output, config.output_sorted_by_confidence);
    output.put(',');
    write_json_key(output, "class_id_tolerance");
    write_json_double(output, static_cast<double>(config.class_id_tolerance));
    output.put(',');
    write_json_key(output, "malformed_candidate_policy");
    write_json_string(
        output,
        malformed_candidate_policy_name(config.malformed_candidate_policy)
    );
    output.put('}');
}

void write_inference_engine_config_json(
    std::ostream& output,
    const InferenceEngineConfig& config
) {
    output.put('{');
    write_json_key(output, "artifact_path");
    write_json_string(output, path_to_utf8(config.artifact_path));
    output.put(',');
    write_json_key(output, "backend");
    write_json_string(output, inference_backend_name(config.backend));
    output.put(',');
    write_json_key(output, "artifact_precision");
    write_json_string(output, artifact_precision_name(config.artifact_precision));
    output.put(',');
    write_json_key(output, "expected_input_name");
    write_json_string(output, config.expected_input_name);
    output.put(',');
    write_json_key(output, "expected_output_name");
    write_json_string(output, config.expected_output_name);
    output.put(',');

    write_json_key(output, "ort");
    output.put('{');
    write_json_key(output, "graph_optimization");
    write_json_string(
        output,
        ort_graph_optimization_name(config.ort.graph_optimization)
    );
    output.put(',');
    write_json_key(output, "intra_op_thread_count");
    output << config.ort.intra_op_thread_count << ',';
    write_json_key(output, "inter_op_thread_count");
    output << config.ort.inter_op_thread_count << ',';
    write_json_key(output, "enable_cpu_memory_arena");
    write_json_bool(output, config.ort.enable_cpu_memory_arena);
    output.put(',');
    write_json_key(output, "enable_memory_pattern");
    write_json_bool(output, config.ort.enable_memory_pattern);
    output.put(',');
    write_json_key(output, "enable_profiling");
    write_json_bool(output, config.ort.enable_profiling);
    output.put(',');
    write_json_key(output, "profiling_output");
    write_json_string(output, path_to_utf8(config.ort.profiling_output));
    output.put('}');
    output.put(',');

    write_json_key(output, "gpu");
    output.put('{');
    write_json_key(output, "device_id");
    output << config.gpu.device_id << ',';
    write_json_key(output, "gpu_memory_limit_bytes");
    output << config.gpu.gpu_memory_limit_bytes << ',';
    write_json_key(output, "arena_extend_strategy");
    write_json_string(
        output,
        ort_cuda_arena_strategy_name_for_report(
            config.gpu.arena_extend_strategy
        )
    );
    output.put(',');
    write_json_key(output, "cudnn_conv_algorithm_search");
    write_json_string(
        output,
        ort_cudnn_search_name_for_report(
            config.gpu.cudnn_conv_algorithm_search
        )
    );
    output.put(',');
    write_json_key(output, "cudnn_conv_use_max_workspace");
    write_json_bool(output, config.gpu.cudnn_conv_use_max_workspace);
    output.put(',');
    write_json_key(output, "prefer_nhwc");
    write_json_bool(output, config.gpu.prefer_nhwc);
    output.put(',');
    write_json_key(output, "use_tf32");
    write_json_bool(output, config.gpu.use_tf32);
    output.put(',');
    write_json_key(output, "use_ep_level_unified_stream");
    write_json_bool(output, config.gpu.use_ep_level_unified_stream);
    output.put(',');
    write_json_key(output, "disable_provider_synchronization");
    write_json_bool(output, config.gpu.disable_provider_synchronization);
    output.put(',');
    write_json_key(output, "enable_cuda_graph");
    write_json_bool(output, config.gpu.enable_cuda_graph);
    output.put('}');
    output.put(',');

    write_json_key(output, "ort_tensorrt");
    output.put('{');
    write_json_key(output, "enable_cuda_fallback");
    write_json_bool(output, config.ort_tensorrt.enable_cuda_fallback);
    output.put(',');
    write_json_key(output, "maximum_workspace_bytes");
    output << config.ort_tensorrt.maximum_workspace_bytes << ',';
    write_json_key(output, "minimum_subgraph_size");
    output << config.ort_tensorrt.minimum_subgraph_size << ',';
    write_json_key(output, "maximum_partition_iterations");
    output << config.ort_tensorrt.maximum_partition_iterations << ',';
    write_json_key(output, "enable_engine_cache");
    write_json_bool(output, config.ort_tensorrt.enable_engine_cache);
    output.put(',');
    write_json_key(output, "engine_cache_directory");
    write_json_string(
        output,
        path_to_utf8(config.ort_tensorrt.engine_cache_directory)
    );
    output.put(',');
    write_json_key(output, "enable_timing_cache");
    write_json_bool(output, config.ort_tensorrt.enable_timing_cache);
    output.put(',');
    write_json_key(output, "timing_cache_directory");
    write_json_string(
        output,
        path_to_utf8(config.ort_tensorrt.timing_cache_directory)
    );
    output.put(',');
    write_json_key(output, "enable_context_memory_sharing");
    write_json_bool(
        output,
        config.ort_tensorrt.enable_context_memory_sharing
    );
    output.put(',');
    write_json_key(output, "enable_cuda_graph");
    write_json_bool(output, config.ort_tensorrt.enable_cuda_graph);
    output.put(',');
    write_json_key(output, "builder_optimization_level");
    output << config.ort_tensorrt.builder_optimization_level << ',';
    write_json_key(output, "auxiliary_streams");
    output << config.ort_tensorrt.auxiliary_streams << ',';
    write_json_key(output, "dump_partitioned_subgraphs");
    write_json_bool(
        output,
        config.ort_tensorrt.dump_partitioned_subgraphs
    );
    output.put('}');
    output.put('}');
}

void write_inference_runtime_info_json(
    std::ostream& output,
    const InferenceRuntimeInfo& info
) {
    output.put('{');
    write_json_key(output, "backend");
    write_json_string(output, inference_backend_name(info.backend));
    output.put(',');
    write_json_key(output, "artifact_precision");
    write_json_string(output, artifact_precision_name(info.artifact_precision));
    output.put(',');
    write_json_key(output, "runtime_name");
    write_json_string(output, info.runtime_name);
    output.put(',');
    write_json_key(output, "runtime_version");
    write_json_string(output, info.runtime_version);
    output.put(',');
    write_json_key(output, "active_execution_providers");
    write_json_string_array(output, info.active_execution_providers);
    output.put(',');

    write_json_key(output, "model");
    output.put('{');
    write_json_key(output, "input_name");
    write_json_string(output, info.model.input_name);
    output.put(',');
    write_json_key(output, "output_name");
    write_json_string(output, info.model.output_name);
    output.put(',');
    write_json_key(output, "input_shape");
    write_json_numeric_array(output, info.model.input_shape);
    output.put(',');
    write_json_key(output, "output_shape");
    write_json_numeric_array(output, info.model.output_shape);
    output.put(',');
    write_json_key(output, "input_element_type");
    write_json_string(
        output,
        tensor_element_type_name(info.model.input_element_type)
    );
    output.put(',');
    write_json_key(output, "backend_output_element_type");
    write_json_string(
        output,
        tensor_element_type_name(info.model.backend_output_element_type)
    );
    output.put(',');
    write_json_key(output, "host_output_element_type");
    write_json_string(
        output,
        tensor_element_type_name(info.model.host_output_element_type)
    );
    output.put(',');
    write_json_key(output, "input_element_count");
    output << info.model.input_element_count << ',';
    write_json_key(output, "input_byte_count");
    output << info.model.input_byte_count << ',';
    write_json_key(output, "output_element_count");
    output << info.model.output_element_count << ',';
    write_json_key(output, "backend_output_byte_count");
    output << info.model.backend_output_byte_count << ',';
    write_json_key(output, "host_output_byte_count");
    output << info.model.host_output_byte_count;
    output.put('}');
    output.put(',');

    write_json_key(output, "buffers");
    output.put('{');
    write_json_key(output, "host_input_bytes");
    output << info.buffers.host_input_bytes << ',';
    write_json_key(output, "host_output_bytes");
    output << info.buffers.host_output_bytes << ',';
    write_json_key(output, "device_input_bytes");
    output << info.buffers.device_input_bytes << ',';
    write_json_key(output, "device_output_bytes");
    output << info.buffers.device_output_bytes << ',';
    write_json_key(output, "device_workspace_bytes");
    output << info.buffers.device_workspace_bytes << ',';
    write_json_key(output, "host_input_is_pinned");
    write_json_bool(output, info.buffers.host_input_is_pinned);
    output.put(',');
    write_json_key(output, "host_output_is_pinned");
    write_json_bool(output, info.buffers.host_output_is_pinned);
    output.put('}');
    output.put('}');
}

void write_native_pipeline_config_json(
    std::ostream& output,
    const NativeTensorRtPipelineConfig& config
) {
    output.put('{');
    write_json_key(output, "engine_path");
    write_json_string(output, path_to_utf8(config.engine_path));
    output.put(',');
    write_json_key(output, "expected_input_name");
    write_json_string(output, config.expected_input_name);
    output.put(',');
    write_json_key(output, "expected_output_name");
    write_json_string(output, config.expected_output_name);
    output.put(',');
    write_json_key(output, "device_id");
    output << config.device_id << ',';
    write_json_key(output, "pipeline_depth");
    output << config.pipeline_depth << ',';
    write_json_key(output, "maximum_host_image_width");
    output << config.maximum_host_image_size.width << ',';
    write_json_key(output, "maximum_host_image_height");
    output << config.maximum_host_image_size.height << ',';
    write_json_key(output, "preprocess");
    write_preprocess_config_json(output, config.preprocess);
    output.put(',');
    write_json_key(output, "postprocess");
    write_postprocess_config_json(output, config.postprocess);
    output.put(',');
    write_json_key(output, "enable_cuda_graph");
    write_json_bool(output, config.enable_cuda_graph);
    output.put(',');
    write_json_key(output, "require_cuda_graph");
    write_json_bool(output, config.require_cuda_graph);
    output.put(',');
    write_json_key(output, "enable_gpu_postprocess");
    write_json_bool(output, config.enable_gpu_postprocess);
    output.put(',');
    write_json_key(output, "use_high_priority_streams");
    write_json_bool(output, config.use_high_priority_streams);
    output.put(',');
    write_json_key(output, "use_engine_auxiliary_streams");
    write_json_bool(output, config.use_engine_auxiliary_streams);
    output.put(',');
    write_json_key(output, "minimize_runtime_instrumentation");
    write_json_bool(output, config.minimize_runtime_instrumentation);
    output.put(',');
    write_json_key(output, "persistent_cache_limit_bytes");
    output << config.persistent_cache_limit_bytes << ',';
    write_json_key(output, "capture_external_sources");
    write_json_bool(output, config.capture_external_sources);
    output.put('}');
}

void write_gpu_pipeline_runtime_info_json(
    std::ostream& output,
    const GpuPipelineRuntimeInfo& info
) {
    output.put('{');
    write_json_key(output, "runtime_name");
    write_json_string(output, info.runtime_name);
    output.put(',');
    write_json_key(output, "runtime_version");
    write_json_string(output, info.runtime_version);
    output.put(',');
    write_json_key(output, "input_element_type");
    write_json_string(output, tensor_element_type_name(info.input_element_type));
    output.put(',');
    write_json_key(output, "output_element_type");
    write_json_string(output, tensor_element_type_name(info.output_element_type));
    output.put(',');
    write_json_key(output, "input_shape");
    write_json_numeric_array(output, info.input_shape);
    output.put(',');
    write_json_key(output, "output_shape");
    write_json_numeric_array(output, info.output_shape);
    output.put(',');
    write_json_key(output, "gpu_postprocess_enabled");
    write_json_bool(output, info.gpu_postprocess_enabled);
    output.put(',');
    write_json_key(output, "cuda_graph_enabled");
    write_json_bool(output, info.cuda_graph_enabled);
    output.put(',');

    write_json_key(output, "memory");
    output.put('{');
    write_json_key(output, "pipeline_depth");
    output << info.memory.pipeline_depth << ',';
    write_json_key(output, "auxiliary_streams_per_slot");
    output << info.memory.auxiliary_streams_per_slot << ',';
    write_json_key(output, "pinned_host_bytes_per_slot");
    output << info.memory.pinned_host_bytes_per_slot << ',';
    write_json_key(output, "device_bytes_per_slot");
    output << info.memory.device_bytes_per_slot << ',';
    write_json_key(output, "total_pinned_host_bytes");
    output << info.memory.total_pinned_host_bytes << ',';
    write_json_key(output, "total_device_bytes");
    output << info.memory.total_device_bytes;
    output.put('}');
    output.put('}');
}

void write_benchmark_config_json(
    std::ostream& output,
    const BenchmarkConfig& config
) {
    output.put('{');
    write_json_key(output, "mode");
    write_json_string(output, benchmark_mode_name(config.mode));
    output.put(',');
    write_json_key(output, "timing_scope");
    write_json_string(output, benchmark_timing_scope_name(config.timing_scope));
    output.put(',');
    write_json_key(output, "warmup_iterations");
    output << config.warmup_iterations << ',';
    write_json_key(output, "measured_iterations");
    output << config.measured_iterations << ',';
    write_json_key(output, "pipeline_depth");
    output << config.pipeline_depth << ',';
    write_json_key(output, "collect_stage_timings");
    write_json_bool(output, config.collect_stage_timings);
    output.put(',');
    write_json_key(output, "retain_raw_samples");
    write_json_bool(output, config.retain_raw_samples);
    output.put(',');
    write_json_key(output, "failure_policy");
    write_json_string(
        output,
        benchmark_failure_policy_name(config.failure_policy)
    );
    output.put(',');
    write_json_key(output, "maximum_recorded_failures");
    output << config.maximum_recorded_failures;
    output.put('}');
}

void write_phase_counts_json(
    std::ostream& output,
    const BenchmarkPhaseCounts& counts
) {
    output.put('{');
    write_json_key(output, "requested");
    output << counts.requested << ',';
    write_json_key(output, "attempted");
    output << counts.attempted << ',';
    write_json_key(output, "submitted");
    output << counts.submitted << ',';
    write_json_key(output, "completed");
    output << counts.completed << ',';
    write_json_key(output, "failed");
    output << counts.failed;
    output.put('}');
}

void write_latency_statistics_json(
    std::ostream& output,
    const std::optional<LatencyStatistics>& statistics
) {
    if (!statistics.has_value()) {
        output << "null";
        return;
    }

    const LatencyStatistics& value = *statistics;

    output.put('{');
    write_json_key(output, "sample_count");
    output << value.sample_count << ',';
    write_json_key(output, "minimum_ms");
    write_json_duration(output, value.minimum);
    output.put(',');
    write_json_key(output, "maximum_ms");
    write_json_duration(output, value.maximum);
    output.put(',');
    write_json_key(output, "mean_ms");
    write_json_duration(output, value.mean);
    output.put(',');
    write_json_key(output, "median_ms");
    write_json_duration(output, value.median);
    output.put(',');
    write_json_key(output, "standard_deviation_ms");
    write_json_duration(output, value.standard_deviation);
    output.put(',');
    write_json_key(output, "p90_ms");
    write_json_duration(output, value.p90);
    output.put(',');
    write_json_key(output, "p95_ms");
    write_json_duration(output, value.p95);
    output.put(',');
    write_json_key(output, "p99_ms");
    write_json_duration(output, value.p99);
    output.put('}');
}

void write_stage_durations_json(
    std::ostream& output,
    const BenchmarkStageDurations& timings
) {
    output.put('{');
    write_json_key(output, "queue_waiting_ms");
    write_json_optional_duration(output, timings.queue_waiting);
    output.put(',');
    write_json_key(output, "submission_ms");
    write_json_optional_duration(output, timings.submission);
    output.put(',');
    write_json_key(output, "preprocessing_ms");
    write_json_optional_duration(output, timings.preprocessing);
    output.put(',');
    write_json_key(output, "backend_execution_ms");
    write_json_optional_duration(output, timings.backend_execution);
    output.put(',');
    write_json_key(output, "postprocessing_ms");
    write_json_optional_duration(output, timings.postprocessing);
    output.put(',');
    write_json_key(output, "host_to_device_ms");
    write_json_optional_duration(output, timings.host_to_device);
    output.put(',');
    write_json_key(output, "device_execution_ms");
    write_json_optional_duration(output, timings.device_execution);
    output.put(',');
    write_json_key(output, "device_to_host_ms");
    write_json_optional_duration(output, timings.device_to_host);
    output.put(',');
    write_json_key(output, "completion_wait_ms");
    write_json_optional_duration(output, timings.completion_wait);
    output.put(',');
    write_json_key(output, "end_to_end_ms");
    write_json_optional_duration(output, timings.end_to_end);
    output.put('}');
}

void write_sample_json(
    std::ostream& output,
    const BenchmarkSample& sample
) {
    output.put('{');
    write_json_key(output, "iteration_index");
    output << sample.iteration_index << ',';
    write_json_key(output, "source_frame_index");
    output << sample.source_frame_index << ',';
    write_json_key(output, "target_sequence_number");
    write_json_optional_integer(output, sample.target_sequence_number);
    output.put(',');
    write_json_key(output, "pipeline_slot_index");
    write_json_optional_integer(output, sample.pipeline_slot_index);
    output.put(',');
    write_json_key(output, "detection_count");
    write_json_optional_integer(output, sample.detection_count);
    output.put(',');
    write_json_key(output, "used_cuda_graph");
    write_json_optional_bool(output, sample.used_cuda_graph);
    output.put(',');
    write_json_key(output, "submitted");
    write_json_bool(output, sample.submitted);
    output.put(',');
    write_json_key(output, "completed");
    write_json_bool(output, sample.completed);
    output.put(',');
    write_json_key(output, "timings");
    write_stage_durations_json(output, sample.timings);
    output.put('}');
}

void write_target_json(
    std::ostream& output,
    const BenchmarkTargetInfo& target
) {
    std::visit(
        [&output](const auto& value) {
            using Target = std::decay_t<decltype(value)>;

            output.put('{');

            if constexpr (
                std::is_same_v<
                    Target,
                    InferenceEngineBenchmarkTargetInfo
                >
            ) {
                write_json_key(output, "type");
                write_json_string(output, "inference_engine");
                output.put(',');
                write_json_key(output, "engine_config");
                write_inference_engine_config_json(
                    output,
                    value.engine_config
                );
                output.put(',');
                write_json_key(output, "runtime_info");
                write_inference_runtime_info_json(
                    output,
                    value.runtime_info
                );
                output.put(',');
                write_json_key(output, "preprocess_config");
                write_preprocess_config_json(
                    output,
                    value.preprocess_config
                );
                output.put(',');
                write_json_key(output, "postprocess_config");
                write_postprocess_config_json(
                    output,
                    value.postprocess_config
                );
            } else {
                write_json_key(output, "type");
                write_json_string(output, "native_tensorrt_pipeline");
                output.put(',');
                write_json_key(output, "pipeline_config");
                write_native_pipeline_config_json(
                    output,
                    value.pipeline_config
                );
                output.put(',');
                write_json_key(output, "runtime_info");
                write_gpu_pipeline_runtime_info_json(
                    output,
                    value.runtime_info
                );
            }

            output.put('}');
        },
        target
    );
}

[[nodiscard]] std::string target_kind_name(
    const BenchmarkTargetInfo& target
) {
    return std::holds_alternative<InferenceEngineBenchmarkTargetInfo>(target)
        ? "inference_engine"
        : "native_tensorrt_pipeline";
}

[[nodiscard]] std::string target_backend_name(
    const BenchmarkTargetInfo& target
) {
    return std::visit(
        [](const auto& value) -> std::string {
            using Target = std::decay_t<decltype(value)>;

            if constexpr (
                std::is_same_v<
                    Target,
                    InferenceEngineBenchmarkTargetInfo
                >
            ) {
                return std::string{
                    inference_backend_name(value.runtime_info.backend)
                };
            } else {
                return "native_tensorrt_fused";
            }
        },
        target
    );
}

[[nodiscard]] std::string target_precision_name(
    const BenchmarkTargetInfo& target
) {
    return std::visit(
        [](const auto& value) -> std::string {
            using Target = std::decay_t<decltype(value)>;

            if constexpr (
                std::is_same_v<
                    Target,
                    InferenceEngineBenchmarkTargetInfo
                >
            ) {
                return std::string{
                    artifact_precision_name(
                        value.runtime_info.artifact_precision
                    )
                };
            } else {
                // A native INT8 TensorRT engine may still expose FP32 or FP16
                // external bindings. Binding type is therefore not artifact
                // precision and must not be reported as such.
                return "engine_defined";
            }
        },
        target
    );
}

[[nodiscard]] std::string target_artifact_path(
    const BenchmarkTargetInfo& target
) {
    return std::visit(
        [](const auto& value) -> std::string {
            using Target = std::decay_t<decltype(value)>;

            if constexpr (
                std::is_same_v<
                    Target,
                    InferenceEngineBenchmarkTargetInfo
                >
            ) {
                return path_to_utf8(value.engine_config.artifact_path);
            } else {
                return path_to_utf8(value.pipeline_config.engine_path);
            }
        },
        target
    );
}

[[nodiscard]] std::string target_runtime_name(
    const BenchmarkTargetInfo& target
) {
    return std::visit(
        [](const auto& value) {
            return value.runtime_info.runtime_name;
        },
        target
    );
}

[[nodiscard]] std::string target_runtime_version(
    const BenchmarkTargetInfo& target
) {
    return std::visit(
        [](const auto& value) {
            return value.runtime_info.runtime_version;
        },
        target
    );
}

[[nodiscard]] std::string csv_escape(
    std::string_view value
) {
    const bool requires_quotes =
        value.find_first_of(",\"\r\n") != std::string_view::npos;

    if (!requires_quotes) {
        return std::string{value};
    }

    std::string escaped{};
    escaped.reserve(value.size() + 2U);
    escaped.push_back('"');

    for (const char character : value) {
        if (character == '"') {
            escaped.push_back('"');
        }

        escaped.push_back(character);
    }

    escaped.push_back('"');
    return escaped;
}

[[nodiscard]] std::string optional_duration_csv(
    const std::optional<BenchmarkDuration>& duration
) {
    if (!duration.has_value()) {
        return {};
    }

    std::ostringstream output{};
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << duration->count();
    return output.str();
}

[[nodiscard]] std::string duration_csv(
    BenchmarkDuration duration
) {
    std::ostringstream output{};
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << duration.count();
    return output.str();
}

[[nodiscard]] std::string optional_uint64_csv(
    const std::optional<std::uint64_t>& value
) {
    return value.has_value() ? std::to_string(*value) : std::string{};
}

[[nodiscard]] std::string optional_uint32_csv(
    const std::optional<std::uint32_t>& value
) {
    return value.has_value() ? std::to_string(*value) : std::string{};
}

[[nodiscard]] std::string optional_bool_csv(
    const std::optional<bool>& value
) {
    if (!value.has_value()) {
        return {};
    }

    return *value ? "true" : "false";
}

[[nodiscard]] std::string optional_stat_csv(
    const std::optional<LatencyStatistics>& statistics,
    BenchmarkDuration LatencyStatistics::* member
) {
    return statistics.has_value()
        ? duration_csv((*statistics).*member)
        : std::string{};
}

[[nodiscard]] std::filesystem::path make_temporary_path(
    const std::filesystem::path& output_path
) {
    static std::atomic<std::uint64_t> unique_counter{0U};

    std::ostringstream suffix{};
    suffix << ".tmp-"
           << BenchmarkClock::now().time_since_epoch().count()
           << '-'
           << std::this_thread::get_id()
           << '-'
           << unique_counter.fetch_add(1U, std::memory_order_relaxed);

    std::filesystem::path temporary_path = output_path;
    temporary_path += suffix.str();
    return temporary_path;
}

void write_text_file(
    const std::filesystem::path& output_path,
    std::string_view contents
) {
    if (output_path.empty()) {
        throw std::invalid_argument("Output path must not be empty.");
    }

    std::error_code error{};
    const std::filesystem::path parent = output_path.parent_path();

    if (!parent.empty()) {
        std::filesystem::create_directories(parent, error);

        if (error) {
            throw std::runtime_error(
                "Failed to create output directory '"
                + path_to_utf8(parent)
                + "': "
                + error.message()
            );
        }
    }

    const std::filesystem::path temporary_path =
        make_temporary_path(output_path);

    try {
        std::ofstream stream{
            temporary_path,
            std::ios::binary | std::ios::trunc
        };

        if (!stream.is_open()) {
            throw std::runtime_error(
                "Failed to open temporary output file '"
                + path_to_utf8(temporary_path)
                + "'."
            );
        }

        if (
            contents.size()
            > static_cast<std::size_t>(
                std::numeric_limits<std::streamsize>::max()
            )
        ) {
            throw std::overflow_error(
                "Output contents exceed std::streamsize capacity."
            );
        }

        stream.write(
            contents.data(),
            static_cast<std::streamsize>(contents.size())
        );
        stream.flush();

        if (!stream.good()) {
            throw std::runtime_error(
                "Failed while writing temporary output file '"
                + path_to_utf8(temporary_path)
                + "'."
            );
        }

        stream.close();

        if (stream.fail()) {
            throw std::runtime_error(
                "Failed to finalize temporary output file '"
                + path_to_utf8(temporary_path)
                + "'."
            );
        }

        error.clear();
        std::filesystem::remove(output_path, error);

        if (error) {
            throw std::runtime_error(
                "Failed to replace existing output file '"
                + path_to_utf8(output_path)
                + "': "
                + error.message()
            );
        }

        error.clear();
        std::filesystem::rename(
            temporary_path,
            output_path,
            error
        );

        if (error) {
            throw std::runtime_error(
                "Failed to publish output file '"
                + path_to_utf8(output_path)
                + "': "
                + error.message()
            );
        }
    } catch (...) {
        error.clear();
        std::filesystem::remove(temporary_path, error);
        throw;
    }
}

}  // namespace

void BenchmarkConfig::validate() const {
    if (!supported_benchmark_mode(mode)) {
        throw std::invalid_argument("BenchmarkConfig.mode is invalid.");
    }

    if (!supported_timing_scope(timing_scope)) {
        throw std::invalid_argument(
            "BenchmarkConfig.timing_scope is invalid."
        );
    }

    if (!supported_failure_policy(failure_policy)) {
        throw std::invalid_argument(
            "BenchmarkConfig.failure_policy is invalid."
        );
    }

    if (measured_iterations == 0U) {
        throw std::invalid_argument(
            "BenchmarkConfig.measured_iterations must be positive."
        );
    }

    if (
        pipeline_depth < kMinimumPipelineDepth
        || pipeline_depth > kMaximumPipelineDepth
    ) {
        throw std::invalid_argument(
            "BenchmarkConfig.pipeline_depth must be inside [1, 8]."
        );
    }

    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
        if (
            measured_iterations
            > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()
            )
        ) {
            throw std::overflow_error(
                "BenchmarkConfig.measured_iterations exceeds size_t capacity."
            );
        }
    }

    if (
        warmup_iterations
        > std::numeric_limits<std::uint64_t>::max()
            - measured_iterations
    ) {
        throw std::overflow_error(
            "Benchmark iteration total overflows uint64_t."
        );
    }
}

bool BenchmarkResult::successful() const noexcept {
    return status == BenchmarkCompletionStatus::Completed
        && counts.measurement.attempted
            == counts.measurement.requested
        && counts.measurement.submitted
            == counts.measurement.requested
        && counts.measurement.completed
            == counts.measurement.requested
        && counts.measurement.failed == 0U
        && primary_latency.has_value()
        && primary_latency->sample_count
            == counts.measurement.completed;
}

std::optional<LatencyStatistics> calculate_latency_statistics(
    const std::vector<BenchmarkDuration>& samples
) {
    if (samples.empty()) {
        return std::nullopt;
    }

    std::vector<double> sorted{};
    sorted.reserve(samples.size());

    for (const BenchmarkDuration sample : samples) {
        if (!finite_non_negative(sample)) {
            throw std::invalid_argument(
                "Latency samples must be finite and non-negative."
            );
        }

        sorted.push_back(sample.count());
    }

    std::sort(sorted.begin(), sorted.end());

    long double sum = 0.0L;

    for (const double value : sorted) {
        sum += static_cast<long double>(value);
    }

    const long double count = static_cast<long double>(sorted.size());
    const long double mean = sum / count;
    long double squared_error_sum = 0.0L;

    for (const double value : sorted) {
        const long double difference =
            static_cast<long double>(value) - mean;
        squared_error_sum += difference * difference;
    }

    const long double standard_deviation = std::sqrt(
        squared_error_sum / count
    );

    const auto percentile = [&sorted](long double probability) {
        if (sorted.size() == 1U) {
            return sorted.front();
        }

        const long double position =
            static_cast<long double>(sorted.size() - 1U)
            * probability;

        const std::size_t lower = static_cast<std::size_t>(
            std::floor(position)
        );
        const std::size_t upper = static_cast<std::size_t>(
            std::ceil(position)
        );
        const long double fraction =
            position - static_cast<long double>(lower);

        const long double lower_value =
            static_cast<long double>(sorted[lower]);
        const long double upper_value =
            static_cast<long double>(sorted[upper]);

        return static_cast<double>(
            lower_value + fraction * (upper_value - lower_value)
        );
    };

    if (
        samples.size()
        > static_cast<std::size_t>(
            std::numeric_limits<std::uint64_t>::max()
        )
    ) {
        throw std::overflow_error(
            "Latency sample count cannot be represented by uint64_t."
        );
    }

    return LatencyStatistics{
        static_cast<std::uint64_t>(samples.size()),
        BenchmarkDuration{sorted.front()},
        BenchmarkDuration{sorted.back()},
        BenchmarkDuration{static_cast<double>(mean)},
        BenchmarkDuration{percentile(0.50L)},
        BenchmarkDuration{static_cast<double>(standard_deviation)},
        BenchmarkDuration{percentile(0.90L)},
        BenchmarkDuration{percentile(0.95L)},
        BenchmarkDuration{percentile(0.99L)}
    };
}

BenchmarkResult run_benchmark(
    InferenceEngine& engine,
    ImageProcessor& image_processor,
    const Postprocessor& postprocessor,
    const std::vector<cv::Mat>& frames,
    const BenchmarkConfig& config,
    const BenchmarkControl& control
) {
    config.validate();
    validate_source_frames(frames);
    validate_generic_target_contract(engine, image_processor);

    BenchmarkResult result = make_generic_result(
        engine,
        image_processor,
        postprocessor,
        config,
        control
    );

    DurationSeries duration_series{};
    duration_series.reserve(
        static_cast<std::size_t>(config.measured_iterations)
    );

    if (config.retain_raw_samples) {
        result.samples.reserve(duration_series.primary.capacity());
    }

    BenchmarkCompletionStatus stop_status =
        BenchmarkCompletionStatus::NotStarted;

    report_progress(
        control,
        BenchmarkPhase::Setup,
        BenchmarkPhaseCounts{},
        0U,
        result.effective_pipeline_depth
    );

    const bool warmup_completed = process_generic_phase(
        engine,
        image_processor,
        postprocessor,
        frames,
        BenchmarkPhase::Warmup,
        result.counts.warmup,
        result,
        duration_series,
        control,
        false,
        stop_status
    );

    if (warmup_completed) {
        static_cast<void>(process_generic_phase(
            engine,
            image_processor,
            postprocessor,
            frames,
            BenchmarkPhase::Measurement,
            result.counts.measurement,
            result,
            duration_series,
            control,
            true,
            stop_status
        ));
    }

    result.status = stop_status == BenchmarkCompletionStatus::NotStarted
        ? BenchmarkCompletionStatus::Completed
        : stop_status;

    finalize_statistics(result, duration_series);
    finalize_throughput(result);

    report_progress(
        control,
        BenchmarkPhase::Complete,
        result.counts.measurement,
        0U,
        result.effective_pipeline_depth
    );

    return result;
}

BenchmarkResult run_benchmark(
    NativeTensorRtPipeline& pipeline,
    const std::vector<cv::Mat>& frames,
    const BenchmarkConfig& config,
    const BenchmarkControl& control
) {
    config.validate();
    validate_source_frames(frames);
    validate_fused_target_contract(pipeline, config);

    BenchmarkResult result = make_fused_result(
        pipeline,
        config,
        control
    );

    DurationSeries duration_series{};
    duration_series.reserve(
        static_cast<std::size_t>(config.measured_iterations)
    );

    if (config.retain_raw_samples) {
        result.samples.reserve(duration_series.primary.capacity());
    }

    BenchmarkCompletionStatus stop_status =
        BenchmarkCompletionStatus::NotStarted;

    report_progress(
        control,
        BenchmarkPhase::Setup,
        BenchmarkPhaseCounts{},
        0U,
        result.effective_pipeline_depth
    );

    const bool warmup_completed =
        config.mode == BenchmarkMode::SynchronousLatency
            ? process_fused_synchronous_phase(
                pipeline,
                frames,
                BenchmarkPhase::Warmup,
                result.counts.warmup,
                result,
                duration_series,
                control,
                false,
                stop_status
            )
            : process_fused_streaming_phase(
                pipeline,
                frames,
                BenchmarkPhase::Warmup,
                result.counts.warmup,
                result,
                duration_series,
                control,
                false,
                stop_status
            );

    if (warmup_completed) {
        if (config.mode == BenchmarkMode::SynchronousLatency) {
            static_cast<void>(process_fused_synchronous_phase(
                pipeline,
                frames,
                BenchmarkPhase::Measurement,
                result.counts.measurement,
                result,
                duration_series,
                control,
                true,
                stop_status
            ));
        } else {
            static_cast<void>(process_fused_streaming_phase(
                pipeline,
                frames,
                BenchmarkPhase::Measurement,
                result.counts.measurement,
                result,
                duration_series,
                control,
                true,
                stop_status
            ));
        }
    }

    result.status = stop_status == BenchmarkCompletionStatus::NotStarted
        ? BenchmarkCompletionStatus::Completed
        : stop_status;

    finalize_statistics(result, duration_series);
    finalize_throughput(result);

    report_progress(
        control,
        BenchmarkPhase::Complete,
        result.counts.measurement,
        pipeline.in_flight_count(),
        result.effective_pipeline_depth
    );

    return result;
}

std::string format_benchmark_summary(
    const BenchmarkResult& result
) {
    std::ostringstream output{};
    output << std::fixed << std::setprecision(3);

    output << "Benchmark result\n"
           << "  status: "
           << benchmark_completion_status_name(result.status)
           << '\n'
           << "  successful: "
           << (result.successful() ? "true" : "false")
           << '\n'
           << "  target: "
           << target_kind_name(result.target)
           << '\n'
           << "  backend: "
           << target_backend_name(result.target)
           << '\n'
           << "  precision: "
           << target_precision_name(result.target)
           << '\n'
           << "  runtime: "
           << target_runtime_name(result.target);

    const std::string runtime_version =
        target_runtime_version(result.target);

    if (!runtime_version.empty()) {
        output << ' ' << runtime_version;
    }

    output << '\n'
           << "  artifact: "
           << target_artifact_path(result.target)
           << '\n'
           << "  mode: "
           << benchmark_mode_name(result.config.mode)
           << '\n'
           << "  timing scope: "
           << benchmark_timing_scope_name(result.config.timing_scope)
           << '\n'
           << "  effective pipeline depth: "
           << result.effective_pipeline_depth
           << '\n'
           << "  warmup requested/attempted/completed/failed: "
           << result.counts.warmup.requested << '/'
           << result.counts.warmup.attempted << '/'
           << result.counts.warmup.completed << '/'
           << result.counts.warmup.failed << '\n'
           << "  measured requested/attempted/submitted/completed/failed: "
           << result.counts.measurement.requested << '/'
           << result.counts.measurement.attempted << '/'
           << result.counts.measurement.submitted << '/'
           << result.counts.measurement.completed << '/'
           << result.counts.measurement.failed << '\n';

    if (result.primary_latency.has_value()) {
        const LatencyStatistics& latency = *result.primary_latency;

        output << "  primary latency samples: "
               << latency.sample_count << '\n'
               << "  primary latency min/mean/median/max ms: "
               << latency.minimum.count() << '/'
               << latency.mean.count() << '/'
               << latency.median.count() << '/'
               << latency.maximum.count() << '\n'
               << "  primary latency p90/p95/p99 ms: "
               << latency.p90.count() << '/'
               << latency.p95.count() << '/'
               << latency.p99.count() << '\n'
               << "  primary latency population stddev ms: "
               << latency.standard_deviation.count() << '\n';
    } else {
        output << "  primary latency: unavailable\n";
    }

    if (result.throughput.has_value()) {
        output << "  throughput fps: "
               << result.throughput->frames_per_second << '\n'
               << "  measured wall time ms: "
               << result.throughput->wall_time.count() << '\n';
    } else {
        output << "  throughput: unavailable\n";
    }

    if (!result.warnings.empty()) {
        output << "  warnings:\n";

        for (const std::string& warning : result.warnings) {
            output << "    - " << warning << '\n';
        }
    }

    if (!result.failures.empty() || result.omitted_failure_count != 0U) {
        output << "  failures:\n";

        for (const BenchmarkFailure& failure : result.failures) {
            output << "    - phase="
                   << benchmark_phase_name(failure.phase)
                   << " iteration="
                   << failure.iteration_index
                   << " operation="
                   << failure.operation
                   << " message="
                   << failure.message
                   << '\n';
        }

        if (result.omitted_failure_count != 0U) {
            output << "    - omitted additional failures: "
                   << result.omitted_failure_count
                   << '\n';
        }
    }

    return output.str();
}

std::string benchmark_result_to_json(
    const BenchmarkResult& result
) {
    std::ostringstream output{};

    output.put('{');
    write_json_key(output, "schema_version");
    output << 1 << ',';
    write_json_key(output, "duration_unit");
    write_json_string(output, "milliseconds");
    output.put(',');
    write_json_key(output, "status");
    write_json_string(
        output,
        benchmark_completion_status_name(result.status)
    );
    output.put(',');
    write_json_key(output, "successful");
    write_json_bool(output, result.successful());
    output.put(',');
    write_json_key(output, "effective_pipeline_depth");
    output << result.effective_pipeline_depth << ',';
    write_json_key(output, "config");
    write_benchmark_config_json(output, result.config);
    output.put(',');
    write_json_key(output, "target");
    write_target_json(output, result.target);
    output.put(',');

    write_json_key(output, "counts");
    output.put('{');
    write_json_key(output, "warmup");
    write_phase_counts_json(output, result.counts.warmup);
    output.put(',');
    write_json_key(output, "measurement");
    write_phase_counts_json(output, result.counts.measurement);
    output.put('}');
    output.put(',');

    write_json_key(output, "phase_durations_ms");
    output.put('{');
    write_json_key(output, "warmup");
    write_json_optional_duration(output, result.phase_durations.warmup);
    output.put(',');
    write_json_key(output, "measurement");
    write_json_optional_duration(output, result.phase_durations.measurement);
    output.put(',');
    write_json_key(output, "drain");
    write_json_optional_duration(output, result.phase_durations.drain);
    output.put('}');
    output.put(',');

    write_json_key(output, "primary_latency");
    write_latency_statistics_json(output, result.primary_latency);
    output.put(',');

    write_json_key(output, "stage_statistics");
    output.put('{');
    write_json_key(output, "queue_waiting");
    write_latency_statistics_json(
        output,
        result.stage_statistics.queue_waiting
    );
    output.put(',');
    write_json_key(output, "submission");
    write_latency_statistics_json(
        output,
        result.stage_statistics.submission
    );
    output.put(',');
    write_json_key(output, "preprocessing");
    write_latency_statistics_json(
        output,
        result.stage_statistics.preprocessing
    );
    output.put(',');
    write_json_key(output, "backend_execution");
    write_latency_statistics_json(
        output,
        result.stage_statistics.backend_execution
    );
    output.put(',');
    write_json_key(output, "postprocessing");
    write_latency_statistics_json(
        output,
        result.stage_statistics.postprocessing
    );
    output.put(',');
    write_json_key(output, "host_to_device");
    write_latency_statistics_json(
        output,
        result.stage_statistics.host_to_device
    );
    output.put(',');
    write_json_key(output, "device_execution");
    write_latency_statistics_json(
        output,
        result.stage_statistics.device_execution
    );
    output.put(',');
    write_json_key(output, "device_to_host");
    write_latency_statistics_json(
        output,
        result.stage_statistics.device_to_host
    );
    output.put(',');
    write_json_key(output, "completion_wait");
    write_latency_statistics_json(
        output,
        result.stage_statistics.completion_wait
    );
    output.put(',');
    write_json_key(output, "end_to_end");
    write_latency_statistics_json(
        output,
        result.stage_statistics.end_to_end
    );
    output.put('}');
    output.put(',');

    write_json_key(output, "throughput");

    if (result.throughput.has_value()) {
        output.put('{');
        write_json_key(output, "completed_frames");
        output << result.throughput->completed_frames << ',';
        write_json_key(output, "wall_time_ms");
        write_json_duration(output, result.throughput->wall_time);
        output.put(',');
        write_json_key(output, "frames_per_second");
        write_json_double(output, result.throughput->frames_per_second);
        output.put('}');
    } else {
        output << "null";
    }

    output.put(',');
    write_json_key(output, "samples");
    output.put('[');

    for (std::size_t index = 0U; index < result.samples.size(); ++index) {
        if (index != 0U) {
            output.put(',');
        }

        write_sample_json(output, result.samples[index]);
    }

    output.put(']');
    output.put(',');

    write_json_key(output, "failures");
    output.put('[');

    for (std::size_t index = 0U; index < result.failures.size(); ++index) {
        if (index != 0U) {
            output.put(',');
        }

        const BenchmarkFailure& failure = result.failures[index];
        output.put('{');
        write_json_key(output, "phase");
        write_json_string(output, benchmark_phase_name(failure.phase));
        output.put(',');
        write_json_key(output, "iteration_index");
        output << failure.iteration_index << ',';
        write_json_key(output, "source_frame_index");
        write_json_optional_integer(output, failure.source_frame_index);
        output.put(',');
        write_json_key(output, "target_sequence_number");
        write_json_optional_integer(
            output,
            failure.target_sequence_number
        );
        output.put(',');
        write_json_key(output, "operation");
        write_json_string(output, failure.operation);
        output.put(',');
        write_json_key(output, "message");
        write_json_string(output, failure.message);
        output.put('}');
    }

    output.put(']');
    output.put(',');
    write_json_key(output, "omitted_failure_count");
    output << result.omitted_failure_count << ',';
    write_json_key(output, "warnings");
    write_json_string_array(output, result.warnings);
    output.put('}');
    output.put('\n');

    return output.str();
}

void write_benchmark_json(
    const BenchmarkResult& result,
    const std::filesystem::path& output_path
) {
    write_text_file(
        output_path,
        benchmark_result_to_json(result)
    );
}

void write_benchmark_summary_csv(
    const std::vector<BenchmarkResult>& results,
    const std::filesystem::path& output_path
) {
    if (results.empty()) {
        throw std::invalid_argument(
            "write_benchmark_summary_csv requires at least one result."
        );
    }

    std::ostringstream output{};
    output
        << "target_type,backend,precision,runtime_name,runtime_version,"
        << "artifact_path,status,successful,mode,timing_scope,"
        << "effective_pipeline_depth,warmup_requested,warmup_completed,"
        << "warmup_failed,measured_requested,measured_attempted,"
        << "measured_submitted,measured_completed,measured_failed,"
        << "primary_sample_count,primary_min_ms,primary_mean_ms,"
        << "primary_median_ms,primary_stddev_ms,primary_p90_ms,"
        << "primary_p95_ms,primary_p99_ms,primary_max_ms,"
        << "measured_wall_ms,frames_per_second,warning_count,"
        << "recorded_failure_count,omitted_failure_count\n";

    for (const BenchmarkResult& result : results) {
        output
            << csv_escape(target_kind_name(result.target)) << ','
            << csv_escape(target_backend_name(result.target)) << ','
            << csv_escape(target_precision_name(result.target)) << ','
            << csv_escape(target_runtime_name(result.target)) << ','
            << csv_escape(target_runtime_version(result.target)) << ','
            << csv_escape(target_artifact_path(result.target)) << ','
            << csv_escape(
                benchmark_completion_status_name(result.status)
            ) << ','
            << (result.successful() ? "true" : "false") << ','
            << csv_escape(benchmark_mode_name(result.config.mode)) << ','
            << csv_escape(
                benchmark_timing_scope_name(result.config.timing_scope)
            ) << ','
            << result.effective_pipeline_depth << ','
            << result.counts.warmup.requested << ','
            << result.counts.warmup.completed << ','
            << result.counts.warmup.failed << ','
            << result.counts.measurement.requested << ','
            << result.counts.measurement.attempted << ','
            << result.counts.measurement.submitted << ','
            << result.counts.measurement.completed << ','
            << result.counts.measurement.failed << ',';

        if (result.primary_latency.has_value()) {
            output << result.primary_latency->sample_count;
        }

        output << ','
               << optional_stat_csv(
                    result.primary_latency,
                    &LatencyStatistics::minimum
                  ) << ','
               << optional_stat_csv(
                    result.primary_latency,
                    &LatencyStatistics::mean
                  ) << ','
               << optional_stat_csv(
                    result.primary_latency,
                    &LatencyStatistics::median
                  ) << ','
               << optional_stat_csv(
                    result.primary_latency,
                    &LatencyStatistics::standard_deviation
                  ) << ','
               << optional_stat_csv(
                    result.primary_latency,
                    &LatencyStatistics::p90
                  ) << ','
               << optional_stat_csv(
                    result.primary_latency,
                    &LatencyStatistics::p95
                  ) << ','
               << optional_stat_csv(
                    result.primary_latency,
                    &LatencyStatistics::p99
                  ) << ','
               << optional_stat_csv(
                    result.primary_latency,
                    &LatencyStatistics::maximum
                  ) << ','
               << optional_duration_csv(
                    result.phase_durations.measurement
                  ) << ',';

        if (result.throughput.has_value()) {
            output << std::setprecision(
                std::numeric_limits<double>::max_digits10
            ) << result.throughput->frames_per_second;
        }

        output << ','
               << result.warnings.size() << ','
               << result.failures.size() << ','
               << result.omitted_failure_count
               << '\n';
    }

    write_text_file(output_path, output.str());
}

void write_benchmark_samples_csv(
    const BenchmarkResult& result,
    const std::filesystem::path& output_path
) {
    std::ostringstream output{};
    output
        << "iteration_index,source_frame_index,target_sequence_number,"
        << "pipeline_slot_index,detection_count,used_cuda_graph,submitted,"
        << "completed,queue_waiting_ms,submission_ms,preprocessing_ms,"
        << "backend_execution_ms,postprocessing_ms,host_to_device_ms,"
        << "device_execution_ms,device_to_host_ms,completion_wait_ms,"
        << "end_to_end_ms\n";

    for (const BenchmarkSample& sample : result.samples) {
        output
            << sample.iteration_index << ','
            << sample.source_frame_index << ','
            << optional_uint64_csv(sample.target_sequence_number) << ','
            << optional_uint32_csv(sample.pipeline_slot_index) << ',';

        if (sample.detection_count.has_value()) {
            output << *sample.detection_count;
        }

        output << ','
               << optional_bool_csv(sample.used_cuda_graph) << ','
               << (sample.submitted ? "true" : "false") << ','
               << (sample.completed ? "true" : "false") << ','
               << optional_duration_csv(
                    sample.timings.queue_waiting
                  ) << ','
               << optional_duration_csv(
                    sample.timings.submission
                  ) << ','
               << optional_duration_csv(
                    sample.timings.preprocessing
                  ) << ','
               << optional_duration_csv(
                    sample.timings.backend_execution
                  ) << ','
               << optional_duration_csv(
                    sample.timings.postprocessing
                  ) << ','
               << optional_duration_csv(
                    sample.timings.host_to_device
                  ) << ','
               << optional_duration_csv(
                    sample.timings.device_execution
                  ) << ','
               << optional_duration_csv(
                    sample.timings.device_to_host
                  ) << ','
               << optional_duration_csv(
                    sample.timings.completion_wait
                  ) << ','
               << optional_duration_csv(
                    sample.timings.end_to_end
                  ) << '\n';
    }

    write_text_file(output_path, output.str());
}

}  // namespace edge
