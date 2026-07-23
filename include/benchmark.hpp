#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "gpu_pipeline.hpp"
#include "image_processor.hpp"
#include "inference_engine.hpp"
#include "postprocess.hpp"

namespace edge {

/**
 * @brief Floating-point millisecond duration used by benchmark reports.
 *
 * benchmark.cpp measures elapsed time with std::chrono::steady_clock and
 * converts the resulting clock durations to this explicit reporting unit.
 */
using BenchmarkDuration = std::chrono::duration<double, std::milli>;

/**
 * @brief High-level workload pattern used by one benchmark run.
 */
enum class BenchmarkMode : std::uint8_t {
    /**
     * Execute one frame to completion before beginning the next frame.
     * The effective in-flight depth is always one.
     */
    SynchronousLatency,

    /**
     * Keep a bounded number of frames in flight and drain every accepted frame
     * before the measured interval ends.
     */
    StreamingThroughput
};

/**
 * @brief Portion of the pipeline selected as the primary latency metric.
 */
enum class BenchmarkTimingScope : std::uint8_t {
    /**
     * Time only the runtime's public execution call.
     *
     * For InferenceEngine this is InferenceEngine::run(). GPU backends include
     * any transfers and synchronization required by that synchronous API. The
     * fused NativeTensorRtPipeline does not expose a network-kernel-only public
     * timing boundary, so this scope is unsupported for that target.
     */
    BackendExecution,

    /**
     * Time the complete image-to-final-detections operation exposed by the
     * selected target.
     */
    EndToEnd
};

/**
 * @brief Lifecycle phase in which progress or a failure was observed.
 */
enum class BenchmarkPhase : std::uint8_t {
    Setup,
    Warmup,
    Measurement,
    Drain,
    Complete
};

/**
 * @brief Policy applied after one frame operation fails.
 */
enum class BenchmarkFailurePolicy : std::uint8_t {
    /**
     * Stop accepting new frames, drain outstanding work, and return.
     */
    StopOnFirstFailure,

    /**
     * Record the failure and attempt the remaining requested frames.
     */
    ContinueAfterFailure
};

/**
 * @brief Final state of one benchmark invocation.
 */
enum class BenchmarkCompletionStatus : std::uint8_t {
    NotStarted,
    Completed,
    Cancelled,
    StoppedOnFailure
};

/**
 * @brief Return the stable reporting name for a benchmark mode.
 */
[[nodiscard]] constexpr std::string_view benchmark_mode_name(
    BenchmarkMode mode
) noexcept {
    switch (mode) {
        case BenchmarkMode::SynchronousLatency:
            return "synchronous_latency";

        case BenchmarkMode::StreamingThroughput:
            return "streaming_throughput";
    }

    return "unknown";
}

/**
 * @brief Return the stable reporting name for a timing scope.
 */
[[nodiscard]] constexpr std::string_view benchmark_timing_scope_name(
    BenchmarkTimingScope scope
) noexcept {
    switch (scope) {
        case BenchmarkTimingScope::BackendExecution:
            return "backend_execution";

        case BenchmarkTimingScope::EndToEnd:
            return "end_to_end";
    }

    return "unknown";
}

/**
 * @brief Return the stable reporting name for a benchmark phase.
 */
[[nodiscard]] constexpr std::string_view benchmark_phase_name(
    BenchmarkPhase phase
) noexcept {
    switch (phase) {
        case BenchmarkPhase::Setup:
            return "setup";

        case BenchmarkPhase::Warmup:
            return "warmup";

        case BenchmarkPhase::Measurement:
            return "measurement";

        case BenchmarkPhase::Drain:
            return "drain";

        case BenchmarkPhase::Complete:
            return "complete";
    }

    return "unknown";
}

/**
 * @brief Return the stable reporting name for a failure policy.
 */
[[nodiscard]] constexpr std::string_view benchmark_failure_policy_name(
    BenchmarkFailurePolicy policy
) noexcept {
    switch (policy) {
        case BenchmarkFailurePolicy::StopOnFirstFailure:
            return "stop_on_first_failure";

        case BenchmarkFailurePolicy::ContinueAfterFailure:
            return "continue_after_failure";
    }

    return "unknown";
}

/**
 * @brief Return the stable reporting name for a completion status.
 */
[[nodiscard]] constexpr std::string_view benchmark_completion_status_name(
    BenchmarkCompletionStatus status
) noexcept {
    switch (status) {
        case BenchmarkCompletionStatus::NotStarted:
            return "not_started";

        case BenchmarkCompletionStatus::Completed:
            return "completed";

        case BenchmarkCompletionStatus::Cancelled:
            return "cancelled";

        case BenchmarkCompletionStatus::StoppedOnFailure:
            return "stopped_on_failure";
    }

    return "unknown";
}

/**
 * @brief Immutable instructions for one benchmark invocation.
 *
 * warmup_iterations and measured_iterations count frames rather than batches.
 * Warm-up work is executed through the same target path but is never included
 * in measured latency or throughput statistics.
 *
 * pipeline_depth is the requested maximum number of in-flight frames for
 * StreamingThroughput and must be inside [1, 8]. SynchronousLatency always
 * reports an effective depth of one. The default of two is the project's
 * balanced live-stream configuration.
 */
struct BenchmarkConfig {
    /**
     * Workload pattern used by the benchmark.
     */
    BenchmarkMode mode = BenchmarkMode::SynchronousLatency;

    /**
     * Stage selected as the primary latency statistic.
     */
    BenchmarkTimingScope timing_scope = BenchmarkTimingScope::EndToEnd;

    /**
     * Number of untimed frames executed before measurement begins.
     */
    std::uint64_t warmup_iterations = 10U;

    /**
     * Number of frames requested for the measured phase.
     */
    std::uint64_t measured_iterations = 100U;

    /**
     * Requested streaming in-flight limit.
     *
     * The valid range is [1, 8]. The default value is two.
     */
    std::size_t pipeline_depth = 2U;

    /**
     * Record additional available stage boundaries in addition to the primary
     * latency scope.
     *
     * Disabling this minimizes host timing instrumentation.
     */
    bool collect_stage_timings = true;

    /**
     * Retain per-frame measured samples in BenchmarkResult::samples.
     *
     * Aggregate statistics are still computed when retention is disabled.
     */
    bool retain_raw_samples = false;

    /**
     * Behavior after a frame operation throws or returns an invalid result.
     */
    BenchmarkFailurePolicy failure_policy =
        BenchmarkFailurePolicy::StopOnFirstFailure;

    /**
     * Maximum number of detailed failure records retained in the result.
     *
     * A value of zero suppresses detailed records while preserving failure
     * counters.
     */
    std::size_t maximum_recorded_failures = 16U;

    /**
     * @brief Validate enum values, counts, depth, and overflow-sensitive totals.
     *
     * @throws std::invalid_argument if any field is unsupported or
     *         inconsistent.
     * @throws std::overflow_error if configured iteration totals overflow.
     */
    void validate() const;
};

/**
 * @brief Counters for one warm-up or measurement phase.
 *
 * requested is copied from configuration.
 *
 * attempted increments when the benchmark begins handling a frame.
 *
 * submitted increments after the target accepts the operation.
 *
 * completed and failed are terminal outcomes. For an ordinary completed phase,
 * every attempted frame is either completed or failed and every submitted
 * asynchronous frame has been collected or failed.
 */
struct BenchmarkPhaseCounts {
    /**
     * Number of frames requested for this phase.
     */
    std::uint64_t requested = 0U;

    /**
     * Number of frames whose processing was started.
     */
    std::uint64_t attempted = 0U;

    /**
     * Number of frame operations accepted by the target.
     */
    std::uint64_t submitted = 0U;

    /**
     * Number of frames that produced a valid final result.
     */
    std::uint64_t completed = 0U;

    /**
     * Number of attempted frames that did not produce a valid result.
     */
    std::uint64_t failed = 0U;
};

/**
 * @brief Separate counters for untimed warm-up and measured work.
 */
struct BenchmarkExecutionCounts {
    /**
     * Untimed warm-up phase counters.
     */
    BenchmarkPhaseCounts warmup{};

    /**
     * Timed measurement phase counters.
     */
    BenchmarkPhaseCounts measurement{};
};

/**
 * @brief Optional timing values associated with one measured frame.
 *
 * A missing value means that the selected public target API did not expose a
 * trustworthy boundary for that stage.
 *
 * Missing values must never be serialized as zero-duration measurements.
 */
struct BenchmarkStageDurations {
    /**
     * Time a ready source frame waited before a target slot became available.
     */
    std::optional<BenchmarkDuration> queue_waiting{};

    /**
     * Host time spent accepting or enqueueing the frame into the target.
     */
    std::optional<BenchmarkDuration> submission{};

    /**
     * CPU image preprocessing duration, when separately observable.
     */
    std::optional<BenchmarkDuration> preprocessing{};

    /**
     * Public backend-execution duration.
     *
     * For GPU InferenceEngine backends this includes required H2D/D2H work and
     * synchronization owned by InferenceEngine::run().
     */
    std::optional<BenchmarkDuration> backend_execution{};

    /**
     * CPU detection postprocessing duration, when separately observable.
     */
    std::optional<BenchmarkDuration> postprocessing{};

    /**
     * Separately measured host-to-device transfer duration, when available.
     */
    std::optional<BenchmarkDuration> host_to_device{};

    /**
     * Separately measured device execution duration, when available.
     */
    std::optional<BenchmarkDuration> device_execution{};

    /**
     * Separately measured device-to-host transfer duration, when available.
     */
    std::optional<BenchmarkDuration> device_to_host{};

    /**
     * Time spent waiting for an accepted asynchronous frame to complete.
     */
    std::optional<BenchmarkDuration> completion_wait{};

    /**
     * Complete latency from the benchmark's frame-arrival boundary to a valid
     * final detection result.
     *
     * Streaming mode includes benchmark-observed queue waiting rather than
     * silently discarding it.
     */
    std::optional<BenchmarkDuration> end_to_end{};
};

/**
 * @brief One retained measured-frame record.
 *
 * Warm-up frames are never stored here.
 *
 * Failed measured attempts may be stored with completed=false and whatever
 * partial timings were safely observed.
 */
struct BenchmarkSample {
    /**
     * Zero-based index inside the measured phase.
     */
    std::uint64_t iteration_index = 0U;

    /**
     * Index of the source cv::Mat selected from the caller's frame vector.
     */
    std::size_t source_frame_index = 0U;

    /**
     * Target sequence number when the selected runtime exposes one.
     */
    std::optional<std::uint64_t> target_sequence_number{};

    /**
     * Native fused-pipeline slot index when available.
     */
    std::optional<std::uint32_t> pipeline_slot_index{};

    /**
     * Number of final detections when output validation or postprocessing ran.
     */
    std::optional<std::uint32_t> detection_count{};

    /**
     * Whether the fused path reported CUDA-graph execution for this frame.
     */
    std::optional<bool> used_cuda_graph{};

    /**
     * Whether the target accepted this frame operation.
     */
    bool submitted = false;

    /**
     * Whether this frame produced a valid final result.
     */
    bool completed = false;

    /**
     * Available per-stage durations for this frame.
     */
    BenchmarkStageDurations timings{};
};

/**
 * @brief Deterministic aggregate statistics for one non-empty duration series.
 *
 * Percentiles use linear interpolation over ascending samples:
 *
 *     position = (N - 1) * percentile
 *
 * The values at floor(position) and ceil(position) are interpolated by the
 * fractional part.
 *
 * Standard deviation is the population standard deviation with divisor N.
 *
 * For one sample, every percentile equals that sample and standard deviation
 * is zero.
 */
struct LatencyStatistics {
    /**
     * Number of successful samples included in the statistics.
     */
    std::uint64_t sample_count = 0U;

    /**
     * Smallest observed duration.
     */
    BenchmarkDuration minimum{};

    /**
     * Largest observed duration.
     */
    BenchmarkDuration maximum{};

    /**
     * Arithmetic mean duration.
     */
    BenchmarkDuration mean{};

    /**
     * 50th percentile using the documented interpolation rule.
     */
    BenchmarkDuration median{};

    /**
     * Population standard deviation.
     */
    BenchmarkDuration standard_deviation{};

    /**
     * 90th percentile.
     */
    BenchmarkDuration p90{};

    /**
     * 95th percentile.
     */
    BenchmarkDuration p95{};

    /**
     * 99th percentile.
     */
    BenchmarkDuration p99{};
};

/**
 * @brief Optional aggregate statistics for every supported timing boundary.
 */
struct BenchmarkStageStatistics {
    /**
     * Aggregate queue-waiting statistics.
     */
    std::optional<LatencyStatistics> queue_waiting{};

    /**
     * Aggregate target-submission statistics.
     */
    std::optional<LatencyStatistics> submission{};

    /**
     * Aggregate CPU preprocessing statistics.
     */
    std::optional<LatencyStatistics> preprocessing{};

    /**
     * Aggregate public backend-execution statistics.
     */
    std::optional<LatencyStatistics> backend_execution{};

    /**
     * Aggregate CPU postprocessing statistics.
     */
    std::optional<LatencyStatistics> postprocessing{};

    /**
     * Aggregate separately observed H2D statistics.
     */
    std::optional<LatencyStatistics> host_to_device{};

    /**
     * Aggregate separately observed device-execution statistics.
     */
    std::optional<LatencyStatistics> device_execution{};

    /**
     * Aggregate separately observed D2H statistics.
     */
    std::optional<LatencyStatistics> device_to_host{};

    /**
     * Aggregate asynchronous completion-wait statistics.
     */
    std::optional<LatencyStatistics> completion_wait{};

    /**
     * Aggregate complete image-to-detections statistics.
     */
    std::optional<LatencyStatistics> end_to_end{};
};

/**
 * @brief Sustained measured-phase throughput derived from completed frames.
 *
 * frames_per_second is calculated as completed_frames divided by wall-clock
 * seconds.
 *
 * The wall interval begins at the first measured frame-arrival boundary and
 * ends only after the final accepted measured frame is completed or failed and
 * all outstanding measured tickets are drained.
 */
struct ThroughputStatistics {
    /**
     * Number of successfully completed measured frames in the numerator.
     */
    std::uint64_t completed_frames = 0U;

    /**
     * Complete measured wall-clock interval.
     */
    BenchmarkDuration wall_time{};

    /**
     * completed_frames divided by wall_time expressed in seconds.
     */
    double frames_per_second = 0.0;
};

/**
 * @brief Wall-clock durations for benchmark lifecycle phases.
 */
struct BenchmarkPhaseDurations {
    /**
     * Complete warm-up interval when warm-up work was attempted.
     */
    std::optional<BenchmarkDuration> warmup{};

    /**
     * Complete measured interval when measured work was attempted.
     */
    std::optional<BenchmarkDuration> measurement{};

    /**
     * Cleanup or drain interval when outstanding asynchronous work existed.
     */
    std::optional<BenchmarkDuration> drain{};
};

/**
 * @brief Detailed description of one recorded benchmark failure.
 */
struct BenchmarkFailure {
    /**
     * Lifecycle phase in which the failure occurred.
     */
    BenchmarkPhase phase = BenchmarkPhase::Setup;

    /**
     * Zero-based iteration inside the associated phase.
     */
    std::uint64_t iteration_index = 0U;

    /**
     * Source-frame index associated with the failure, when applicable.
     */
    std::optional<std::size_t> source_frame_index{};

    /**
     * Target sequence number associated with the failure, when available.
     */
    std::optional<std::uint64_t> target_sequence_number{};

    /**
     * Short stable name of the operation that failed.
     */
    std::string operation{};

    /**
     * Owning diagnostic message.
     */
    std::string message{};
};

/**
 * @brief Snapshot of generic-engine and CPU pipeline configuration and
 *        metadata.
 */
struct InferenceEngineBenchmarkTargetInfo {
    /**
     * Validated construction configuration copied from the engine.
     */
    InferenceEngineConfig engine_config{};

    /**
     * Runtime, provider, tensor, and persistent-buffer information.
     */
    InferenceRuntimeInfo runtime_info{};

    /**
     * Preprocessing configuration used by the benchmark.
     */
    PreprocessConfig preprocess_config{};

    /**
     * Postprocessing configuration used by the benchmark.
     */
    PostprocessConfig postprocess_config{};
};

/**
 * @brief Snapshot of fused native TensorRT pipeline configuration and metadata.
 */
struct NativeTensorRtPipelineBenchmarkTargetInfo {
    /**
     * Validated construction configuration copied from the pipeline.
     */
    NativeTensorRtPipelineConfig pipeline_config{};

    /**
     * TensorRT, tensor, memory, CUDA-graph, and postprocessing metadata.
     */
    GpuPipelineRuntimeInfo runtime_info{};
};

/**
 * @brief Owning target snapshot for exactly one benchmark implementation.
 */
using BenchmarkTargetInfo = std::variant<
    InferenceEngineBenchmarkTargetInfo,
    NativeTensorRtPipelineBenchmarkTargetInfo
>;

/**
 * @brief Progress snapshot delivered synchronously to an optional callback.
 */
struct BenchmarkProgress {
    /**
     * Current lifecycle phase.
     */
    BenchmarkPhase phase = BenchmarkPhase::Setup;

    /**
     * Counters for the phase currently being reported.
     */
    BenchmarkPhaseCounts counts{};

    /**
     * Current number of accepted asynchronous frames not yet collected.
     */
    std::size_t in_flight = 0U;

    /**
     * Effective in-flight limit used by this run.
     */
    std::size_t effective_pipeline_depth = 1U;
};

/**
 * @brief Return true to request cooperative benchmark cancellation.
 */
using BenchmarkCancellationCallback = std::function<bool()>;

/**
 * @brief Receive benchmark progress on the benchmark's calling thread.
 */
using BenchmarkProgressCallback = std::function<void(
    const BenchmarkProgress&
)>;

/**
 * @brief Optional cooperative cancellation and progress hooks.
 *
 * Callbacks execute synchronously on the thread that called run_benchmark().
 *
 * They must not retain references to the supplied BenchmarkProgress object.
 *
 * Callback exceptions are not benchmark-target failures and are allowed to
 * propagate.
 */
struct BenchmarkControl {
    /**
     * Empty means the benchmark is not externally cancellable.
     */
    BenchmarkCancellationCallback should_cancel{};

    /**
     * Empty means progress notifications are disabled.
     */
    BenchmarkProgressCallback on_progress{};
};

/**
 * @brief Complete owning result of one benchmark invocation.
 *
 * primary_latency corresponds to BenchmarkConfig::timing_scope.
 *
 * It is empty when no successful measured sample exposed that timing boundary.
 *
 * samples is populated only when retain_raw_samples=true.
 *
 * failures may be truncated at maximum_recorded_failures.
 * omitted_failure_count records the remainder.
 */
struct BenchmarkResult {
    /**
     * Configuration snapshot used for this invocation.
     */
    BenchmarkConfig config{};

    /**
     * Owning target configuration and runtime snapshot.
     */
    BenchmarkTargetInfo target{};

    /**
     * Final benchmark lifecycle status.
     */
    BenchmarkCompletionStatus status =
        BenchmarkCompletionStatus::NotStarted;

    /**
     * Actual in-flight limit used after applying mode and target constraints.
     */
    std::size_t effective_pipeline_depth = 1U;

    /**
     * Separate warm-up and measured execution counters.
     */
    BenchmarkExecutionCounts counts{};

    /**
     * Wall-clock durations for benchmark phases.
     */
    BenchmarkPhaseDurations phase_durations{};

    /**
     * Aggregate statistics for the configured primary timing scope.
     */
    std::optional<LatencyStatistics> primary_latency{};

    /**
     * Aggregate statistics for every separately available stage.
     */
    BenchmarkStageStatistics stage_statistics{};

    /**
     * Measured-phase throughput when at least one frame completed.
     */
    std::optional<ThroughputStatistics> throughput{};

    /**
     * Per-frame measured records retained only when requested.
     */
    std::vector<BenchmarkSample> samples{};

    /**
     * Detailed failures retained up to the configured limit.
     */
    std::vector<BenchmarkFailure> failures{};

    /**
     * Failures counted but omitted because the detailed-record limit was
     * reached.
     */
    std::uint64_t omitted_failure_count = 0U;

    /**
     * Non-fatal unsupported-metric, fallback, or interpretation notes.
     */
    std::vector<std::string> warnings{};

    /**
     * @brief Return true only for a complete, uncancelled, failure-free run.
     *
     * A successful result has status Completed, completes every requested
     * measured frame, records no measured failures, and exposes the configured
     * primary latency statistic.
     */
    [[nodiscard]] bool successful() const noexcept;
};

/**
 * @brief Calculate deterministic statistics for one duration series.
 *
 * @return Empty when samples is empty; otherwise statistics following the
 *         interpolation and population-standard-deviation rules documented by
 *         LatencyStatistics.
 *
 * @throws std::invalid_argument if any duration is negative or non-finite.
 */
[[nodiscard]] std::optional<LatencyStatistics>
calculate_latency_statistics(
    const std::vector<BenchmarkDuration>& samples
);

/**
 * @brief Benchmark the generic preprocessing, inference, and postprocessing
 *        pipeline.
 */
[[nodiscard]] BenchmarkResult run_benchmark(
    InferenceEngine& engine,
    ImageProcessor& image_processor,
    const Postprocessor& postprocessor,
    const std::vector<cv::Mat>& frames,
    const BenchmarkConfig& config,
    const BenchmarkControl& control = {}
);

/**
 * @brief Benchmark the fused native TensorRT image-to-detections pipeline.
 *
 * SynchronousLatency executes one frame completely before starting the next.
 *
 * StreamingThroughput uses the pipeline's asynchronous submit/collect API with
 * bounded in-flight work and drains every accepted ticket before returning.
 *
 * Only BenchmarkTimingScope::EndToEnd is supported because the fused public
 * interface does not expose trustworthy independent preprocessing, H2D,
 * TensorRT-kernel, postprocessing, or D2H timing boundaries.
 *
 * The supplied pipeline must have no outstanding tickets when benchmarking
 * begins. The requested benchmark pipeline depth must not exceed the physical
 * depth configured when the pipeline was constructed.
 *
 * @throws std::invalid_argument if configuration, frames, timing scope,
 *         requested pipeline depth, or initial pipeline state is invalid.
 * @throws std::runtime_error for setup failures that prevent the benchmark
 *         result from being initialized. Per-frame failures follow the
 *         configured failure policy.
 */
[[nodiscard]] BenchmarkResult run_benchmark(
    NativeTensorRtPipeline& pipeline,
    const std::vector<cv::Mat>& frames,
    const BenchmarkConfig& config,
    const BenchmarkControl& control = {}
);

/**
 * @brief Format one deterministic human-readable benchmark report.
 */
[[nodiscard]] std::string format_benchmark_summary(
    const BenchmarkResult& result
);

/**
 * @brief Serialize one benchmark result to a JSON document.
 *
 * Unsupported optional metrics are emitted as null rather than zero.
 */
[[nodiscard]] std::string benchmark_result_to_json(
    const BenchmarkResult& result
);

/**
 * @brief Write one benchmark result as UTF-8 JSON.
 *
 * The destination is replaced.
 *
 * Parent directories are created when required.
 *
 * @throws std::runtime_error if directories or the output file cannot be
 *         created, written, or finalized.
 */
void write_benchmark_json(
    const BenchmarkResult& result,
    const std::filesystem::path& output_path
);

/**
 * @brief Write one comparison-oriented CSV row per result.
 *
 * The destination is replaced and a header row is always written.
 *
 * Unsupported metrics are represented by empty fields rather than numeric
 * zero.
 *
 * @throws std::invalid_argument if results is empty.
 * @throws std::runtime_error if the destination cannot be written.
 */
void write_benchmark_summary_csv(
    const std::vector<BenchmarkResult>& results,
    const std::filesystem::path& output_path
);

/**
 * @brief Write retained per-frame samples for one result as CSV.
 *
 * The destination is replaced.
 *
 * The function writes a valid header-only file when raw retention was disabled
 * or no measured sample was retained.
 *
 * Unsupported stage timings are represented by empty fields.
 *
 * @throws std::runtime_error if the destination cannot be written.
 */
void write_benchmark_samples_csv(
    const BenchmarkResult& result,
    const std::filesystem::path& output_path
);

}  // namespace edge
