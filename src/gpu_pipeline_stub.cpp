#include "gpu_pipeline.hpp"

#include <stdexcept>
#include <utility>

namespace edge {

class NativeTensorRtPipeline::Impl final {};

namespace {

[[noreturn]] void throw_not_compiled() {
    throw std::runtime_error(
        "The fused CUDA/TensorRT pipeline is not compiled. Reconfigure with "
        "EDGE_ENABLE_TENSORRT=ON and EDGE_ENABLE_CUDA_PIPELINE=ON."
    );
}

}  // namespace

NativeTensorRtPipeline::NativeTensorRtPipeline(
    NativeTensorRtPipelineConfig
) {
    throw_not_compiled();
}

NativeTensorRtPipeline::~NativeTensorRtPipeline() = default;

NativeTensorRtPipeline::NativeTensorRtPipeline(
    NativeTensorRtPipeline&& other
) noexcept = default;

NativeTensorRtPipeline& NativeTensorRtPipeline::operator=(
    NativeTensorRtPipeline&& other
) noexcept = default;

const NativeTensorRtPipelineConfig& NativeTensorRtPipeline::config() const {
    throw_not_compiled();
}

const GpuPipelineRuntimeInfo& NativeTensorRtPipeline::runtime_info() const {
    throw_not_compiled();
}

GpuPipelineTicket NativeTensorRtPipeline::submit(
    const cv::Mat&
) {
    throw_not_compiled();
}

GpuPipelineTicket NativeTensorRtPipeline::submit_pinned_bgr(
    PinnedBgrImageView
) {
    throw_not_compiled();
}

GpuPipelineTicket NativeTensorRtPipeline::submit_device(
    DeviceImageView
) {
    throw_not_compiled();
}

bool NativeTensorRtPipeline::ready(
    GpuPipelineTicket
) const {
    throw_not_compiled();
}

GpuPipelineResult NativeTensorRtPipeline::collect(
    GpuPipelineTicket
) {
    throw_not_compiled();
}

GpuPipelineResult NativeTensorRtPipeline::run(
    const cv::Mat&
) {
    throw_not_compiled();
}

GpuPipelineResult NativeTensorRtPipeline::run_device(
    DeviceImageView
) {
    throw_not_compiled();
}

std::size_t NativeTensorRtPipeline::pipeline_depth() const noexcept {
    return 0U;
}

std::size_t NativeTensorRtPipeline::in_flight_count() const noexcept {
    return 0U;
}

NativeTensorRtPipeline::Impl& NativeTensorRtPipeline::implementation() {
    throw_not_compiled();
}

const NativeTensorRtPipeline::Impl&
NativeTensorRtPipeline::implementation() const {
    throw_not_compiled();
}

}  // namespace edge
