#include "gpu_pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>

namespace edge {
namespace {


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

[[nodiscard]] bool supported_device_format(
    DeviceImageFormat format
) noexcept {
    switch (format) {
        case DeviceImageFormat::Bgr8:
        case DeviceImageFormat::Rgb8:
        case DeviceImageFormat::Nv12:
            return true;
    }

    return false;
}

[[nodiscard]] bool supported_yuv_matrix(
    YuvColorMatrix matrix
) noexcept {
    switch (matrix) {
        case YuvColorMatrix::Bt601:
        case YuvColorMatrix::Bt709:
            return true;
    }

    return false;
}

[[nodiscard]] bool supported_yuv_range(
    YuvRange range
) noexcept {
    switch (range) {
        case YuvRange::Limited:
        case YuvRange::Full:
            return true;
    }

    return false;
}

[[nodiscard]] bool checked_three_channel_row_bytes(
    int width,
    std::size_t& row_bytes
) noexcept {
    if (width <= 0) {
        return false;
    }

    constexpr std::size_t maximum =
        std::numeric_limits<std::size_t>::max();

    const std::size_t unsigned_width =
        static_cast<std::size_t>(width);

    if (unsigned_width > maximum / 3U) {
        return false;
    }

    row_bytes = unsigned_width * 3U;
    return true;
}

}  // namespace

bool PinnedBgrImageView::valid() const noexcept {
    std::size_t required_row_bytes = 0U;

    return data != nullptr
        && height > 0
        && checked_three_channel_row_bytes(width, required_row_bytes)
        && row_stride_bytes >= required_row_bytes;
}

bool DeviceImageView::valid() const noexcept {
    if (
        plane0 == nullptr
        || width <= 0
        || height <= 0
        || device_id < 0
        || !supported_device_format(format)
        || !supported_yuv_matrix(yuv_matrix)
        || !supported_yuv_range(yuv_range)
    ) {
        return false;
    }

    if (format == DeviceImageFormat::Nv12) {
        return plane1 != nullptr
            && (width & 1) == 0
            && (height & 1) == 0
            && plane0_pitch_bytes
                >= static_cast<std::size_t>(width)
            && plane1_pitch_bytes
                >= static_cast<std::size_t>(width);
    }

    std::size_t required_row_bytes = 0U;

    return plane1 == nullptr
        && checked_three_channel_row_bytes(width, required_row_bytes)
        && plane0_pitch_bytes >= required_row_bytes;
}

void NativeTensorRtPipelineConfig::validate() const {
    if (engine_path.empty()) {
        throw std::invalid_argument(
            "NativeTensorRtPipelineConfig.engine_path must not be empty."
        );
    }

    if (
        ascii_lowercase(engine_path.extension().string())
        != ".engine"
    ) {
        throw std::invalid_argument(
            "NativeTensorRtPipelineConfig.engine_path must use .engine."
        );
    }

    if (expected_input_name.empty()
        || expected_input_name.find('\0') != std::string::npos) {
        throw std::invalid_argument(
            "NativeTensorRtPipelineConfig.expected_input_name is empty or "
            "contains an embedded NUL."
        );
    }

    if (expected_output_name.empty()
        || expected_output_name.find('\0') != std::string::npos) {
        throw std::invalid_argument(
            "NativeTensorRtPipelineConfig.expected_output_name is empty or "
            "contains an embedded NUL."
        );
    }

    if (device_id < 0) {
        throw std::invalid_argument(
            "NativeTensorRtPipelineConfig.device_id must be non-negative."
        );
    }

    if (pipeline_depth == 0U || pipeline_depth > 8U) {
        throw std::invalid_argument(
            "NativeTensorRtPipelineConfig.pipeline_depth must be inside "
            "[1, 8]."
        );
    }

    if (!maximum_host_image_size.valid()) {
        throw std::invalid_argument(
            "NativeTensorRtPipelineConfig.maximum_host_image_size must be "
            "strictly positive."
        );
    }

    preprocess.validate();
    postprocess.validate();

    if (
        preprocess.network_size.width != 640
        || preprocess.network_size.height != 640
    ) {
        throw std::invalid_argument(
            "The fused pipeline requires the fixed 640x640 model input."
        );
    }

    if (preprocess.interpolation != ResizeInterpolation::Linear) {
        throw std::invalid_argument(
            "The fused CUDA pipeline currently requires linear resize."
        );
    }

    if (
        postprocess.malformed_candidate_policy
        != MalformedCandidatePolicy::Discard
    ) {
        throw std::invalid_argument(
            "The fused CUDA postprocessor requires malformed-candidate "
            "discard policy."
        );
    }

    if (require_cuda_graph && !enable_cuda_graph) {
        throw std::invalid_argument(
            "require_cuda_graph cannot be true when enable_cuda_graph is "
            "false."
        );
    }
}

}  // namespace edge
