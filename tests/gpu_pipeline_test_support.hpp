#pragma once

// Test-only entry points into the production CUDA kernels. These declarations
// are not installed and their definitions exist only in CUDA test builds.
#include "image_processor.hpp"
#include "postprocess.hpp"

#include <vector>

namespace edge::testing {

[[nodiscard]] LetterboxTransform preprocess_bgr(
    const cv::Mat& image,
    const PreprocessConfig& config,
    MutableTensorView destination
);

[[nodiscard]] PostprocessSummary postprocess(
    const std::vector<float>& raw,
    TensorElementType output_type,
    const LetterboxTransform& transform,
    const PostprocessConfig& config,
    DetectionBuffer& detections
);

}  // namespace edge::testing
