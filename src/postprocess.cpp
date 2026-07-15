#include "postprocess.hpp"

#include "image_processor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace edge {
namespace {

constexpr std::size_t kXMinIndex = 0U;
constexpr std::size_t kYMinIndex = 1U;
constexpr std::size_t kXMaxIndex = 2U;
constexpr std::size_t kYMaxIndex = 3U;
constexpr std::size_t kConfidenceIndex = 4U;
constexpr std::size_t kClassIdIndex = 5U;

/**
 * @brief Return true when a floating-point value is neither NaN nor infinity.
 */
[[nodiscard]] bool is_finite(
    float value
) noexcept {
    return std::isfinite(value) != 0;
}

/**
 * @brief Check whether a malformed-candidate policy enum value is supported.
 */
[[nodiscard]] bool is_supported_malformed_policy(
    MalformedCandidatePolicy policy
) noexcept {
    switch (policy) {
        case MalformedCandidatePolicy::Discard:
        case MalformedCandidatePolicy::Throw:
            return true;
    }

    return false;
}

/**
 * @brief Return true when every coordinate in one box is finite.
 */
[[nodiscard]] bool has_finite_coordinates(
    const BoundingBox& box
) noexcept {
    return is_finite(box.x_min)
        && is_finite(box.y_min)
        && is_finite(box.x_max)
        && is_finite(box.y_max);
}

/**
 * @brief Validate letterbox metadata before it is used for coordinate math.
 */
void validate_letterbox_transform(
    const LetterboxTransform& transform
) {
    if (!transform.valid()) {
        throw std::invalid_argument(
            "The letterbox transform is structurally invalid."
        );
    }

    if (
        !is_finite(transform.scale_x)
        || !is_finite(transform.scale_y)
    ) {
        throw std::invalid_argument(
            "The letterbox transform contains a non-finite scale."
        );
    }
}

}  // namespace

/**
 * @brief Validate thresholds, limits, class range, and malformed-row policy.
 */
void PostprocessConfig::validate() const {
    if (
        !is_finite(confidence_threshold)
        || confidence_threshold < 0.0F
        || confidence_threshold > 1.0F
    ) {
        throw std::invalid_argument(
            "PostprocessConfig.confidence_threshold must be finite and "
            "inside [0, 1]."
        );
    }

    if (
        maximum_detections == 0U
        || maximum_detections > kModelMaximumDetections
    ) {
        throw std::invalid_argument(
            "PostprocessConfig.maximum_detections must be inside [1, 300]."
        );
    }

    constexpr std::size_t maximum_representable_class_count =
        static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max()
        ) + 1U;

    if (
        class_count == 0U
        || class_count > maximum_representable_class_count
    ) {
        throw std::invalid_argument(
            "PostprocessConfig.class_count must be positive and every class "
            "ID must fit in std::int32_t."
        );
    }

    if (
        !is_finite(class_id_tolerance)
        || class_id_tolerance < 0.0F
        || class_id_tolerance >= 0.5F
    ) {
        throw std::invalid_argument(
            "PostprocessConfig.class_id_tolerance must be finite and inside "
            "[0, 0.5)."
        );
    }

    if (
        !is_supported_malformed_policy(
            malformed_candidate_policy
        )
    ) {
        throw std::invalid_argument(
            "PostprocessConfig.malformed_candidate_policy is invalid."
        );
    }
}

/**
 * @brief Construct a postprocessor with a validated immutable-per-call config.
 */
Postprocessor::Postprocessor(
    PostprocessConfig config
)
    : config_(std::move(config)) {
    config_.validate();
}

/**
 * @brief Return the active postprocessing configuration.
 */
const PostprocessConfig& Postprocessor::config() const noexcept {
    return config_;
}

/**
 * @brief Install a new configuration after validating it completely.
 */
void Postprocessor::set_config(
    PostprocessConfig config
) {
    config.validate();
    config_ = std::move(config);
}

/**
 * @brief Decode, threshold, restore, clip, and store one output tensor.
 */
PostprocessSummary Postprocessor::process(
    DetectionTensorView output,
    const LetterboxTransform& transform,
    DetectionBuffer& detections
) const {
    // Clearing first guarantees that stale detections are never observed after
    // this call, including when structural validation throws an exception.
    detections.clear();

    validate_output_view(output);
    validate_letterbox_transform(transform);

    PostprocessSummary summary{};

    for (
        std::size_t candidate_index = 0U;
        candidate_index < output.candidate_count;
        ++candidate_index
    ) {
        ++summary.candidates_examined;

        const float* const candidate =
            output.candidate(candidate_index);

        const float confidence =
            candidate[kConfidenceIndex];

        // A non-finite confidence cannot participate in threshold comparison
        // safely and indicates malformed backend output.
        if (!is_finite(confidence)) {
            handle_malformed_candidate(
                candidate_index,
                "confidence is not finite",
                summary
            );
            continue;
        }

        // Ultralytics keeps end-to-end rows only when confidence is strictly
        // greater than the threshold, so equality is rejected for parity.
        if (!(confidence > config_.confidence_threshold)) {
            ++summary.rejected_by_confidence;
            continue;
        }

        if (confidence > 1.0F) {
            handle_malformed_candidate(
                candidate_index,
                "confidence is outside [0, 1]",
                summary
            );
            continue;
        }

        // Stop before decoding another accepted row once the configured output
        // limit has been filled. Model row order is preserved deliberately.
        if (detections.size() >= config_.maximum_detections) {
            summary.maximum_reached = true;
            break;
        }

        if (
            !is_finite(candidate[kXMinIndex])
            || !is_finite(candidate[kYMinIndex])
            || !is_finite(candidate[kXMaxIndex])
            || !is_finite(candidate[kYMaxIndex])
        ) {
            handle_malformed_candidate(
                candidate_index,
                "one or more box coordinates are not finite",
                summary
            );
            continue;
        }

        std::int32_t class_id = -1;

        if (
            !decode_class_id(
                candidate[kClassIdIndex],
                class_id
            )
        ) {
            handle_malformed_candidate(
                candidate_index,
                "class ID is non-finite, non-integral, or out of range",
                summary
            );
            continue;
        }

        const BoundingBox restored_box =
            restore_box(candidate, transform);

        // Finite network coordinates can still overflow during restoration if
        // backend output is pathologically large, so validate the result too.
        if (!has_finite_coordinates(restored_box)) {
            handle_malformed_candidate(
                candidate_index,
                "restored box coordinates are not finite",
                summary
            );
            continue;
        }

        if (
            config_.discard_degenerate_boxes
            && !restored_box.has_positive_area()
        ) {
            ++summary.rejected_as_degenerate;
            continue;
        }

        const Detection detection{
            restored_box,
            confidence,
            class_id
        };

        if (!detections.try_append(detection)) {
            // This is unreachable for a validated config unless the fixed model
            // capacity and configuration contract are changed inconsistently.
            summary.maximum_reached = true;
            break;
        }
    }

    summary.detections_written = detections.size();
    return summary;
}

/**
 * @brief Validate a non-owning output view without reading tensor elements.
 */
void Postprocessor::validate_output_view(
    DetectionTensorView output
) const {
    if (output.candidate_count > kModelMaximumDetections) {
        throw std::invalid_argument(
            "The output tensor contains more than 300 candidate rows."
        );
    }

    if (output.row_stride < kDetectionValueCount) {
        throw std::invalid_argument(
            "The output tensor row stride must contain at least 6 floats."
        );
    }

    // An empty view is valid and produces an empty DetectionBuffer. No pointer
    // is read in that case.
    if (output.candidate_count == 0U) {
        return;
    }

    if (output.data == nullptr) {
        throw std::invalid_argument(
            "The output tensor pointer must not be null for a non-empty view."
        );
    }

    const std::uintptr_t address =
        reinterpret_cast<std::uintptr_t>(output.data);

    if (address % alignof(float) != 0U) {
        throw std::invalid_argument(
            "The output tensor pointer is not aligned for float access."
        );
    }

    constexpr std::size_t maximum =
        std::numeric_limits<std::size_t>::max();

    const std::size_t preceding_row_count =
        output.candidate_count - 1U;

    if (
        preceding_row_count
        > (maximum - kDetectionValueCount) / output.row_stride
    ) {
        throw std::invalid_argument(
            "The output tensor dimensions overflow size_t."
        );
    }

    const std::size_t required_element_count =
        preceding_row_count * output.row_stride
        + kDetectionValueCount;

    if (output.element_count < required_element_count) {
        throw std::invalid_argument(
            "The output tensor view is too small. Required at least "
            + std::to_string(required_element_count)
            + " float elements, received "
            + std::to_string(output.element_count)
            + "."
        );
    }
}

/**
 * @brief Decode an exact-enough floating class index into std::int32_t.
 */
bool Postprocessor::decode_class_id(
    float class_value,
    std::int32_t& class_id
) const noexcept {
    if (!is_finite(class_value)) {
        return false;
    }

    const double class_value_double =
        static_cast<double>(class_value);

    const double rounded_class_value =
        std::round(class_value_double);

    if (
        std::fabs(
            class_value_double - rounded_class_value
        ) > static_cast<double>(config_.class_id_tolerance)
    ) {
        return false;
    }

    if (
        rounded_class_value < 0.0
        || rounded_class_value
            >= static_cast<double>(config_.class_count)
        || rounded_class_value
            > static_cast<double>(
                std::numeric_limits<std::int32_t>::max()
            )
    ) {
        return false;
    }

    class_id = static_cast<std::int32_t>(
        rounded_class_value
    );

    return true;
}

/**
 * @brief Map one XYXY box from letterboxed network space to source pixels.
 */
BoundingBox Postprocessor::restore_box(
    const float* candidate,
    const LetterboxTransform& transform
) const noexcept {
    const float inverse_scale_x =
        1.0F / transform.scale_x;

    const float inverse_scale_y =
        1.0F / transform.scale_y;

    const float horizontal_padding =
        static_cast<float>(transform.pad_left);

    const float vertical_padding =
        static_cast<float>(transform.pad_top);

    BoundingBox box{
        (
            candidate[kXMinIndex] - horizontal_padding
        ) * inverse_scale_x,
        (
            candidate[kYMinIndex] - vertical_padding
        ) * inverse_scale_y,
        (
            candidate[kXMaxIndex] - horizontal_padding
        ) * inverse_scale_x,
        (
            candidate[kYMaxIndex] - vertical_padding
        ) * inverse_scale_y
    };

    if (config_.clip_boxes) {
        const float maximum_x =
            static_cast<float>(transform.original_size.width);

        const float maximum_y =
            static_cast<float>(transform.original_size.height);

        box.x_min = std::clamp(
            box.x_min,
            0.0F,
            maximum_x
        );

        box.y_min = std::clamp(
            box.y_min,
            0.0F,
            maximum_y
        );

        box.x_max = std::clamp(
            box.x_max,
            0.0F,
            maximum_x
        );

        box.y_max = std::clamp(
            box.y_max,
            0.0F,
            maximum_y
        );
    }

    return box;
}

/**
 * @brief Count or throw for one malformed high-confidence output row.
 */
void Postprocessor::handle_malformed_candidate(
    std::size_t candidate_index,
    std::string_view reason,
    PostprocessSummary& summary
) const {
    ++summary.rejected_as_malformed;

    switch (config_.malformed_candidate_policy) {
        case MalformedCandidatePolicy::Discard:
            return;

        case MalformedCandidatePolicy::Throw:
            throw std::runtime_error(
                "Malformed detection candidate at index "
                + std::to_string(candidate_index)
                + ": "
                + std::string(reason)
                + "."
            );
    }

    // Configuration validation makes this unreachable, but retaining a hard
    // failure avoids silently ignoring a corrupted enum value.
    throw std::logic_error(
        "Unsupported malformed-candidate policy."
    );
}

}  // namespace edge