#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace edge {

struct LetterboxTransform;

/**
 * @brief Number of scalar values in one end-to-end YOLO26 detection row.
 *
 * Every candidate row uses the following contiguous layout:
 *
 *     [x_min, y_min, x_max, y_max, confidence, class_id]
 */
inline constexpr std::size_t kDetectionValueCount = 6U;

/**
 * @brief Maximum number of detections emitted by the exported YOLO26M head.
 *
 * The project's fixed-shape ONNX and TensorRT artifacts expose output shape
 * [1, 300, 6], so no valid inference result can contain more than 300 rows.
 */
inline constexpr std::size_t kModelMaximumDetections = 300U;

/**
 * @brief Number of classes in the project's BDD100K detection label space.
 */
inline constexpr std::size_t kBdd100kClassCount = 10U;

/**
 * @brief Compile-time BDD100K class names in the exact training-ID order.
 */
inline constexpr std::array<std::string_view, kBdd100kClassCount>
    kBdd100kClassNames{
        "person",
        "rider",
        "car",
        "bus",
        "truck",
        "bike",
        "motor",
        "traffic light",
        "traffic sign",
        "train"
    };

/**
 * @brief Return a BDD100K class name or "unknown" for an invalid class ID.
 */
[[nodiscard]] constexpr std::string_view bdd100k_class_name(
    std::int32_t class_id
) noexcept {
    return class_id >= 0
        && static_cast<std::size_t>(class_id) < kBdd100kClassNames.size()
            ? kBdd100kClassNames[static_cast<std::size_t>(class_id)]
            : std::string_view{"unknown"};
}

/**
 * @brief Axis-aligned bounding box in original-image pixel coordinates.
 *
 * Coordinates follow the XYXY convention:
 *
 *     x_min, y_min = top-left corner
 *     x_max, y_max = bottom-right corner
 *
 * The postprocessor clips coordinates to [0, image_width] and
 * [0, image_height] when clipping is enabled, matching the coordinate
 * convention used by the Ultralytics reference pipeline.
 */
struct BoundingBox {
    float x_min = 0.0F;
    float y_min = 0.0F;
    float x_max = 0.0F;
    float y_max = 0.0F;

    /**
     * @brief Return the signed box width.
     */
    [[nodiscard]] constexpr float width() const noexcept {
        return x_max - x_min;
    }

    /**
     * @brief Return the signed box height.
     */
    [[nodiscard]] constexpr float height() const noexcept {
        return y_max - y_min;
    }

    /**
     * @brief Return true when both dimensions are strictly positive.
     */
    [[nodiscard]] constexpr bool has_positive_area() const noexcept {
        return width() > 0.0F && height() > 0.0F;
    }

    /**
     * @brief Return the box area, or zero for a degenerate box.
     */
    [[nodiscard]] constexpr float area() const noexcept {
        return has_positive_area()
            ? width() * height()
            : 0.0F;
    }
};

/**
 * @brief One finalized object detection in original-image coordinates.
 */
struct Detection {
    BoundingBox box{};
    float confidence = 0.0F;
    std::int32_t class_id = -1;
};

/**
 * @brief Non-owning view over a batch-one end-to-end detection tensor.
 *
 * The project artifacts return contiguous FP32 output even when the model
 * input or internal execution uses FP16 or INT8. For the standard output
 * shape [1, 300, 6]:
 *
 *     data             -> first float in output0
 *     element_count    -> 1800
 *     candidate_count  -> 300
 *     row_stride       -> 6
 *
 * row_stride is expressed in float elements, not bytes. Supporting a stride
 * allows the postprocessor to consume either tightly packed output or a
 * backend-owned row-padded view without copying.
 */
struct DetectionTensorView {
    const float* data = nullptr;
    std::size_t element_count = 0U;
    std::size_t candidate_count = 0U;
    std::size_t row_stride = kDetectionValueCount;

    /**
     * @brief Construct a tightly packed [candidate_count, 6] tensor view.
     *
     * An overflowed candidate count intentionally produces element_count=0,
     * which is rejected by Postprocessor before any memory is read.
     */
    [[nodiscard]] static constexpr DetectionTensorView contiguous(
        const float* data,
        std::size_t candidate_count
    ) noexcept {
        constexpr std::size_t maximum =
            std::numeric_limits<std::size_t>::max();

        const bool multiplication_is_safe =
            candidate_count <= maximum / kDetectionValueCount;

        return DetectionTensorView{
            data,
            multiplication_is_safe
                ? candidate_count * kDetectionValueCount
                : 0U,
            candidate_count,
            kDetectionValueCount
        };
    }

    /**
     * @brief Return a pointer to one candidate row.
     *
     * The caller must validate the view and ensure candidate_index is in
     * range before calling this helper.
     */
    [[nodiscard]] constexpr const float* candidate(
        std::size_t candidate_index
    ) const noexcept {
        return data + candidate_index * row_stride;
    }
};

/**
 * @brief Policy for a high-confidence row containing malformed values.
 *
 * Low-confidence rows are normal padding and are always ignored. This policy
 * applies only after a row passes the configured confidence threshold but
 * contains non-finite coordinates, an invalid class ID, or another malformed
 * value.
 */
enum class MalformedCandidatePolicy : std::uint8_t {
    Discard,
    Throw
};

/**
 * @brief Configuration for decoding the project's end-to-end YOLO26 output.
 *
 * YOLO26M's one-to-one head is NMS-free and already emits final candidate
 * rows. This configuration therefore intentionally contains no IoU threshold
 * and the postprocessor does not run NMS a second time.
 */
struct PostprocessConfig {
    float confidence_threshold = 0.25F;

    std::size_t maximum_detections = kModelMaximumDetections;
    std::size_t class_count = kBdd100kClassCount;

    bool clip_boxes = true;
    bool discard_degenerate_boxes = true;

    float class_id_tolerance = 1.0e-3F;

    MalformedCandidatePolicy malformed_candidate_policy =
        MalformedCandidatePolicy::Discard;

    /**
     * @brief Validate every postprocessing invariant.
     *
     * @throws std::invalid_argument if thresholds, limits, class count, enum
     *         values, or another configuration field is invalid.
     */
    void validate() const;
};

/**
 * @brief Per-call diagnostics produced while decoding one output tensor.
 *
 * These counters allow correctness tests and benchmark reports to distinguish
 * ordinary confidence filtering from malformed or degenerate output rows.
 */
struct PostprocessSummary {
    std::size_t candidates_examined = 0U;
    std::size_t rejected_by_confidence = 0U;
    std::size_t rejected_as_malformed = 0U;
    std::size_t rejected_as_degenerate = 0U;
    std::size_t detections_written = 0U;

    bool maximum_reached = false;
};

/**
 * @brief Fixed-capacity, allocation-free storage for finalized detections.
 *
 * Capacity exactly matches the model's maximum output count. Reusing one
 * buffer per inference worker prevents per-frame heap allocation while still
 * providing familiar container-style access.
 */
class DetectionBuffer final {
public:
    using value_type = Detection;
    using size_type = std::size_t;
    using iterator = Detection*;
    using const_iterator = const Detection*;

    DetectionBuffer() = default;
    ~DetectionBuffer() = default;

    DetectionBuffer(const DetectionBuffer&) = default;
    DetectionBuffer& operator=(const DetectionBuffer&) = default;

    DetectionBuffer(DetectionBuffer&&) noexcept = default;
    DetectionBuffer& operator=(DetectionBuffer&&) noexcept = default;

    /**
     * @brief Remove all logical elements without releasing storage.
     */
    constexpr void clear() noexcept {
        size_ = 0U;
    }

    /**
     * @brief Return the number of valid detections currently stored.
     */
    [[nodiscard]] constexpr size_type size() const noexcept {
        return size_;
    }

    /**
     * @brief Return the compile-time storage capacity.
     */
    [[nodiscard]] static constexpr size_type capacity() noexcept {
        return kModelMaximumDetections;
    }

    /**
     * @brief Return true when the buffer contains no detections.
     */
    [[nodiscard]] constexpr bool empty() const noexcept {
        return size_ == 0U;
    }

    /**
     * @brief Return a pointer to the first storage element.
     */
    [[nodiscard]] constexpr Detection* data() noexcept {
        return storage_.data();
    }

    /**
     * @brief Return a pointer to the first storage element.
     */
    [[nodiscard]] constexpr const Detection* data() const noexcept {
        return storage_.data();
    }

    /**
     * @brief Return an iterator to the first valid detection.
     */
    [[nodiscard]] constexpr iterator begin() noexcept {
        return storage_.data();
    }

    /**
     * @brief Return a constant iterator to the first valid detection.
     */
    [[nodiscard]] constexpr const_iterator begin() const noexcept {
        return storage_.data();
    }

    /**
     * @brief Return a constant iterator to the first valid detection.
     */
    [[nodiscard]] constexpr const_iterator cbegin() const noexcept {
        return storage_.data();
    }

    /**
     * @brief Return an iterator one past the final valid detection.
     */
    [[nodiscard]] constexpr iterator end() noexcept {
        return storage_.data() + size_;
    }

    /**
     * @brief Return a constant iterator one past the final valid detection.
     */
    [[nodiscard]] constexpr const_iterator end() const noexcept {
        return storage_.data() + size_;
    }

    /**
     * @brief Return a constant iterator one past the final valid detection.
     */
    [[nodiscard]] constexpr const_iterator cend() const noexcept {
        return storage_.data() + size_;
    }

    /**
     * @brief Access one valid detection without bounds checking.
     */
    [[nodiscard]] constexpr Detection& operator[](
        size_type index
    ) noexcept {
        return storage_[index];
    }

    /**
     * @brief Access one valid detection without bounds checking.
     */
    [[nodiscard]] constexpr const Detection& operator[](
        size_type index
    ) const noexcept {
        return storage_[index];
    }

private:
    friend class Postprocessor;

    /**
     * @brief Append one detection when capacity is available.
     *
     * Only Postprocessor may mutate the logical contents directly.
     */
    [[nodiscard]] constexpr bool try_append(
        const Detection& detection
    ) noexcept {
        if (size_ >= storage_.size()) {
            return false;
        }

        storage_[size_] = detection;
        ++size_;
        return true;
    }

    std::array<Detection, kModelMaximumDetections> storage_{};
    size_type size_ = 0U;
};

/**
 * @brief Decode NMS-free YOLO26 output into original-image detections.
 *
 * The class is stateless during process(), so one configured instance may be
 * used concurrently by multiple workers as long as set_config() is not called
 * concurrently. Each worker must provide its own DetectionBuffer.
 *
 * Processing preserves the model's row order. The exported end-to-end head
 * already ranks and limits detections, so the implementation performs no
 * additional sorting and no NMS.
 */
class Postprocessor final {
public:
    explicit Postprocessor(PostprocessConfig config = {});

    ~Postprocessor() = default;

    Postprocessor(const Postprocessor&) = default;
    Postprocessor& operator=(const Postprocessor&) = default;

    Postprocessor(Postprocessor&&) noexcept = default;
    Postprocessor& operator=(Postprocessor&&) noexcept = default;

    /**
     * @brief Return the active postprocessing configuration.
     */
    [[nodiscard]] const PostprocessConfig& config() const noexcept;

    /**
     * @brief Replace the postprocessing configuration after validation.
     *
     * Do not call this concurrently with process().
     *
     * @throws std::invalid_argument if the new configuration is invalid.
     */
    void set_config(PostprocessConfig config);

    /**
     * @brief Decode one batch-one [1, N, 6] output tensor.
     *
     * Each accepted row is:
     *
     *     [x_min, y_min, x_max, y_max, confidence, class_id]
     *
     * Coordinates are converted from the 640x640 letterboxed network space
     * back to the original image using transform, optionally clipped, and
     * written into caller-owned fixed-capacity output storage.
     *
     * The output buffer is cleared at the start of every call.
     *
     * @throws std::invalid_argument if output, transform, or their dimensions
     *         are structurally invalid.
     * @throws std::runtime_error when malformed_candidate_policy is Throw and
     *         a malformed high-confidence candidate is encountered.
     */
    [[nodiscard]] PostprocessSummary process(
        DetectionTensorView output,
        const LetterboxTransform& transform,
        DetectionBuffer& detections
    ) const;

private:
    /**
     * @brief Validate tensor pointer, shape-equivalent counts, stride, capacity,
     *        and the fixed model-output limits before reading memory.
     */
    void validate_output_view(
        DetectionTensorView output
    ) const;

    /**
     * @brief Convert a floating class value to a checked integer class ID.
     *
     * @return True only when the value is finite, sufficiently close to an
     *         integer, and inside [0, class_count).
     */
    [[nodiscard]] bool decode_class_id(
        float class_value,
        std::int32_t& class_id
    ) const noexcept;

    /**
     * @brief Undo letterbox padding/scaling and optionally clip the XYXY box to
     *        original-image bounds.
     */
    [[nodiscard]] BoundingBox restore_box(
        const float* candidate,
        const LetterboxTransform& transform
    ) const noexcept;

    /**
     * @brief Apply the configured malformed-row policy and update diagnostics.
     */
    void handle_malformed_candidate(
        std::size_t candidate_index,
        std::string_view reason,
        PostprocessSummary& summary
    ) const;

    PostprocessConfig config_;
};

}  // namespace edge