#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>

#include <opencv2/core/mat.hpp>

namespace edge {

/**
 * @brief Scalar storage type exposed by a model input binding.
 *
 * Float16 values are stored as raw IEEE-754 binary16 bit patterns inside
 * std::uint16_t elements. This keeps preprocessing independent of ONNX
 * Runtime, CUDA, and TensorRT-specific half-precision wrapper types.
 *
 * The project's quantized ONNX and TensorRT artifacts still expose floating
 * input bindings, so an Int8 input type is intentionally not included.
 */
enum class TensorElementType : std::uint8_t {
    Float32,
    Float16
};

/**
 * @brief Return the byte width of one tensor element.
 *
 * @return Zero only when an invalid enum value has been produced through an
 *         explicit cast.
 */
[[nodiscard]] constexpr std::size_t tensor_element_size(
    TensorElementType type
) noexcept {
    switch (type) {
        case TensorElementType::Float32:
            return sizeof(float);

        case TensorElementType::Float16:
            return sizeof(std::uint16_t);
    }

    return 0U;
}

/**
 * @brief Mutable view over caller-owned, preallocated tensor storage.
 *
 * ImageProcessor writes directly into this memory. The caller retains
 * ownership and must keep the storage alive until the inference backend has
 * finished consuming it.
 *
 * The pointer may refer to ordinary heap memory, aligned memory, pinned host
 * memory, or another CPU-writable allocation supplied by the inference layer.
 */
struct MutableTensorView {
    void* data = nullptr;
    std::size_t byte_capacity = 0U;
    TensorElementType element_type = TensorElementType::Float32;

    /**
     * @brief Construct a view over float32 tensor storage.
     */
    [[nodiscard]] static constexpr MutableTensorView float32(
        float* data,
        std::size_t element_capacity
    ) noexcept {
        return MutableTensorView{
            data,
            element_capacity * sizeof(float),
            TensorElementType::Float32
        };
    }

    /**
     * @brief Construct a view over raw IEEE-754 binary16 storage.
     */
    [[nodiscard]] static constexpr MutableTensorView float16(
        std::uint16_t* data,
        std::size_t element_capacity
    ) noexcept {
        return MutableTensorView{
            data,
            element_capacity * sizeof(std::uint16_t),
            TensorElementType::Float16
        };
    }

    /**
     * @brief Return the number of complete scalar elements available.
     */
    [[nodiscard]] constexpr std::size_t element_capacity() const noexcept {
        const std::size_t element_size =
            tensor_element_size(element_type);

        return element_size == 0U
            ? 0U
            : byte_capacity / element_size;
    }
};

/**
 * @brief Integer image dimensions in width-height order.
 */
struct ImageSize {
    int width = 0;
    int height = 0;

    /**
     * @brief Return true when both dimensions are strictly positive.
     */
    [[nodiscard]] constexpr bool valid() const noexcept {
        return width > 0 && height > 0;
    }
};

/**
 * @brief Channel order written into the model input tensor.
 */
enum class ChannelOrder : std::uint8_t {
    RGB,
    BGR
};

/**
 * @brief OpenCV interpolation policy used during aspect-preserving resize.
 *
 * Linear is the project default because it matches the pinned Ultralytics
 * letterbox path used to export and validate the YOLO26M artifacts.
 *
 * Automatic remains available for experiments and selects area interpolation
 * while shrinking and linear interpolation while enlarging.
 */
enum class ResizeInterpolation : std::uint8_t {
    Linear,
    Automatic,
    Nearest,
    Area,
    Cubic
};

/**
 * @brief CPU execution policy for the HWC-to-NCHW conversion stage.
 *
 * Automatic uses OpenCV's existing worker pool only when the fixed tensor is
 * large enough to amortize scheduling overhead. Serial remains available for
 * low-core-count systems and deterministic microbenchmark comparisons.
 */
enum class PreprocessExecutionPolicy : std::uint8_t {
    Automatic,
    Serial,
    OpenCvParallel
};

/**
 * @brief Complete preprocessing configuration for a fixed-shape detector.
 *
 * Normalization is applied independently to each final output channel:
 *
 *     output = (pixel * pixel_scale - mean[channel]) / stddev[channel]
 *
 * mean and stddev are indexed in output-channel order. With RGB output,
 * index 0 is R, index 1 is G, and index 2 is B.
 *
 * The defaults match this project's YOLO26M artifacts:
 *
 *     input shape:    [1, 3, 640, 640]
 *     source format:  OpenCV BGR uint8
 *     output order:   RGB
 *     pixel range:    [0.0, 1.0]
 *     interpolation:  linear
 *     padding value:  114
 *     centered:       yes
 *     scale-up:       allowed
 *     conversion:     automatically parallelized when worthwhile
 */
struct PreprocessConfig {
    ImageSize network_size{640, 640};

    bool allow_scale_up = true;
    bool center_letterbox = true;

    std::array<std::uint8_t, 3> padding_bgr{114U, 114U, 114U};

    ChannelOrder output_channel_order = ChannelOrder::RGB;
    ResizeInterpolation interpolation = ResizeInterpolation::Linear;

    PreprocessExecutionPolicy execution_policy =
        PreprocessExecutionPolicy::Automatic;

    std::size_t parallel_minimum_pixels = 256U * 256U;

    float pixel_scale = 1.0F / 255.0F;
    std::array<float, 3> mean{0.0F, 0.0F, 0.0F};
    std::array<float, 3> stddev{1.0F, 1.0F, 1.0F};

    /**
     * @brief Validate every configuration invariant.
     *
     * @throws std::invalid_argument if dimensions, normalization values,
     *         enum values, or another configuration field is invalid.
     * @throws std::overflow_error if the resulting tensor size cannot be
     *         represented by std::size_t.
     */
    void validate() const;
};

/**
 * @brief Geometry required to map coordinates between original and network
 *        image spaces.
 *
 * scale_x and scale_y are derived from the final integer resize dimensions.
 * They can differ very slightly because width and height are rounded
 * independently before OpenCV performs the resize.
 */
struct LetterboxTransform {
    ImageSize original_size{};
    ImageSize resized_size{};
    ImageSize network_size{};

    float scale_x = 0.0F;
    float scale_y = 0.0F;

    int pad_left = 0;
    int pad_top = 0;
    int pad_right = 0;
    int pad_bottom = 0;

    /**
     * @brief Return true when all dimensions, scales, and padding values are
     *        internally plausible.
     */
    [[nodiscard]] constexpr bool valid() const noexcept {
        return original_size.valid()
            && resized_size.valid()
            && network_size.valid()
            && resized_size.width <= network_size.width
            && resized_size.height <= network_size.height
            && scale_x > 0.0F
            && scale_y > 0.0F
            && pad_left >= 0
            && pad_top >= 0
            && pad_right >= 0
            && pad_bottom >= 0
            && resized_size.width + pad_left + pad_right
                == network_size.width
            && resized_size.height + pad_top + pad_bottom
                == network_size.height;
    }

    /**
     * @brief Convert an x-coordinate from network space to original space.
     *
     * The returned coordinate is not clipped to the original image bounds.
     */
    [[nodiscard]] constexpr float network_to_original_x(
        float network_x
    ) const noexcept {
        return (
            network_x - static_cast<float>(pad_left)
        ) / scale_x;
    }

    /**
     * @brief Convert a y-coordinate from network space to original space.
     *
     * The returned coordinate is not clipped to the original image bounds.
     */
    [[nodiscard]] constexpr float network_to_original_y(
        float network_y
    ) const noexcept {
        return (
            network_y - static_cast<float>(pad_top)
        ) / scale_y;
    }

    /**
     * @brief Convert an x-coordinate from original space to network space.
     */
    [[nodiscard]] constexpr float original_to_network_x(
        float original_x
    ) const noexcept {
        return original_x * scale_x
            + static_cast<float>(pad_left);
    }

    /**
     * @brief Convert a y-coordinate from original space to network space.
     */
    [[nodiscard]] constexpr float original_to_network_y(
        float original_y
    ) const noexcept {
        return original_y * scale_y
            + static_cast<float>(pad_top);
    }
};

/**
 * @brief Metadata describing one completed preprocessing operation.
 */
struct PreprocessResult {
    std::array<std::int64_t, 4> tensor_shape{1, 3, 0, 0};

    TensorElementType element_type = TensorElementType::Float32;
    std::size_t element_count = 0U;
    std::size_t byte_count = 0U;

    LetterboxTransform transform{};
};

/**
 * @brief Reusable image preprocessor for fixed-shape object detection models.
 *
 * The processor accepts OpenCV BGR uint8 images and writes a contiguous NCHW
 * tensor directly into caller-owned memory. It preserves aspect ratio through
 * Ultralytics-compatible letterboxing and returns the exact geometry required
 * to restore model-space detections to original-image coordinates.
 *
 * The implementation owns one reusable, final-size OpenCV workspace. Padding
 * is filled directly into that buffer and resizing is performed into an ROI,
 * avoiding a separate resized-image allocation and copyMakeBorder pass.
 *
 * Normalized FP32 and FP16 lookup tables are cached and rebuilt only when the
 * configuration changes.
 *
 * One ImageProcessor instance is not safe for concurrent preprocess() calls
 * because its workspace is mutable. Use one instance per worker thread.
 */
class ImageProcessor final {
public:
    explicit ImageProcessor(PreprocessConfig config = {});

    ~ImageProcessor() = default;

    ImageProcessor(const ImageProcessor&) = delete;
    ImageProcessor& operator=(const ImageProcessor&) = delete;

    ImageProcessor(ImageProcessor&&) noexcept = default;
    ImageProcessor& operator=(ImageProcessor&&) noexcept = default;

    /**
     * @brief Return the active preprocessing configuration.
     */
    [[nodiscard]] const PreprocessConfig& config() const noexcept;

    /**
     * @brief Replace the preprocessing configuration.
     *
     * This validates the new configuration, rebuilds normalization lookup
     * tables, and releases incompatible workspace storage.
     *
     * @throws std::invalid_argument if the new configuration is invalid.
     * @throws std::overflow_error if the configured tensor size overflows.
     */
    void set_config(PreprocessConfig config);

    /**
     * @brief Return the fixed model input shape in NCHW order.
     */
    [[nodiscard]] std::array<std::int64_t, 4> tensor_shape() const noexcept;

    /**
     * @brief Return the number of scalar elements in one input tensor.
     */
    [[nodiscard]] std::size_t required_element_count() const noexcept;

    /**
     * @brief Return the number of bytes required for one input tensor.
     */
    [[nodiscard]] std::size_t required_byte_count(
        TensorElementType type
    ) const noexcept;

    /**
     * @brief Load an image from disk as a three-channel BGR uint8 cv::Mat.
     *
     * The implementation reads encoded bytes through std::filesystem and then
     * calls cv::imdecode, providing reliable support for Unicode Windows paths.
     *
     * @throws std::invalid_argument if the path is empty.
     * @throws std::runtime_error if the file cannot be read or decoded.
     */
    [[nodiscard]] static cv::Mat load_bgr_image(
        const std::filesystem::path& image_path
    );

    /**
     * @brief Compute the exact Ultralytics-compatible letterbox geometry for
     *        an image size without allocating or touching pixel storage.
     *
     * This shared helper is used by both the CPU ImageProcessor and the
     * optional fused CUDA/TensorRT pipeline so both paths restore detections
     * with identical integer resize dimensions and padding.
     *
     * @throws std::invalid_argument if source_size or config is invalid.
     * @throws std::runtime_error if rounding produces impossible geometry.
     */
    [[nodiscard]] static LetterboxTransform calculate_letterbox_transform(
        ImageSize source_size,
        const PreprocessConfig& config
    );

    /**
     * @brief Hot-path variant for a PreprocessConfig that was already validated.
     *
     * ImageProcessor and NativeTensorRtPipeline validate configuration during
     * construction or set_config(), then use this overload per frame to avoid
     * repeating invariant and overflow checks in the latency-critical path.
     * Calling it with an unvalidated configuration is a programming error.
     */
    [[nodiscard]] static LetterboxTransform
    calculate_letterbox_transform_unchecked(
        ImageSize source_size,
        const PreprocessConfig& validated_config
    );

    /**
     * @brief Letterbox, normalize, reorder, and write one image to NCHW.
     *
     * Source requirements:
     *   - non-empty
     *   - two-dimensional
     *   - CV_8UC3
     *   - OpenCV BGR channel order
     *
     * Destination layout:
     *   - shape [1, 3, network_height, network_width]
     *   - contiguous planar NCHW storage
     *   - FP32 or raw IEEE-754 binary16 according to destination.element_type
     *
     * The destination must provide at least required_byte_count() writable
     * bytes and satisfy the natural alignment of the selected scalar type.
     *
     * @throws std::invalid_argument if the source image or destination view is
     *         invalid.
     * @throws std::runtime_error if OpenCV resize or workspace preparation
     *         fails.
     */
    [[nodiscard]] PreprocessResult preprocess(
        const cv::Mat& bgr_image,
        MutableTensorView destination
    );

    /**
     * @brief Release reusable OpenCV workspace memory.
     *
     * Normalization lookup tables remain cached. This is mainly useful when a
     * long-running process will keep the processor idle for an extended time.
     */
    void release_workspace() noexcept;

private:
    /**
     * @brief Calculate parity-sensitive resize dimensions, scale factors, and
     *        the exact left/top/right/bottom letterbox padding.
     */
    [[nodiscard]] LetterboxTransform compute_letterbox_transform(
        const cv::Mat& bgr_image
    ) const;

    /**
     * @brief Translate the configured interpolation policy to an OpenCV flag.
     */
    [[nodiscard]] int resolve_interpolation(
        const LetterboxTransform& transform
    ) const noexcept;

    /**
     * @brief Fill the reusable final-size workspace with padding and resize the
     *        source image directly into its unpadded ROI.
     */
    void prepare_letterboxed_bgr(
        const cv::Mat& bgr_image,
        const LetterboxTransform& transform
    );

    /**
     * @brief Write the prepared BGR workspace directly into planar FP32 NCHW
     *        destination memory using cached normalization values and the
     *        configured serial/parallel execution policy.
     */
    void write_nchw_float32(
        float* destination
    ) const;

    /**
     * @brief Write the prepared BGR workspace directly into planar FP16 NCHW
     *        destination memory using cached binary16 values and the configured
     *        serial/parallel execution policy.
     */
    void write_nchw_float16(
        std::uint16_t* destination
    ) const;

    /**
     * @brief Rebuild channel mapping and all FP32/FP16 normalization lookup
     *        tables after construction or a configuration change.
     */
    void rebuild_normalization_tables() noexcept;

    /**
     * @brief Convert one float32 value to raw IEEE-754 binary16 bits using
     *        round-to-nearest, ties-to-even.
     */
    [[nodiscard]] static std::uint16_t float32_to_float16_bits(
        float value
    ) noexcept;

    /**
     * @brief Validate source image dimensionality, scalar type, channels, and
     *        readable storage.
     */
    void validate_source_image(
        const cv::Mat& bgr_image
    ) const;

    /**
     * @brief Validate destination type, pointer, capacity, and alignment.
     */
    void validate_destination(
        MutableTensorView destination
    ) const;

    PreprocessConfig config_;

    // Maps output planes to source OpenCV BGR channel indices.
    std::array<std::size_t, 3> source_channel_for_output_{2U, 1U, 0U};

    // Cached normalization results for every possible uint8 channel value.
    std::array<std::array<float, 256>, 3> float32_lookup_{};
    std::array<std::array<std::uint16_t, 256>, 3> float16_lookup_{};

    // Reusable final-size BGR workspace containing both padding and image ROI.
    cv::Mat letterboxed_bgr_;
};

}  // namespace edge
