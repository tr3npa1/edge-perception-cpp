#include "image_processor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace edge {
namespace {

constexpr std::size_t kChannelCount = 3U;
constexpr std::size_t kUint8ValueCount = 256U;

/**
 * @brief Convert a filesystem path to UTF-8 for exception messages.
 *
 * std::ifstream receives the original std::filesystem::path, so this conversion
 * is used only for diagnostics and does not affect Unicode path handling.
 */
[[nodiscard]] std::string path_to_utf8(
    const std::filesystem::path& path
) {
#if defined(__cpp_lib_char8_t)
    const std::u8string utf8 = path.u8string();

    return std::string(
        reinterpret_cast<const char*>(utf8.data()),
        utf8.size()
    );
#else
    return path.u8string();
#endif
}

/**
 * @brief Return true when a float is neither infinity nor NaN.
 */
[[nodiscard]] bool is_finite(
    float value
) noexcept {
    return std::isfinite(value) != 0;
}

/**
 * @brief Validate tensor-size arithmetic and return the NCHW element count.
 *
 * The byte-overflow check uses float because Float32 is the largest currently
 * supported input element type.
 */
[[nodiscard]] std::size_t checked_tensor_element_count(
    const ImageSize& network_size
) {
    if (!network_size.valid()) {
        throw std::invalid_argument(
            "Network dimensions must be strictly positive."
        );
    }

    const auto width =
        static_cast<std::size_t>(network_size.width);

    const auto height =
        static_cast<std::size_t>(network_size.height);

    constexpr std::size_t maximum =
        std::numeric_limits<std::size_t>::max();

    if (width > maximum / height) {
        throw std::overflow_error(
            "The configured network pixel count overflows size_t."
        );
    }

    const std::size_t pixel_count = width * height;

    if (pixel_count > maximum / kChannelCount) {
        throw std::overflow_error(
            "The configured input tensor element count overflows size_t."
        );
    }

    const std::size_t element_count =
        pixel_count * kChannelCount;

    if (element_count > maximum / sizeof(float)) {
        throw std::overflow_error(
            "The configured float32 input tensor byte count overflows size_t."
        );
    }

    return element_count;
}

/**
 * @brief Check whether a channel-order enum value is supported.
 */
[[nodiscard]] bool is_supported_channel_order(
    ChannelOrder order
) noexcept {
    switch (order) {
        case ChannelOrder::RGB:
        case ChannelOrder::BGR:
            return true;
    }

    return false;
}

/**
 * @brief Check whether a resize-interpolation enum value is supported.
 */
[[nodiscard]] bool is_supported_interpolation(
    ResizeInterpolation interpolation
) noexcept {
    switch (interpolation) {
        case ResizeInterpolation::Linear:
        case ResizeInterpolation::Automatic:
        case ResizeInterpolation::Nearest:
        case ResizeInterpolation::Area:
        case ResizeInterpolation::Cubic:
            return true;
    }

    return false;
}

/**
 * @brief Check whether a destination tensor element type is supported.
 */
[[nodiscard]] bool is_supported_tensor_type(
    TensorElementType type
) noexcept {
    switch (type) {
        case TensorElementType::Float32:
        case TensorElementType::Float16:
            return true;
    }

    return false;
}

/**
 * @brief Reproduce Python's round-to-nearest, ties-to-even for an int result.
 *
 * Ultralytics computes resized dimensions with Python round(). std::round and
 * std::lround use a different halfway rule, so using either could cause a
 * one-pixel mismatch for particular source dimensions.
 */
[[nodiscard]] int python_round_to_int(
    double value
) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(
            "Cannot round a non-finite letterbox value."
        );
    }

    const double lower = std::floor(value);
    const double fraction = value - lower;

    double rounded = lower;

    if (fraction > 0.5) {
        rounded = lower + 1.0;
    } else if (
        fraction == 0.5
        && std::fmod(lower, 2.0) != 0.0
    ) {
        rounded = lower + 1.0;
    }

    const double minimum =
        static_cast<double>(std::numeric_limits<int>::min());

    const double maximum =
        static_cast<double>(std::numeric_limits<int>::max());

    if (rounded < minimum || rounded > maximum) {
        throw std::overflow_error(
            "Rounded letterbox value cannot be represented by int."
        );
    }

    return static_cast<int>(rounded);
}

/**
 * @brief Write a BGR uint8 image into planar NCHW destination storage.
 *
 * The lookup tables already contain channel reordering, normalization, and—
 * for the uint16_t specialization—FP16 conversion results. The hot loop only
 * reads three bytes and performs three table lookups and three stores.
 */
template <typename Scalar>
void write_nchw_from_lookup(
    const cv::Mat& letterboxed_bgr,
    const std::array<std::size_t, kChannelCount>&
        source_channel_for_output,
    const std::array<
        std::array<Scalar, kUint8ValueCount>,
        kChannelCount
    >& lookup,
    Scalar* destination
) {
    const int width = letterboxed_bgr.cols;
    const int height = letterboxed_bgr.rows;

    const std::size_t plane_size =
        static_cast<std::size_t>(width)
        * static_cast<std::size_t>(height);

    Scalar* const output_plane_0 = destination;
    Scalar* const output_plane_1 = destination + plane_size;
    Scalar* const output_plane_2 = destination + (2U * plane_size);

    const int source_channel_0 = static_cast<int>(
        source_channel_for_output[0]
    );

    const int source_channel_1 = static_cast<int>(
        source_channel_for_output[1]
    );

    const int source_channel_2 = static_cast<int>(
        source_channel_for_output[2]
    );

    for (int row_index = 0; row_index < height; ++row_index) {
        const cv::Vec3b* const source_row =
            letterboxed_bgr.ptr<cv::Vec3b>(row_index);

        const std::size_t row_offset =
            static_cast<std::size_t>(row_index)
            * static_cast<std::size_t>(width);

        for (int column_index = 0; column_index < width; ++column_index) {
            const cv::Vec3b& pixel = source_row[column_index];

            const std::size_t destination_index =
                row_offset + static_cast<std::size_t>(column_index);

            const std::size_t value_0 = static_cast<std::size_t>(
                pixel[source_channel_0]
            );

            const std::size_t value_1 = static_cast<std::size_t>(
                pixel[source_channel_1]
            );

            const std::size_t value_2 = static_cast<std::size_t>(
                pixel[source_channel_2]
            );

            output_plane_0[destination_index] = lookup[0][value_0];
            output_plane_1[destination_index] = lookup[1][value_1];
            output_plane_2[destination_index] = lookup[2][value_2];
        }
    }
}

}  // namespace

/**
 * @brief Validate dimensions, enum values, normalization, and tensor size.
 */
void PreprocessConfig::validate() const {
    if (!network_size.valid()) {
        throw std::invalid_argument(
            "PreprocessConfig.network_size must contain positive dimensions."
        );
    }

    if (!is_supported_channel_order(output_channel_order)) {
        throw std::invalid_argument(
            "PreprocessConfig.output_channel_order is invalid."
        );
    }

    if (!is_supported_interpolation(interpolation)) {
        throw std::invalid_argument(
            "PreprocessConfig.interpolation is invalid."
        );
    }

    if (!is_finite(pixel_scale) || pixel_scale <= 0.0F) {
        throw std::invalid_argument(
            "PreprocessConfig.pixel_scale must be finite and positive."
        );
    }

    for (std::size_t channel = 0U; channel < mean.size(); ++channel) {
        if (!is_finite(mean[channel])) {
            throw std::invalid_argument(
                "Every PreprocessConfig.mean value must be finite."
            );
        }

        if (!is_finite(stddev[channel]) || stddev[channel] <= 0.0F) {
            throw std::invalid_argument(
                "Every PreprocessConfig.stddev value must be finite and "
                "strictly positive."
            );
        }

        const float normalized_minimum =
            (0.0F - mean[channel]) / stddev[channel];

        const float normalized_maximum =
            (
                255.0F * pixel_scale
                - mean[channel]
            ) / stddev[channel];

        if (
            !is_finite(normalized_minimum)
            || !is_finite(normalized_maximum)
        ) {
            throw std::invalid_argument(
                "The configured normalization produces a non-finite value."
            );
        }
    }

    static_cast<void>(
        checked_tensor_element_count(network_size)
    );
}

/**
 * @brief Construct a validated processor and build its normalization caches.
 */
ImageProcessor::ImageProcessor(
    PreprocessConfig config
)
    : config_(std::move(config)) {
    config_.validate();
    rebuild_normalization_tables();
}

/**
 * @brief Return the active preprocessing configuration.
 */
const PreprocessConfig& ImageProcessor::config() const noexcept {
    return config_;
}

/**
 * @brief Install a validated configuration and refresh dependent state.
 */
void ImageProcessor::set_config(
    PreprocessConfig config
) {
    config.validate();

    const bool network_size_changed =
        config.network_size.width != config_.network_size.width
        || config.network_size.height != config_.network_size.height;

    config_ = std::move(config);
    rebuild_normalization_tables();

    if (network_size_changed) {
        release_workspace();
    }
}

/**
 * @brief Return the fixed batch-one, three-channel NCHW input shape.
 */
std::array<std::int64_t, 4> ImageProcessor::tensor_shape() const noexcept {
    return {
        1,
        3,
        static_cast<std::int64_t>(config_.network_size.height),
        static_cast<std::int64_t>(config_.network_size.width)
    };
}

/**
 * @brief Return the validated scalar element count for one input tensor.
 */
std::size_t ImageProcessor::required_element_count() const noexcept {
    const auto width =
        static_cast<std::size_t>(config_.network_size.width);

    const auto height =
        static_cast<std::size_t>(config_.network_size.height);

    return kChannelCount * width * height;
}

/**
 * @brief Return the byte count for one tensor of the selected scalar type.
 */
std::size_t ImageProcessor::required_byte_count(
    TensorElementType type
) const noexcept {
    return required_element_count() * tensor_element_size(type);
}

/**
 * @brief Read encoded bytes through std::filesystem and decode a BGR image.
 */
cv::Mat ImageProcessor::load_bgr_image(
    const std::filesystem::path& image_path
) {
    if (image_path.empty()) {
        throw std::invalid_argument(
            "The image path must not be empty."
        );
    }

    std::ifstream input(
        image_path,
        std::ios::binary | std::ios::ate
    );

    if (!input.is_open()) {
        throw std::runtime_error(
            "Failed to open image file: " + path_to_utf8(image_path)
        );
    }

    const std::streampos end_position = input.tellg();

    if (end_position == std::streampos{-1}) {
        throw std::runtime_error(
            "Failed to determine image file size: "
            + path_to_utf8(image_path)
        );
    }

    const std::streamoff file_size =
        end_position - std::streampos{0};

    if (file_size <= 0) {
        throw std::runtime_error(
            "Image file is empty: " + path_to_utf8(image_path)
        );
    }

    const auto unsigned_file_size =
        static_cast<std::uintmax_t>(file_size);

    if (
        unsigned_file_size
        > static_cast<std::uintmax_t>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw std::runtime_error(
            "Image file is too large to fit in memory: "
            + path_to_utf8(image_path)
        );
    }

    if (
        unsigned_file_size
        > static_cast<std::uintmax_t>(
            std::numeric_limits<std::streamsize>::max()
        )
    ) {
        throw std::runtime_error(
            "Image file is too large for a single stream read: "
            + path_to_utf8(image_path)
        );
    }

    std::vector<std::uint8_t> encoded_bytes(
        static_cast<std::size_t>(unsigned_file_size)
    );

    input.seekg(0, std::ios::beg);

    if (!input.good()) {
        throw std::runtime_error(
            "Failed to seek to the beginning of image file: "
            + path_to_utf8(image_path)
        );
    }

    input.read(
        reinterpret_cast<char*>(encoded_bytes.data()),
        static_cast<std::streamsize>(encoded_bytes.size())
    );

    if (
        !input
        || input.gcount()
            != static_cast<std::streamsize>(encoded_bytes.size())
    ) {
        throw std::runtime_error(
            "Failed to read the complete image file: "
            + path_to_utf8(image_path)
        );
    }

    cv::Mat decoded_image;

    try {
        decoded_image = cv::imdecode(
            encoded_bytes,
            cv::IMREAD_COLOR
        );
    } catch (const cv::Exception& exception) {
        throw std::runtime_error(
            "OpenCV failed to decode image '"
            + path_to_utf8(image_path)
            + "': "
            + exception.what()
        );
    }

    if (decoded_image.empty()) {
        throw std::runtime_error(
            "OpenCV could not decode image: "
            + path_to_utf8(image_path)
        );
    }

    if (
        decoded_image.dims != 2
        || decoded_image.type() != CV_8UC3
    ) {
        throw std::runtime_error(
            "Decoded image does not have the expected CV_8UC3 format: "
            + path_to_utf8(image_path)
        );
    }

    return decoded_image;
}

/**
 * @brief Execute the complete image-to-tensor preprocessing pipeline.
 */
PreprocessResult ImageProcessor::preprocess(
    const cv::Mat& bgr_image,
    MutableTensorView destination
) {
    validate_source_image(bgr_image);
    validate_destination(destination);

    const LetterboxTransform transform =
        compute_letterbox_transform(bgr_image);

    if (!transform.valid()) {
        throw std::runtime_error(
            "Computed letterbox geometry is internally inconsistent."
        );
    }

    try {
        prepare_letterboxed_bgr(
            bgr_image,
            transform
        );
    } catch (const cv::Exception& exception) {
        throw std::runtime_error(
            std::string{"OpenCV preprocessing failed: "}
            + exception.what()
        );
    }

    switch (destination.element_type) {
        case TensorElementType::Float32:
            write_nchw_float32(
                static_cast<float*>(destination.data)
            );
            break;

        case TensorElementType::Float16:
            write_nchw_float16(
                static_cast<std::uint16_t*>(destination.data)
            );
            break;
    }

    return PreprocessResult{
        tensor_shape(),
        destination.element_type,
        required_element_count(),
        required_byte_count(destination.element_type),
        transform
    };
}

/**
 * @brief Release only the reusable OpenCV pixel workspace.
 */
void ImageProcessor::release_workspace() noexcept {
    letterboxed_bgr_.release();
}

/**
 * @brief Reproduce Ultralytics letterbox resize and padding calculations.
 */
LetterboxTransform ImageProcessor::compute_letterbox_transform(
    const cv::Mat& bgr_image
) const {
    const int original_width = bgr_image.cols;
    const int original_height = bgr_image.rows;

    const int network_width = config_.network_size.width;
    const int network_height = config_.network_size.height;

    const double width_scale =
        static_cast<double>(network_width)
        / static_cast<double>(original_width);

    const double height_scale =
        static_cast<double>(network_height)
        / static_cast<double>(original_height);

    double resize_ratio = std::min(
        width_scale,
        height_scale
    );

    if (!config_.allow_scale_up) {
        resize_ratio = std::min(resize_ratio, 1.0);
    }

    int resized_width = python_round_to_int(
        static_cast<double>(original_width) * resize_ratio
    );

    int resized_height = python_round_to_int(
        static_cast<double>(original_height) * resize_ratio
    );

    if (
        resized_width <= 0
        || resized_height <= 0
        || resized_width > network_width
        || resized_height > network_height
    ) {
        throw std::runtime_error(
            "Letterbox resize produced an invalid intermediate dimension."
        );
    }

    double horizontal_padding = static_cast<double>(
        network_width - resized_width
    );

    double vertical_padding = static_cast<double>(
        network_height - resized_height
    );

    if (config_.center_letterbox) {
        horizontal_padding /= 2.0;
        vertical_padding /= 2.0;
    }

    // The ±0.1 convention is intentionally identical to Ultralytics. It
    // deterministically assigns the extra pixel when total padding is odd.
    const int pad_left = config_.center_letterbox
        ? python_round_to_int(horizontal_padding - 0.1)
        : 0;

    const int pad_top = config_.center_letterbox
        ? python_round_to_int(vertical_padding - 0.1)
        : 0;

    const int pad_right =
        python_round_to_int(horizontal_padding + 0.1);

    const int pad_bottom =
        python_round_to_int(vertical_padding + 0.1);

    return LetterboxTransform{
        ImageSize{original_width, original_height},
        ImageSize{resized_width, resized_height},
        ImageSize{network_width, network_height},
        static_cast<float>(
            static_cast<double>(resized_width)
            / static_cast<double>(original_width)
        ),
        static_cast<float>(
            static_cast<double>(resized_height)
            / static_cast<double>(original_height)
        ),
        pad_left,
        pad_top,
        pad_right,
        pad_bottom
    };
}

/**
 * @brief Select the concrete OpenCV interpolation constant.
 */
int ImageProcessor::resolve_interpolation(
    const LetterboxTransform& transform
) const noexcept {
    switch (config_.interpolation) {
        case ResizeInterpolation::Linear:
            return cv::INTER_LINEAR;

        case ResizeInterpolation::Nearest:
            return cv::INTER_NEAREST;

        case ResizeInterpolation::Area:
            return cv::INTER_AREA;

        case ResizeInterpolation::Cubic:
            return cv::INTER_CUBIC;

        case ResizeInterpolation::Automatic:
            break;
    }

    const bool is_downscaling =
        transform.resized_size.width < transform.original_size.width
        || transform.resized_size.height < transform.original_size.height;

    return is_downscaling
        ? cv::INTER_AREA
        : cv::INTER_LINEAR;
}

/**
 * @brief Prepare the final BGR letterbox buffer without an intermediate image.
 */
void ImageProcessor::prepare_letterboxed_bgr(
    const cv::Mat& bgr_image,
    const LetterboxTransform& transform
) {
    letterboxed_bgr_.create(
        transform.network_size.height,
        transform.network_size.width,
        CV_8UC3
    );

    const bool padding_required =
        transform.pad_left != 0
        || transform.pad_top != 0
        || transform.pad_right != 0
        || transform.pad_bottom != 0;

    if (padding_required) {
        letterboxed_bgr_.setTo(
            cv::Scalar{
                static_cast<double>(config_.padding_bgr[0]),
                static_cast<double>(config_.padding_bgr[1]),
                static_cast<double>(config_.padding_bgr[2])
            }
        );
    }

    const cv::Rect resized_region{
        transform.pad_left,
        transform.pad_top,
        transform.resized_size.width,
        transform.resized_size.height
    };

    cv::Mat resized_view = letterboxed_bgr_(resized_region);

    const bool resize_required =
        transform.resized_size.width != transform.original_size.width
        || transform.resized_size.height != transform.original_size.height;

    if (resize_required) {
        cv::resize(
            bgr_image,
            resized_view,
            resized_view.size(),
            0.0,
            0.0,
            resolve_interpolation(transform)
        );
    } else {
        bgr_image.copyTo(resized_view);
    }

    if (
        letterboxed_bgr_.cols != transform.network_size.width
        || letterboxed_bgr_.rows != transform.network_size.height
        || letterboxed_bgr_.type() != CV_8UC3
    ) {
        throw std::runtime_error(
            "Letterbox workspace has an unexpected shape or scalar type."
        );
    }
}

/**
 * @brief Convert the prepared image directly to contiguous FP32 NCHW memory.
 */
void ImageProcessor::write_nchw_float32(
    float* destination
) const {
    write_nchw_from_lookup(
        letterboxed_bgr_,
        source_channel_for_output_,
        float32_lookup_,
        destination
    );
}

/**
 * @brief Convert the prepared image directly to contiguous FP16 NCHW memory.
 */
void ImageProcessor::write_nchw_float16(
    std::uint16_t* destination
) const {
    write_nchw_from_lookup(
        letterboxed_bgr_,
        source_channel_for_output_,
        float16_lookup_,
        destination
    );
}

/**
 * @brief Cache channel mapping, normalization, and FP16 conversion results.
 */
void ImageProcessor::rebuild_normalization_tables() noexcept {
    source_channel_for_output_ =
        config_.output_channel_order == ChannelOrder::RGB
            ? std::array<std::size_t, kChannelCount>{2U, 1U, 0U}
            : std::array<std::size_t, kChannelCount>{0U, 1U, 2U};

    for (
        std::size_t channel = 0U;
        channel < kChannelCount;
        ++channel
    ) {
        for (
            std::size_t raw_value = 0U;
            raw_value < kUint8ValueCount;
            ++raw_value
        ) {
            const float scaled_value =
                static_cast<float>(raw_value)
                * config_.pixel_scale;

            const float normalized_value =
                (
                    scaled_value
                    - config_.mean[channel]
                ) / config_.stddev[channel];

            float32_lookup_[channel][raw_value] = normalized_value;

            float16_lookup_[channel][raw_value] =
                float32_to_float16_bits(normalized_value);
        }
    }
}

/**
 * @brief Convert IEEE-754 float32 to binary16 with ties-to-even rounding.
 */
std::uint16_t ImageProcessor::float32_to_float16_bits(
    float value
) noexcept {
    static_assert(
        sizeof(float) == sizeof(std::uint32_t),
        "float must occupy exactly 32 bits."
    );

    static_assert(
        std::numeric_limits<float>::is_iec559,
        "float must use the IEEE-754 representation."
    );

    std::uint32_t bits = 0U;

    std::memcpy(
        &bits,
        &value,
        sizeof(bits)
    );

    const std::uint16_t sign = static_cast<std::uint16_t>(
        (bits >> 16U) & 0x8000U
    );

    const std::uint32_t exponent =
        (bits >> 23U) & 0xFFU;

    const std::uint32_t mantissa =
        bits & 0x7FFFFFU;

    // Preserve infinity and canonicalize every NaN to a quiet half NaN.
    if (exponent == 0xFFU) {
        if (mantissa == 0U) {
            return static_cast<std::uint16_t>(
                sign | 0x7C00U
            );
        }

        return static_cast<std::uint16_t>(
            sign | 0x7E00U
        );
    }

    int half_exponent =
        static_cast<int>(exponent) - 127 + 15;

    // Values above the binary16 finite range become signed infinity.
    if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(
            sign | 0x7C00U
        );
    }

    // Values below the normal binary16 range become subnormal or signed zero.
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return sign;
        }

        const std::uint32_t significand =
            mantissa | 0x800000U;

        const int shift = 14 - half_exponent;

        std::uint32_t half_mantissa =
            significand >> static_cast<unsigned int>(shift);

        const std::uint32_t remainder_mask =
            (1U << static_cast<unsigned int>(shift)) - 1U;

        const std::uint32_t remainder =
            significand & remainder_mask;

        const std::uint32_t halfway =
            1U << static_cast<unsigned int>(shift - 1);

        if (
            remainder > halfway
            || (
                remainder == halfway
                && (half_mantissa & 1U) != 0U
            )
        ) {
            ++half_mantissa;
        }

        return static_cast<std::uint16_t>(
            sign | half_mantissa
        );
    }

    // Normal values retain ten mantissa bits using ties-to-even rounding.
    std::uint32_t half_mantissa =
        mantissa >> 13U;

    const std::uint32_t remainder =
        mantissa & 0x1FFFU;

    if (
        remainder > 0x1000U
        || (
            remainder == 0x1000U
            && (half_mantissa & 1U) != 0U
        )
    ) {
        ++half_mantissa;

        if (half_mantissa == 0x400U) {
            half_mantissa = 0U;
            ++half_exponent;

            if (half_exponent >= 31) {
                return static_cast<std::uint16_t>(
                    sign | 0x7C00U
                );
            }
        }
    }

    return static_cast<std::uint16_t>(
        sign
        | (
            static_cast<std::uint16_t>(half_exponent)
            << 10U
        )
        | static_cast<std::uint16_t>(half_mantissa)
    );
}

/**
 * @brief Reject unsupported or unreadable OpenCV source images.
 */
void ImageProcessor::validate_source_image(
    const cv::Mat& bgr_image
) const {
    if (bgr_image.empty()) {
        throw std::invalid_argument(
            "The source image must not be empty."
        );
    }

    if (bgr_image.dims != 2) {
        throw std::invalid_argument(
            "The source image must be two-dimensional."
        );
    }

    if (bgr_image.type() != CV_8UC3) {
        throw std::invalid_argument(
            "The source image must have type CV_8UC3 in BGR order."
        );
    }

    if (bgr_image.cols <= 0 || bgr_image.rows <= 0) {
        throw std::invalid_argument(
            "The source image must have positive dimensions."
        );
    }

    if (bgr_image.data == nullptr) {
        throw std::invalid_argument(
            "The source image has no readable pixel storage."
        );
    }
}

/**
 * @brief Reject unsupported, undersized, null, or misaligned destinations.
 */
void ImageProcessor::validate_destination(
    MutableTensorView destination
) const {
    if (!is_supported_tensor_type(destination.element_type)) {
        throw std::invalid_argument(
            "The destination tensor element type is invalid."
        );
    }

    if (destination.data == nullptr) {
        throw std::invalid_argument(
            "The destination tensor pointer must not be null."
        );
    }

    const std::size_t required_bytes =
        required_byte_count(destination.element_type);

    if (destination.byte_capacity < required_bytes) {
        throw std::invalid_argument(
            "The destination tensor buffer is too small. Required "
            + std::to_string(required_bytes)
            + " bytes, received "
            + std::to_string(destination.byte_capacity)
            + " bytes."
        );
    }

    const std::uintptr_t address =
        reinterpret_cast<std::uintptr_t>(destination.data);

    const std::size_t required_alignment =
        destination.element_type == TensorElementType::Float32
            ? alignof(float)
            : alignof(std::uint16_t);

    if (address % required_alignment != 0U) {
        throw std::invalid_argument(
            "The destination tensor pointer is not correctly aligned."
        );
    }
}

}  // namespace edge