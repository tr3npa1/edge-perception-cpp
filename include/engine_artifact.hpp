#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace edge {
namespace engine_artifact_detail {

// Validate the envelope without interpreting metadata as runtime configuration.
// Limits prevent malformed files from causing unbounded parsing/recursion.
class JsonObjectValidator {
public:
    explicit JsonObjectValidator(std::string_view text) : text_(text) {}
    bool valid() {
        space();
        return peek() == '{' && value(0) && (space(), position_ == text_.size());
    }
private:
    char peek() const { return position_ < text_.size() ? text_[position_] : '\0'; }
    void space() { while (peek() == ' ' || peek() == '\n' || peek() == '\r' || peek() == '\t') ++position_; }
    bool take(char c) { if (peek() != c) return false; ++position_; return true; }
    bool literal(std::string_view word) {
        if (text_.substr(position_, word.size()) != word) return false;
        position_ += word.size(); return true;
    }
    bool string() {
        if (!take('"')) return false;
        while (position_ < text_.size()) {
            const auto c = static_cast<unsigned char>(text_[position_++]);
            if (c == '"') return true;
            if (c < 0x20) return false;
            if (c != '\\') continue;
            if (position_ == text_.size()) return false;
            const char escape = text_[position_++];
            if (escape == 'u') {
                for (int i = 0; i < 4; ++i) {
                    const char digit = peek();
                    if (!((digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f')
                        || (digit >= 'A' && digit <= 'F'))) return false;
                    ++position_;
                }
            } else if (std::string_view{"\"\\/bfnrt"}.find(escape) == std::string_view::npos) return false;
        }
        return false;
    }
    bool number() {
        take('-');
        if (!take('0')) {
            if (peek() < '1' || peek() > '9') return false;
            while (peek() >= '0' && peek() <= '9') ++position_;
        }
        if (take('.')) {
            if (peek() < '0' || peek() > '9') return false;
            while (peek() >= '0' && peek() <= '9') ++position_;
        }
        if (take('e') || take('E')) {
            if (!take('+')) take('-');
            if (peek() < '0' || peek() > '9') return false;
            while (peek() >= '0' && peek() <= '9') ++position_;
        }
        return true;
    }
    bool value(unsigned depth) {
        if (depth > 64) return false;
        space();
        if (peek() == '"') return string();
        if (peek() == '{' || peek() == '[') {
            const bool object = take('{');
            if (!object) ++position_;
            const char end = object ? '}' : ']';
            space();
            if (take(end)) return true;
            do {
                space();
                if (object && (!string() || (space(), !take(':')))) return false;
                if (!value(depth + 1)) return false;
                space();
                if (take(end)) return true;
            } while (take(','));
            return false;
        }
        if (peek() == 't') return literal("true");
        if (peek() == 'f') return literal("false");
        if (peek() == 'n') return literal("null");
        return number();
    }
    std::string_view text_;
    std::size_t position_ = 0;
};
}  // namespace engine_artifact_detail

struct TensorRtPayload {
    const std::uint8_t* data;
    std::size_t size;
    std::size_t metadata_bytes;
};

// Ultralytics envelope: uint32 little-endian JSON byte length, JSON object,
// followed by the unchanged TensorRT plan. Raw plans pass through untouched.
inline TensorRtPayload tensorrt_payload(const std::vector<std::uint8_t>& bytes) {
    if (bytes.empty()) throw std::runtime_error("TensorRT engine file is empty.");
    if (bytes.size() <= 4) return {bytes.data(), bytes.size(), 0};
    std::size_t first = 4;
    while (first < bytes.size() && (bytes[first] == ' ' || bytes[first] == '\r'
        || bytes[first] == '\n' || bytes[first] == '\t')) ++first;
    if (first == bytes.size() || bytes[first] != '{') return {bytes.data(), bytes.size(), 0};
    const std::uint32_t length = std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8)
        | (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
    if (length == 0 || length > 16U * 1024U * 1024U || length >= bytes.size() - 4)
        throw std::runtime_error("Invalid/truncated Ultralytics engine metadata or missing plan.");
    const std::string_view metadata{reinterpret_cast<const char*>(bytes.data() + 4), length};
    if (!engine_artifact_detail::JsonObjectValidator{metadata}.valid())
        throw std::runtime_error("Malformed Ultralytics engine metadata JSON.");
    return {bytes.data() + 4 + length, bytes.size() - 4 - length, length};
}
}  // namespace edge
