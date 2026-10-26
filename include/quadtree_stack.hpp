#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mp48mb {

constexpr uint32_t kMaxImageDimension = 4096u;

struct QuadStackEntry {
    uint32_t raw = 0;

    static QuadStackEntry pack(uint16_t x, uint16_t y, uint8_t log2_size, uint8_t flags = 0) {
        if ((x > 0x0FFFu) || (y > 0x0FFFu) || (log2_size > 0x0Fu) || (flags > 0x0Fu)) {
            throw std::invalid_argument("QuadStackEntry exceeds the 32-bit bit layout");
        }

        uint32_t value = 0;
        value |= static_cast<uint32_t>(x & 0x0FFFu);
        value |= (static_cast<uint32_t>(y & 0x0FFFu) << 12u);
        value |= (static_cast<uint32_t>(log2_size & 0x0Fu) << 24u);
        value |= (static_cast<uint32_t>(flags & 0x0Fu) << 28u);
        return QuadStackEntry{value};
    }

    uint16_t x() const { return static_cast<uint16_t>(raw & 0x0FFFu); }
    uint16_t y() const { return static_cast<uint16_t>((raw >> 12u) & 0x0FFFu); }
    uint8_t log2_size() const { return static_cast<uint8_t>((raw >> 24u) & 0x0Fu); }
    uint8_t flags() const { return static_cast<uint8_t>((raw >> 28u) & 0x0Fu); }

    uint32_t size() const { return 1u << log2_size(); }
    uint32_t depth() const { return static_cast<uint32_t>(log2_size()); }
};

inline uint32_t worst_case_stack_capacity(uint32_t max_dimension, uint8_t max_depth) {
    if (max_dimension == 0u || max_depth == 0u) {
        return 1u;
    }
    uint32_t root_size = 1u;
    while ((root_size << 1u) <= max_dimension) {
        root_size <<= 1u;
    }
    uint32_t capacity = 1u;
    for (uint8_t level = 0; level < max_depth; ++level) {
        capacity *= 4u;
    }
    if (capacity < 1u) {
        capacity = 1u;
    }
    if (root_size > kMaxImageDimension) {
        throw std::invalid_argument("Requested image dimensions exceed the documented 4096-pixel limit");
    }
    return capacity + 16u;
}

inline bool is_valid_stack_entry(const QuadStackEntry& entry, uint32_t image_width, uint32_t image_height) {
    uint32_t size = entry.size();
    return entry.x() + size <= image_width && entry.y() + size <= image_height && size > 0u;
}

inline std::string describe_stack_layout() {
    return "Layout: 12 bits x, 12 bits y, 4 bits log2(block_size), 4 bits flags/reserved."
           " Total width = 32 bits; entries support up to 4096x4096 images with a root size of 4096.";
}

}  // namespace mp48mb
