#include "codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define MP48_HAS_AVX2_TARGET 1
#endif

namespace mp48mb {
namespace {

constexpr uint32_t kMagic = 0x4D503438u; // 'MP48'
constexpr uint8_t kVersion = 3u;
constexpr uint8_t kPixelFormatRgb = 3u;
constexpr std::array<uint8_t, 8> kFixedRgb565Magic{'M', 'P', '4', '8', 'B', '5', '6', '5'};
constexpr uint8_t kFixedRgb565Version = 1u;
constexpr uint8_t kFixedRgb565Format = 1u;
constexpr std::size_t kFixedRgb565HeaderSize = 20u;
constexpr uint32_t kMaxFixedBlockDimension = 4096u;
constexpr uint64_t kMaxFixedBlockPixels = 16'800'000u;

uint32_t read_le_u32(const std::vector<uint8_t>& bytes, std::size_t offset) {
    if (offset + 4u > bytes.size()) {
        throw std::runtime_error("Unexpected end of stream while reading uint32_t");
    }
    return static_cast<uint32_t>(bytes[offset]) |
           (static_cast<uint32_t>(bytes[offset + 1u]) << 8u) |
           (static_cast<uint32_t>(bytes[offset + 2u]) << 16u) |
           (static_cast<uint32_t>(bytes[offset + 3u]) << 24u);
}

uint16_t read_le_u16(const std::vector<uint8_t>& bytes, std::size_t offset) {
    if (offset + 2u > bytes.size()) {
        throw std::runtime_error("Unexpected end of stream while reading uint16_t");
    }
    return static_cast<uint16_t>(bytes[offset]) |
           (static_cast<uint16_t>(bytes[offset + 1u]) << 8u);
}

void write_le_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    bytes.push_back(static_cast<uint8_t>(value & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 8u) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 16u) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 24u) & 0xFFu));
}

void write_le_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 8u) & 0xFFu));
}

uint8_t expand_5_to_8(uint8_t value) {
    return static_cast<uint8_t>((value << 3u) | (value >> 2u));
}

uint8_t expand_6_to_8(uint8_t value) {
    return static_cast<uint8_t>((value << 2u) | (value >> 4u));
}

uint8_t clamp_component(uint64_t v) {
    if (v > 255u) {
        return 255u;
    }
    return static_cast<uint8_t>(v);
}

struct ResidualColor {
    int8_t r = 0;
    int8_t g = 0;
    int8_t b = 0;
};

ResidualColor make_residual(const Pixel& current, const Pixel& previous) {
    return {
        static_cast<int8_t>(std::clamp<int>(static_cast<int>(current.r) - static_cast<int>(previous.r), -128, 127)),
        static_cast<int8_t>(std::clamp<int>(static_cast<int>(current.g) - static_cast<int>(previous.g), -128, 127)),
        static_cast<int8_t>(std::clamp<int>(static_cast<int>(current.b) - static_cast<int>(previous.b), -128, 127))
    };
}

Pixel reconstruct_from_residual(const Pixel& previous, const ResidualColor& residual) {
    Pixel result{};
    result.r = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(previous.r) + static_cast<int>(residual.r), 0, 255));
    result.g = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(previous.g) + static_cast<int>(residual.g), 0, 255));
    result.b = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(previous.b) + static_cast<int>(residual.b), 0, 255));
    return result;
}

struct Region {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
};

BlockStats compute_region_stats(const Image& image, const Region& region, uint32_t x, uint32_t y, uint32_t block_size) {
    BlockStats stats{};
    const uint32_t x_end = std::min({x + block_size, region.x + region.width, image.width()});
    const uint32_t y_end = std::min({y + block_size, region.y + region.height, image.height()});
    if (x >= x_end || y >= y_end) {
        return stats;
    }

    for (uint32_t yy = y; yy < y_end; ++yy) {
        for (uint32_t xx = x; xx < x_end; ++xx) {
            const Pixel pixel = image.at(xx, yy);
            ++stats.pixel_count;
            stats.sum_r += pixel.r;
            stats.sum_g += pixel.g;
            stats.sum_b += pixel.b;
            stats.sum_sq_r += static_cast<uint64_t>(pixel.r) * pixel.r;
            stats.sum_sq_g += static_cast<uint64_t>(pixel.g) * pixel.g;
            stats.sum_sq_b += static_cast<uint64_t>(pixel.b) * pixel.b;
        }
    }

    const double count = static_cast<double>(stats.pixel_count);
    const double mean_r = static_cast<double>(stats.sum_r) / count;
    const double mean_g = static_cast<double>(stats.sum_g) / count;
    const double mean_b = static_cast<double>(stats.sum_b) / count;
    const double variance_r = static_cast<double>(stats.sum_sq_r) / count - mean_r * mean_r;
    const double variance_g = static_cast<double>(stats.sum_sq_g) / count - mean_g * mean_g;
    const double variance_b = static_cast<double>(stats.sum_sq_b) / count - mean_b * mean_b;
    stats.mse = std::max(0.0, (variance_r + variance_g + variance_b) / 3.0);
    stats.avg = Pixel{
        static_cast<uint8_t>(stats.sum_r / stats.pixel_count),
        static_cast<uint8_t>(stats.sum_g / stats.pixel_count),
        static_cast<uint8_t>(stats.sum_b / stats.pixel_count)
    };
    return stats;
}

}  // namespace

uint32_t FrameCodec::next_power_of_two(uint32_t value) {
    if (value == 0u) {
        return 1u;
    }
    uint32_t result = 1u;
    while (result < value) {
        result <<= 1u;
    }
    return result;
}

uint8_t FrameCodec::log2_block_size(uint32_t value) {
    uint8_t log2 = 0;
    uint32_t current = 1u;
    while (current < value) {
        current <<= 1u;
        ++log2;
    }
    return log2;
}

BlockStats compute_block_stats_scalar(const Image& image, uint32_t x, uint32_t y, uint32_t block_size) {
    BlockStats stats{};
    const uint32_t x_end = std::min(x + block_size, image.width());
    const uint32_t y_end = std::min(y + block_size, image.height());

    if (x_end <= x || y_end <= y) {
        return stats;
    }

    for (uint32_t yy = y; yy < y_end; ++yy) {
        for (uint32_t xx = x; xx < x_end; ++xx) {
            const Pixel p = image.at(xx, yy);
            ++stats.pixel_count;
            stats.sum_r += p.r;
            stats.sum_g += p.g;
            stats.sum_b += p.b;
            stats.sum_sq_r += static_cast<uint64_t>(p.r) * p.r;
            stats.sum_sq_g += static_cast<uint64_t>(p.g) * p.g;
            stats.sum_sq_b += static_cast<uint64_t>(p.b) * p.b;
        }
    }

    if (stats.pixel_count == 0u) {
        return stats;
    }

    const double inv_n = 1.0 / static_cast<double>(stats.pixel_count);
    const double mean_r = static_cast<double>(stats.sum_r) * inv_n;
    const double mean_g = static_cast<double>(stats.sum_g) * inv_n;
    const double mean_b = static_cast<double>(stats.sum_b) * inv_n;

    double sq_err_r = 0.0;
    double sq_err_g = 0.0;
    double sq_err_b = 0.0;

    for (uint32_t yy = y; yy < y_end; ++yy) {
        for (uint32_t xx = x; xx < x_end; ++xx) {
            const Pixel p = image.at(xx, yy);
            sq_err_r += (static_cast<double>(p.r) - mean_r) * (static_cast<double>(p.r) - mean_r);
            sq_err_g += (static_cast<double>(p.g) - mean_g) * (static_cast<double>(p.g) - mean_g);
            sq_err_b += (static_cast<double>(p.b) - mean_b) * (static_cast<double>(p.b) - mean_b);
        }
    }
    stats.mse = (sq_err_r + sq_err_g + sq_err_b) * inv_n / 3.0;

    stats.avg.r = clamp_component(stats.sum_r / stats.pixel_count);
    stats.avg.g = clamp_component(stats.sum_g / stats.pixel_count);
    stats.avg.b = clamp_component(stats.sum_b / stats.pixel_count);

    return stats;
}

#if defined(MP48_HAS_AVX2_TARGET)
__attribute__((target("avx2")))
BlockStats compute_block_stats_simd_avx2(const Image& image, uint32_t x, uint32_t y, uint32_t block_size) {
    BlockStats stats{};
    const uint32_t x_end = std::min(x + block_size, image.width());
    const uint32_t y_end = std::min(y + block_size, image.height());
    const uint32_t width = x_end - x;
    const uint32_t height = y_end - y;
    if (width == 0u || height == 0u) {
        return stats;
    }

    std::vector<uint8_t> r_values;
    std::vector<uint8_t> g_values;
    std::vector<uint8_t> b_values;
    r_values.reserve(static_cast<std::size_t>(width) * height);
    g_values.reserve(static_cast<std::size_t>(width) * height);
    b_values.reserve(static_cast<std::size_t>(width) * height);

    for (uint32_t yy = y; yy < y_end; ++yy) {
        for (uint32_t xx = x; xx < x_end; ++xx) {
            const Pixel p = image.at(xx, yy);
            r_values.push_back(p.r);
            g_values.push_back(p.g);
            b_values.push_back(p.b);
        }
    }

    const std::size_t pixel_count = r_values.size();
    std::size_t i = 0u;
    for (; i + 8u <= pixel_count; i += 8u) {
        const __m128i vr8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&r_values[i]));
        const __m128i vg8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&g_values[i]));
        const __m128i vb8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&b_values[i]));
        const __m256i vr16 = _mm256_cvtepu8_epi16(vr8);
        const __m256i vg16 = _mm256_cvtepu8_epi16(vg8);
        const __m256i vb16 = _mm256_cvtepu8_epi16(vb8);
        const __m256i vr_sq = _mm256_mullo_epi16(vr16, vr16);
        const __m256i vg_sq = _mm256_mullo_epi16(vg16, vg16);
        const __m256i vb_sq = _mm256_mullo_epi16(vb16, vb16);

        alignas(32) uint16_t r_lanes[16] = {};
        alignas(32) uint16_t g_lanes[16] = {};
        alignas(32) uint16_t b_lanes[16] = {};
        alignas(32) uint16_t rr_lanes[16] = {};
        alignas(32) uint16_t gg_lanes[16] = {};
        alignas(32) uint16_t bb_lanes[16] = {};
        _mm256_store_si256(reinterpret_cast<__m256i*>(r_lanes), vr16);
        _mm256_store_si256(reinterpret_cast<__m256i*>(g_lanes), vg16);
        _mm256_store_si256(reinterpret_cast<__m256i*>(b_lanes), vb16);
        _mm256_store_si256(reinterpret_cast<__m256i*>(rr_lanes), vr_sq);
        _mm256_store_si256(reinterpret_cast<__m256i*>(gg_lanes), vg_sq);
        _mm256_store_si256(reinterpret_cast<__m256i*>(bb_lanes), vb_sq);

        for (int j = 0; j < 8; ++j) {
            stats.sum_r += r_lanes[j];
            stats.sum_g += g_lanes[j];
            stats.sum_b += b_lanes[j];
            stats.sum_sq_r += rr_lanes[j];
            stats.sum_sq_g += gg_lanes[j];
            stats.sum_sq_b += bb_lanes[j];
        }
        stats.pixel_count += 8u;
    }

    for (; i < pixel_count; ++i) {
        const uint8_t r = r_values[i];
        const uint8_t g = g_values[i];
        const uint8_t b = b_values[i];
        stats.sum_r += r;
        stats.sum_g += g;
        stats.sum_b += b;
        stats.sum_sq_r += static_cast<uint64_t>(r) * r;
        stats.sum_sq_g += static_cast<uint64_t>(g) * g;
        stats.sum_sq_b += static_cast<uint64_t>(b) * b;
        ++stats.pixel_count;
    }

    if (stats.pixel_count == 0u) {
        return stats;
    }

    const double inv_n = 1.0 / static_cast<double>(stats.pixel_count);
    const double mean_r = static_cast<double>(stats.sum_r) * inv_n;
    const double mean_g = static_cast<double>(stats.sum_g) * inv_n;
    const double mean_b = static_cast<double>(stats.sum_b) * inv_n;
    stats.avg.r = clamp_component(static_cast<uint64_t>(mean_r));
    stats.avg.g = clamp_component(static_cast<uint64_t>(mean_g));
    stats.avg.b = clamp_component(static_cast<uint64_t>(mean_b));
    stats.mse = ((static_cast<double>(stats.sum_sq_r) / static_cast<double>(stats.pixel_count)) - (mean_r * mean_r) +
                 (static_cast<double>(stats.sum_sq_g) / static_cast<double>(stats.pixel_count)) - (mean_g * mean_g) +
                 (static_cast<double>(stats.sum_sq_b) / static_cast<double>(stats.pixel_count)) - (mean_b * mean_b)) /
                3.0;
    return stats;
}
#endif

BlockStats compute_block_stats_simd(const Image& image, uint32_t x, uint32_t y, uint32_t block_size) {
#if defined(MP48_HAS_AVX2_TARGET)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2")) {
        return compute_block_stats_simd_avx2(image, x, y, block_size);
    }
#endif
    return compute_block_stats_scalar(image, x, y, block_size);
}

BlockStats compute_block_stats(const Image& image, uint32_t x, uint32_t y, uint32_t block_size) {
    return compute_block_stats_scalar(image, x, y, block_size);
}

BlockStats compute_block_stats(const Image& image, uint32_t x, uint32_t y, uint32_t block_size, BlockStatsStrategy strategy) {
    switch (strategy) {
        case BlockStatsStrategy::SIMD:
            return compute_block_stats_simd(image, x, y, block_size);
        case BlockStatsStrategy::Scalar:
        default:
            return compute_block_stats_scalar(image, x, y, block_size);
    }
}

std::vector<uint8_t> FixedBlockRgb565Codec::encode(const Image& image, uint32_t block_size) {
    if (image.width() == 0u || image.height() == 0u
        || image.width() > kMaxFixedBlockDimension || image.height() > kMaxFixedBlockDimension
        || static_cast<uint64_t>(image.width()) * image.height() > kMaxFixedBlockPixels) {
        throw std::invalid_argument("Fixed-block image dimensions exceed the codec resource limits");
    }
    if (block_size != 8u) {
        throw std::invalid_argument("The experimental RGB565 format currently supports only 8x8 blocks");
    }

    const uint32_t blocks_x = (image.width() + block_size - 1u) / block_size;
    const uint32_t blocks_y = (image.height() + block_size - 1u) / block_size;
    const uint64_t block_count = static_cast<uint64_t>(blocks_x) * blocks_y;
    if (block_count > (std::numeric_limits<std::size_t>::max() - kFixedRgb565HeaderSize) / 2u) {
        throw std::length_error("Fixed-block encoded image is too large");
    }

    std::vector<uint8_t> encoded;
    encoded.reserve(kFixedRgb565HeaderSize + static_cast<std::size_t>(block_count) * 2u);
    encoded.insert(encoded.end(), kFixedRgb565Magic.begin(), kFixedRgb565Magic.end());
    encoded.push_back(kFixedRgb565Version);
    encoded.push_back(static_cast<uint8_t>(block_size));
    encoded.push_back(kFixedRgb565Format);
    encoded.push_back(0u);
    write_le_u32(encoded, image.width());
    write_le_u32(encoded, image.height());

    for (uint32_t block_y = 0; block_y < image.height(); block_y += block_size) {
        const uint32_t block_height = std::min(block_size, image.height() - block_y);
        for (uint32_t block_x = 0; block_x < image.width(); block_x += block_size) {
            const uint32_t block_width = std::min(block_size, image.width() - block_x);
            uint64_t red_sum = 0;
            uint64_t green_sum = 0;
            uint64_t blue_sum = 0;
            const uint64_t pixel_count = static_cast<uint64_t>(block_width) * block_height;
            for (uint32_t y = block_y; y < block_y + block_height; ++y) {
                for (uint32_t x = block_x; x < block_x + block_width; ++x) {
                    const Pixel pixel = image.at(x, y);
                    red_sum += pixel.r;
                    green_sum += pixel.g;
                    blue_sum += pixel.b;
                }
            }
            const uint32_t red = static_cast<uint32_t>(red_sum / pixel_count);
            const uint32_t green = static_cast<uint32_t>(green_sum / pixel_count);
            const uint32_t blue = static_cast<uint32_t>(blue_sum / pixel_count);
            const uint16_t packed = static_cast<uint16_t>(
                (((red * 31u + 127u) / 255u) << 11u)
                | (((green * 63u + 127u) / 255u) << 5u)
                | ((blue * 31u + 127u) / 255u));
            write_le_u16(encoded, packed);
        }
    }
    return encoded;
}

bool FixedBlockRgb565Codec::decode(const std::vector<uint8_t>& encoded, Image& image, std::string& error) {
    try {
        if (encoded.size() < kFixedRgb565HeaderSize) {
            throw std::runtime_error("Fixed-block stream is shorter than its header");
        }
        if (!std::equal(kFixedRgb565Magic.begin(), kFixedRgb565Magic.end(), encoded.begin())) {
            throw std::runtime_error("Invalid fixed-block RGB565 magic");
        }
        if (encoded[8u] != kFixedRgb565Version) {
            throw std::runtime_error("Unsupported fixed-block RGB565 version");
        }
        if (encoded[9u] != 8u) {
            throw std::runtime_error("Unsupported fixed-block size");
        }
        if (encoded[10u] != kFixedRgb565Format || encoded[11u] != 0u) {
            throw std::runtime_error("Unsupported fixed-block pixel format or reserved flags");
        }
        const uint32_t width = read_le_u32(encoded, 12u);
        const uint32_t height = read_le_u32(encoded, 16u);
        if (width == 0u || height == 0u || width > kMaxFixedBlockDimension || height > kMaxFixedBlockDimension
            || static_cast<uint64_t>(width) * height > kMaxFixedBlockPixels) {
            throw std::runtime_error("Invalid fixed-block image dimensions");
        }
        const uint64_t blocks_x = (static_cast<uint64_t>(width) + 7u) / 8u;
        const uint64_t blocks_y = (static_cast<uint64_t>(height) + 7u) / 8u;
        const uint64_t expected_size = kFixedRgb565HeaderSize + blocks_x * blocks_y * 2u;
        if (expected_size != encoded.size()) {
            throw std::runtime_error(
                expected_size > encoded.size()
                    ? "Fixed-block stream is truncated"
                    : "Fixed-block stream contains trailing bytes");
        }

        Image decoded(width, height);
        std::size_t offset = kFixedRgb565HeaderSize;
        for (uint32_t block_y = 0; block_y < height; block_y += 8u) {
            const uint32_t block_height = std::min(8u, height - block_y);
            for (uint32_t block_x = 0; block_x < width; block_x += 8u) {
                const uint32_t block_width = std::min(8u, width - block_x);
                const uint16_t packed = read_le_u16(encoded, offset);
                offset += 2u;
                const Pixel pixel{
                    expand_5_to_8(static_cast<uint8_t>((packed >> 11u) & 0x1Fu)),
                    expand_6_to_8(static_cast<uint8_t>((packed >> 5u) & 0x3Fu)),
                    expand_5_to_8(static_cast<uint8_t>(packed & 0x1Fu)),
                };
                for (uint32_t y = block_y; y < block_y + block_height; ++y) {
                    for (uint32_t x = block_x; x < block_x + block_width; ++x) {
                        decoded.set_pixel(x, y, pixel);
                    }
                }
            }
        }
        image = std::move(decoded);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

std::vector<uint8_t> FrameCodec::encode(const Image& image, const EncoderConfig& config) const {
    return encode(image, config, nullptr, PartitionMode::Auto);
}

std::vector<uint8_t> FrameCodec::encode(const Image& image, const EncoderConfig& config, const Image* previous_frame) const {
    return encode(image, config, previous_frame, PartitionMode::Auto);
}

std::vector<uint8_t> FrameCodec::encode(
    const Image& image,
    const EncoderConfig& config,
    const Image* previous_frame,
    PartitionMode partition_mode) const {
    if (image.width() == 0 || image.height() == 0) {
        throw std::invalid_argument("Image dimensions must be non-zero");
    }
    if (image.width() > kMaxImageDimension || image.height() > kMaxImageDimension ||
        image.width() > config.max_dimension || image.height() > config.max_dimension) {
        throw std::invalid_argument("Image dimensions exceed the configured 4096-pixel limit");
    }
    if (config.max_block_size == 0u || config.max_block_size > kMaxImageDimension ||
        (config.max_block_size & (config.max_block_size - 1u)) != 0u ||
        config.min_block_size == 0u || config.min_block_size > config.max_block_size ||
        (config.min_block_size & (config.min_block_size - 1u)) != 0u) {
        throw std::invalid_argument("Block sizes must be powers of two with 1 <= min <= max <= 4096");
    }
    if (previous_frame != nullptr &&
        (previous_frame->width() != image.width() || previous_frame->height() != image.height())) {
        throw std::invalid_argument("Previous frame dimensions must match the current frame");
    }

    const bool use_four_regions = partition_mode == PartitionMode::Four ||
        (partition_mode == PartitionMode::Auto && (image.width() > 640u || image.height() > 480u));
    std::vector<Region> regions;
    if (use_four_regions) {
        const uint32_t mid_x = image.width() / 2u;
        const uint32_t mid_y = image.height() / 2u;
        regions = {
            Region{0u, 0u, mid_x, mid_y},
            Region{mid_x, 0u, image.width() - mid_x, mid_y},
            Region{0u, mid_y, mid_x, image.height() - mid_y},
            Region{mid_x, mid_y, image.width() - mid_x, image.height() - mid_y}
        };
    } else {
        regions = {Region{0u, 0u, image.width(), image.height()}};
    }
    const bool use_prediction = previous_frame != nullptr;

    std::vector<uint8_t> bytes;
    bytes.reserve(std::min<std::size_t>(image.size() / 8u + 128u, 64u * 1024u * 1024u));

    write_le_u32(bytes, kMagic);
    bytes.push_back(kVersion);
    bytes.push_back(kPixelFormatRgb);
    write_le_u32(bytes, image.width());
    write_le_u32(bytes, image.height());
    write_le_u32(bytes, config.max_block_size);
    write_le_u32(bytes, config.min_block_size);
    write_le_u32(bytes, config.max_depth);
    write_le_u32(bytes, config.error_threshold);
    write_le_u32(bytes, next_power_of_two(std::max(image.width(), image.height())));
    bytes.push_back(use_four_regions ? 1u : 0u);
    bytes.push_back(static_cast<uint8_t>(regions.size()));
    write_le_u32(bytes, use_prediction ? 1u : 0u);

    for (const Region& region : regions) {
        write_le_u16(bytes, static_cast<uint16_t>(region.x));
        write_le_u16(bytes, static_cast<uint16_t>(region.y));
        write_le_u16(bytes, static_cast<uint16_t>(region.width));
        write_le_u16(bytes, static_cast<uint16_t>(region.height));

        std::vector<LeafNode> leaves;
        if (region.width != 0u && region.height != 0u) {
            const uint32_t root_size = next_power_of_two(std::max(region.width, region.height));
            const uint32_t root_log2 = log2_block_size(root_size);
            const uint32_t depth_limit = std::min(config.max_depth, root_log2);
            std::vector<uint32_t> stack(1u + 3u * depth_limit);
            std::size_t stack_pointer = 0u;
            stack[stack_pointer++] = QuadStackEntry::pack(
                static_cast<uint16_t>(region.x),
                static_cast<uint16_t>(region.y),
                static_cast<uint8_t>(root_log2)).raw;

            while (stack_pointer != 0u) {
                const QuadStackEntry entry{stack[--stack_pointer]};
                const uint32_t x = entry.x();
                const uint32_t y = entry.y();
                const uint32_t block_size = entry.size();
                const uint32_t depth = root_log2 - entry.log2_size();
                const bool whole_block_in_region =
                    x + block_size <= region.x + region.width &&
                    y + block_size <= region.y + region.height;
                const BlockStats stats = whole_block_in_region && config.use_simd
                    ? compute_block_stats(image, x, y, block_size, config.stats_strategy)
                    : compute_region_stats(image, region, x, y, block_size);
                const bool at_limit = block_size <= config.min_block_size ||
                    depth >= depth_limit || block_size == 1u;
                const bool within_max_block = block_size <= config.max_block_size;

                if (at_limit || (within_max_block && stats.mse <= config.error_threshold)) {
                    leaves.push_back(LeafNode{
                        static_cast<uint16_t>(x), static_cast<uint16_t>(y), static_cast<uint16_t>(block_size),
                        stats.avg.r, stats.avg.g, stats.avg.b
                    });
                    continue;
                }

                const uint32_t child_size = block_size / 2u;
                const uint8_t child_log2 = static_cast<uint8_t>(entry.log2_size() - 1u);
                const uint32_t region_right = region.x + region.width;
                const uint32_t region_bottom = region.y + region.height;
                const std::array<std::pair<uint32_t, uint32_t>, 4> origins{{
                    {x, y}, {x + child_size, y}, {x, y + child_size}, {x + child_size, y + child_size}
                }};
                for (auto child = origins.rbegin(); child != origins.rend(); ++child) {
                    if (child->first < region_right && child->second < region_bottom) {
                        if (stack_pointer >= stack.size()) {
                            throw std::runtime_error("Quadtree traversal stack capacity exceeded");
                        }
                        stack[stack_pointer++] = QuadStackEntry::pack(
                            static_cast<uint16_t>(child->first),
                            static_cast<uint16_t>(child->second),
                            child_log2).raw;
                    }
                }
            }
        }

        write_le_u32(bytes, static_cast<uint32_t>(leaves.size()));
        for (const LeafNode& leaf : leaves) {
            uint8_t flags = 0u;
            uint8_t r = leaf.r;
            uint8_t g = leaf.g;
            uint8_t b = leaf.b;
            if (use_prediction) {
                const Pixel prev_avg = compute_region_stats(*previous_frame, region, leaf.x, leaf.y, leaf.size).avg;
                const ResidualColor residual = make_residual(Pixel{leaf.r, leaf.g, leaf.b}, prev_avg);
                const int max_delta = std::max({
                    std::abs(static_cast<int>(residual.r)),
                    std::abs(static_cast<int>(residual.g)),
                    std::abs(static_cast<int>(residual.b))
                });
                if (max_delta <= 32) {
                    flags = 1u;
                    r = static_cast<uint8_t>(residual.r);
                    g = static_cast<uint8_t>(residual.g);
                    b = static_cast<uint8_t>(residual.b);
                }
            }
            write_le_u16(bytes, leaf.x);
            write_le_u16(bytes, leaf.y);
            write_le_u16(bytes, leaf.size);
            bytes.push_back(flags);
            bytes.push_back(r);
            bytes.push_back(g);
            bytes.push_back(b);
        }
    }
    return bytes;
}

bool FrameCodec::decode(const std::vector<uint8_t>& encoded, Image& image, std::string& error) const {
    return decode(encoded, image, error, nullptr);
}

bool FrameCodec::decode(const std::vector<uint8_t>& encoded, Image& image, std::string& error, const Image* previous_frame) const {
    try {
        if (encoded.size() < 38u) {
            error = "Stream too short to contain a valid frame header";
            return false;
        }
        const uint32_t magic = read_le_u32(encoded, 0u);
        if (magic != kMagic) {
            error = "Invalid stream magic: not MP48";
            return false;
        }
        const uint8_t version = encoded[4u];
        if (version < 1u || version > kVersion) {
            error = "Unsupported bitstream version";
            return false;
        }
        const uint8_t pixel_format = encoded[5u];
        if (pixel_format != kPixelFormatRgb) {
            error = "Unsupported pixel format in stream";
            return false;
        }

        const uint32_t width = read_le_u32(encoded, 6u);
        const uint32_t height = read_le_u32(encoded, 10u);

        if (width == 0u || height == 0u || width > kMaxImageDimension || height > kMaxImageDimension) {
            error = "Invalid frame dimensions in stream";
            return false;
        }

        Image reconstructed(width, height, Pixel{0, 0, 0});
        if (version == 3u) {
            if (encoded.size() < 40u) {
                error = "Truncated version 3 frame header";
                return false;
            }
            const uint8_t partition_flag = encoded[34u];
            const uint8_t region_count = encoded[35u];
            const uint32_t prediction_flag = read_le_u32(encoded, 36u);
            if (partition_flag > 1u || prediction_flag > 1u ||
                region_count != (partition_flag == 1u ? 4u : 1u)) {
                error = "Invalid partition or prediction metadata";
                return false;
            }
            const bool use_prediction = prediction_flag == 1u;
            if (use_prediction && (previous_frame == nullptr ||
                previous_frame->width() != width || previous_frame->height() != height)) {
                error = "This frame requires a matching previous reference frame";
                return false;
            }

            std::vector<Region> expected_regions;
            if (partition_flag == 1u) {
                const uint32_t mid_x = width / 2u;
                const uint32_t mid_y = height / 2u;
                expected_regions = {
                    Region{0u, 0u, mid_x, mid_y},
                    Region{mid_x, 0u, width - mid_x, mid_y},
                    Region{0u, mid_y, mid_x, height - mid_y},
                    Region{mid_x, mid_y, width - mid_x, height - mid_y}
                };
            } else {
                expected_regions = {Region{0u, 0u, width, height}};
            }

            std::vector<uint8_t> coverage(reconstructed.size(), 0u);
            std::size_t cursor = 40u;
            for (uint32_t region_index = 0; region_index < region_count; ++region_index) {
                if (cursor + 12u > encoded.size()) {
                    error = "Truncated region descriptor";
                    return false;
                }
                const Region region{
                    read_le_u16(encoded, cursor),
                    read_le_u16(encoded, cursor + 2u),
                    read_le_u16(encoded, cursor + 4u),
                    read_le_u16(encoded, cursor + 6u)
                };
                cursor += 8u;
                const Region& expected = expected_regions[region_index];
                if (region.x != expected.x || region.y != expected.y ||
                    region.width != expected.width || region.height != expected.height) {
                    error = "Region metadata does not match the declared partition";
                    return false;
                }
                const uint32_t leaf_count = read_le_u32(encoded, cursor);
                cursor += 4u;
                if (leaf_count > (encoded.size() - cursor) / 10u) {
                    error = "Truncated leaf records in region";
                    return false;
                }
                if ((region.width == 0u || region.height == 0u) && leaf_count != 0u) {
                    error = "Empty region contains leaf data";
                    return false;
                }

                for (uint32_t i = 0; i < leaf_count; ++i) {
                    const uint16_t x = read_le_u16(encoded, cursor);
                    const uint16_t y = read_le_u16(encoded, cursor + 2u);
                    const uint16_t size = read_le_u16(encoded, cursor + 4u);
                    const uint8_t flags = encoded[cursor + 6u];
                    const uint8_t r_raw = encoded[cursor + 7u];
                    const uint8_t g_raw = encoded[cursor + 8u];
                    const uint8_t b_raw = encoded[cursor + 9u];
                    cursor += 10u;
                    if (size == 0u || size > kMaxImageDimension || (size & (size - 1u)) != 0u ||
                        (flags & 0xFEu) != 0u ||
                        x < region.x || y < region.y ||
                        x >= region.x + region.width || y >= region.y + region.height ||
                        ((flags & 1u) != 0u && !use_prediction)) {
                        error = "Invalid leaf metadata";
                        return false;
                    }

                    Pixel color{r_raw, g_raw, b_raw};
                    if ((flags & 1u) != 0u) {
                        const Pixel prev_avg = compute_region_stats(*previous_frame, region, x, y, size).avg;
                        color = reconstruct_from_residual(prev_avg, ResidualColor{
                            static_cast<int8_t>(r_raw),
                            static_cast<int8_t>(g_raw),
                            static_cast<int8_t>(b_raw)
                        });
                    }
                    const uint32_t x_end = std::min({static_cast<uint32_t>(x) + size, region.x + region.width, width});
                    const uint32_t y_end = std::min({static_cast<uint32_t>(y) + size, region.y + region.height, height});
                    for (uint32_t yy = y; yy < y_end; ++yy) {
                        for (uint32_t xx = x; xx < x_end; ++xx) {
                            const std::size_t index = static_cast<std::size_t>(yy) * width + xx;
                            if (coverage[index] != 0u) {
                                error = "Quadtree leaves overlap";
                                return false;
                            }
                            coverage[index] = 1u;
                            reconstructed.set_pixel(xx, yy, color);
                        }
                    }
                }
            }
            if (cursor != encoded.size()) {
                error = "Trailing bytes remain after decoding all regions";
                return false;
            }
            if (std::find(coverage.begin(), coverage.end(), 0u) != coverage.end()) {
                error = "Quadtree regions do not cover every frame pixel";
                return false;
            }
        } else {
            if (version == 2u && encoded.size() < 42u) {
                error = "Truncated version 2 frame header";
                return false;
            }
            const bool use_prediction = version == 2u && read_le_u32(encoded, 34u) != 0u;
            if (use_prediction && (previous_frame == nullptr ||
                previous_frame->width() != width || previous_frame->height() != height)) {
                error = "This frame requires a matching previous reference frame";
                return false;
            }
            const uint32_t leaf_count = read_le_u32(encoded, version == 2u ? 38u : 34u);
            std::size_t cursor = version == 2u ? 42u : 38u;
            const std::size_t record_size = version == 2u ? 10u : 9u;
            if (leaf_count > (encoded.size() - cursor) / record_size) {
                error = "Truncated leaf records in stream";
                return false;
            }
            for (uint32_t i = 0; i < leaf_count; ++i) {
                const uint16_t x = read_le_u16(encoded, cursor);
                const uint16_t y = read_le_u16(encoded, cursor + 2u);
                const uint16_t size = read_le_u16(encoded, cursor + 4u);
                uint8_t flags = 0u;
                uint8_t r = 0u;
                uint8_t g = 0u;
                uint8_t b = 0u;
                if (version == 2u) {
                    flags = encoded[cursor + 6u];
                    r = encoded[cursor + 7u];
                    g = encoded[cursor + 8u];
                    b = encoded[cursor + 9u];
                } else {
                    r = encoded[cursor + 6u];
                    g = encoded[cursor + 7u];
                    b = encoded[cursor + 8u];
                }
                cursor += record_size;
                if (size == 0u || x >= width || y >= height) {
                    error = "Invalid block coordinates or size during decode";
                    return false;
                }
                Pixel color{r, g, b};
                if (use_prediction && (flags & 1u) != 0u) {
                    const Pixel prev_avg = compute_block_stats(*previous_frame, x, y, size).avg;
                    color = reconstruct_from_residual(prev_avg, ResidualColor{
                        static_cast<int8_t>(r), static_cast<int8_t>(g), static_cast<int8_t>(b)
                    });
                }
                const uint32_t x_end = std::min<uint32_t>(static_cast<uint32_t>(x) + size, width);
                const uint32_t y_end = std::min<uint32_t>(static_cast<uint32_t>(y) + size, height);
                for (uint32_t yy = y; yy < y_end; ++yy) {
                    for (uint32_t xx = x; xx < x_end; ++xx) {
                        reconstructed.set_pixel(xx, yy, color);
                    }
                }
            }
            if (cursor != encoded.size()) {
                error = "Trailing bytes remain after decoding all leaves";
                return false;
            }
        }

        image = std::move(reconstructed);
        return true;
    } catch (const std::exception& ex) {
        error = ex.what();
        return false;
    }
}

}  // namespace mp48mb
