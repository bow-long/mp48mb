#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "codec.hpp"
#include "frame.hpp"
#include "quadtree_stack.hpp"

using namespace mp48mb;

int main() {
    // Stack layout validation.
    const QuadStackEntry entry = QuadStackEntry::pack(123u, 456u, 9u, 3u);
    assert(entry.x() == 123u);
    assert(entry.y() == 456u);
    assert(entry.log2_size() == 9u);
    assert(entry.flags() == 3u);
    assert(entry.size() == 512u);
    const QuadStackEntry max_entry = QuadStackEntry::pack(4095u, 4095u, 12u, 15u);
    assert(max_entry.size() == 4096u);
    assert(max_entry.x() == 4095u && max_entry.y() == 4095u && max_entry.flags() == 15u);

    // Graceful overflow detection.
    try {
        QuadStackEntry::pack(8192u, 0u, 4u, 0u);
        assert(false && "Expected packing overflow to throw");
    } catch (const std::invalid_argument&) {
    }

    // Reference encode/decode round trip for a synthetic image.
    Image src(64u, 64u);
    for (uint32_t y = 0; y < 64u; ++y) {
        for (uint32_t x = 0; x < 64u; ++x) {
            const uint8_t r = static_cast<uint8_t>((x * 255u) / 63u);
            const uint8_t g = static_cast<uint8_t>((y * 255u) / 63u);
            const uint8_t b = static_cast<uint8_t>(((x + y) * 255u) / 127u);
            src.set_pixel(x, y, Pixel{r, g, b});
        }
    }

    FrameCodec codec;
    EncoderConfig config;
    config.max_block_size = 64u;
    config.min_block_size = 8u;
    config.max_depth = 6u;
    config.error_threshold = 24u;
    config.use_simd = true;
    config.stats_strategy = BlockStatsStrategy::SIMD;
    const auto encoded = codec.encode(src, config);

    Image decoded(64u, 64u, Pixel{0, 0, 0});
    std::string error;
    const bool ok = codec.decode(encoded, decoded, error);
    assert(ok && error.empty());
    assert(decoded.width() == src.width());
    assert(decoded.height() == src.height());

    for (uint32_t y = 0; y < 64u; ++y) {
        for (uint32_t x = 0; x < 64u; ++x) {
            const Pixel a = src.at(x, y);
            const Pixel bpx = decoded.at(x, y);
            assert(std::abs(static_cast<int>(a.r) - static_cast<int>(bpx.r)) <= 128);
            assert(std::abs(static_cast<int>(a.g) - static_cast<int>(bpx.g)) <= 128);
            assert(std::abs(static_cast<int>(a.b) - static_cast<int>(bpx.b)) <= 128);
        }
    }

    Image solid(17u, 9u, Pixel{100u, 150u, 200u});
    const auto fixed_solid = FixedBlockRgb565Codec::encode(solid);
    assert(fixed_solid.size() == 20u + 3u * 2u * 2u);
    assert(std::string(fixed_solid.begin(), fixed_solid.begin() + 8) == "MP48B565");
    Image fixed_solid_decoded;
    std::string fixed_error;
    assert(FixedBlockRgb565Codec::decode(fixed_solid, fixed_solid_decoded, fixed_error));
    assert(fixed_error.empty());
    assert(fixed_solid_decoded.width() == solid.width() && fixed_solid_decoded.height() == solid.height());
    for (uint32_t y = 0; y < solid.height(); ++y) {
        for (uint32_t x = 0; x < solid.width(); ++x) {
            const Pixel actual = fixed_solid_decoded.at(x, y);
            assert(std::abs(static_cast<int>(actual.r) - 100) <= 4);
            assert(std::abs(static_cast<int>(actual.g) - 150) <= 2);
            assert(std::abs(static_cast<int>(actual.b) - 200) <= 4);
        }
    }

    Image odd_fixed_edges(9u, 9u, Pixel{0u, 0u, 0u});
    for (uint32_t y = 0; y < odd_fixed_edges.height(); ++y) {
        for (uint32_t x = 0; x < odd_fixed_edges.width(); ++x) {
            if (x == 8u || y == 8u) {
                odd_fixed_edges.set_pixel(x, y, Pixel{200u, 100u, 50u});
            }
        }
    }
    const auto odd_fixed_stream = FixedBlockRgb565Codec::encode(odd_fixed_edges);
    Image odd_fixed_decoded;
    std::string odd_fixed_error;
    assert(FixedBlockRgb565Codec::decode(odd_fixed_stream, odd_fixed_decoded, odd_fixed_error));
    for (uint32_t y = 0; y < odd_fixed_edges.height(); ++y) {
        for (uint32_t x = 0; x < odd_fixed_edges.width(); ++x) {
            const Pixel actual = odd_fixed_decoded.at(x, y);
            if (x == 8u || y == 8u) {
                assert(std::abs(static_cast<int>(actual.r) - 200) <= 4);
                assert(std::abs(static_cast<int>(actual.g) - 100) <= 2);
                assert(std::abs(static_cast<int>(actual.b) - 50) <= 4);
            } else {
                assert(actual.r == 0u && actual.g == 0u && actual.b == 0u);
            }
        }
    }

    Image checkerboard(8u, 8u);
    for (uint32_t y = 0; y < 8u; ++y) {
        for (uint32_t x = 0; x < 8u; ++x) {
            const uint8_t value = ((x + y) & 1u) == 0u ? 0u : 255u;
            checkerboard.set_pixel(x, y, Pixel{value, value, value});
        }
    }
    const auto fixed_checkerboard = FixedBlockRgb565Codec::encode(checkerboard);
    Image fixed_checkerboard_decoded;
    std::string checkerboard_error;
    assert(FixedBlockRgb565Codec::decode(fixed_checkerboard, fixed_checkerboard_decoded, checkerboard_error));
    const Pixel checker_pixel = fixed_checkerboard_decoded.at(0u, 0u);
    for (uint32_t y = 0; y < 8u; ++y) {
        for (uint32_t x = 0; x < 8u; ++x) {
            const Pixel actual = fixed_checkerboard_decoded.at(x, y);
            assert(actual.r == checker_pixel.r && actual.g == checker_pixel.g && actual.b == checker_pixel.b);
        }
    }
    assert(checker_pixel.r >= 120u && checker_pixel.r <= 132u);

    const Image fixed_1080p(1920u, 1080u, Pixel{60u, 120u, 180u});
    const auto fixed_1080p_stream = FixedBlockRgb565Codec::encode(fixed_1080p);
    assert(fixed_1080p_stream.size() == 20u + 64'800u);
    Image fixed_1080p_decoded;
    std::string fixed_1080p_error;
    assert(FixedBlockRgb565Codec::decode(fixed_1080p_stream, fixed_1080p_decoded, fixed_1080p_error));
    assert(fixed_1080p_decoded.width() == 1920u && fixed_1080p_decoded.height() == 1080u);

    std::vector<uint8_t> truncated_fixed = fixed_solid;
    truncated_fixed.pop_back();
    Image truncated_fixed_output;
    std::string truncated_fixed_error;
    assert(!FixedBlockRgb565Codec::decode(truncated_fixed, truncated_fixed_output, truncated_fixed_error));
    assert(truncated_fixed_error.find("truncated") != std::string::npos);
    std::vector<uint8_t> trailing_fixed = fixed_solid;
    trailing_fixed.push_back(0u);
    Image trailing_fixed_output;
    std::string trailing_fixed_error;
    assert(!FixedBlockRgb565Codec::decode(trailing_fixed, trailing_fixed_output, trailing_fixed_error));
    std::vector<uint8_t> invalid_fixed = fixed_solid;
    invalid_fixed[8u] = 2u;
    Image invalid_fixed_output;
    std::string invalid_fixed_error;
    assert(!FixedBlockRgb565Codec::decode(invalid_fixed, invalid_fixed_output, invalid_fixed_error));
    bool invalid_block_size_rejected = false;
    try {
        (void)FixedBlockRgb565Codec::encode(solid, 4u);
    } catch (const std::invalid_argument&) {
        invalid_block_size_rejected = true;
    }
    assert(invalid_block_size_rejected);

    // Malformed stream detection.
    std::vector<uint8_t> bogus = { 'B', 'A', 'D', '!' };
    Image bogus_img;
    std::string bogus_error;
    assert(!codec.decode(bogus, bogus_img, bogus_error));

    const BlockStats scalar = compute_block_stats_scalar(src, 10u, 10u, 16u);
    const BlockStats simd = compute_block_stats_simd(src, 10u, 10u, 16u);
    assert(scalar.pixel_count == simd.pixel_count);
    assert(scalar.sum_r == simd.sum_r);
    assert(scalar.sum_g == simd.sum_g);
    assert(scalar.sum_b == simd.sum_b);
    assert(std::abs(scalar.mse - simd.mse) < 1e-6);
    Image high_contrast(32u, 32u);
    for (uint32_t y = 0; y < high_contrast.height(); ++y) {
        for (uint32_t x = 0; x < high_contrast.width(); ++x) {
            const uint8_t high = ((x + y) & 1u) == 0u ? 255u : 0u;
            high_contrast.set_pixel(x, y, Pixel{high, static_cast<uint8_t>(255u - high), high});
        }
    }
    const BlockStats high_scalar = compute_block_stats_scalar(high_contrast, 0u, 0u, 32u);
    const BlockStats high_simd = compute_block_stats_simd(high_contrast, 0u, 0u, 32u);
    assert(high_scalar.sum_sq_r == high_simd.sum_sq_r);
    assert(high_scalar.sum_sq_g == high_simd.sum_sq_g);
    assert(high_scalar.sum_sq_b == high_simd.sum_sq_b);
    assert(std::abs(high_scalar.mse - high_simd.mse) < 1e-6);

    Image previous(64u, 64u, Pixel{10u, 20u, 30u});
    Image current(64u, 64u, Pixel{13u, 22u, 31u});
    const auto encoded_prediction = codec.encode(current, config, &previous);
    Image decoded_prediction(64u, 64u, Pixel{0, 0, 0});
    std::string prediction_error;
    assert(codec.decode(encoded_prediction, decoded_prediction, prediction_error, &previous));
    for (uint32_t y = 0; y < 64u; ++y) {
        for (uint32_t x = 0; x < 64u; ++x) {
            const Pixel ref = current.at(x, y);
            const Pixel out = decoded_prediction.at(x, y);
            assert(ref.r == out.r);
            assert(ref.g == out.g);
            assert(ref.b == out.b);
        }
    }

    EncoderConfig exact_config = config;
    exact_config.min_block_size = 1u;
    exact_config.max_block_size = 64u;
    exact_config.max_depth = 12u;
    exact_config.error_threshold = 0u;
    exact_config.use_simd = false;

    Image odd(65u, 53u);
    for (uint32_t y = 0; y < odd.height(); ++y) {
        for (uint32_t x = 0; x < odd.width(); ++x) {
            const bool right = x >= odd.width() / 2u;
            const bool bottom = y >= odd.height() / 2u;
            odd.set_pixel(x, y, Pixel{
                static_cast<uint8_t>(right ? 200u : 20u),
                static_cast<uint8_t>(bottom ? 180u : 30u),
                static_cast<uint8_t>((right != bottom) ? 220u : 40u)
            });
        }
    }
    const auto four_regions = codec.encode(odd, exact_config, nullptr, PartitionMode::Four);
    assert(four_regions[34u] == 1u);
    assert(four_regions[35u] == 4u);
    Image odd_decoded;
    std::string odd_error;
    assert(codec.decode(four_regions, odd_decoded, odd_error));
    for (uint32_t y = 0; y < odd.height(); ++y) {
        for (uint32_t x = 0; x < odd.width(); ++x) {
            const Pixel expected = odd.at(x, y);
            const Pixel actual = odd_decoded.at(x, y);
            assert(expected.r == actual.r && expected.g == actual.g && expected.b == actual.b);
        }
    }

    std::vector<uint8_t> invalid_partition = four_regions;
    invalid_partition[35u] = 3u;
    Image invalid_partition_output;
    std::string invalid_partition_error;
    assert(!codec.decode(invalid_partition, invalid_partition_output, invalid_partition_error));
    std::vector<uint8_t> invalid_leaf = four_regions;
    invalid_leaf[56u] = 0u;
    invalid_leaf[57u] = 0u;
    Image invalid_leaf_output;
    std::string invalid_leaf_error;
    assert(!codec.decode(invalid_leaf, invalid_leaf_output, invalid_leaf_error));
    std::vector<uint8_t> truncated_four = four_regions;
    truncated_four.pop_back();
    Image truncated_output;
    std::string truncated_error;
    assert(!codec.decode(truncated_four, truncated_output, truncated_error));

    const Image threshold_small(640u, 480u, Pixel{7u, 11u, 13u});
    const auto auto_small = codec.encode(threshold_small, exact_config);
    assert(auto_small[34u] == 0u && auto_small[35u] == 1u);
    Image threshold_small_decoded;
    std::string threshold_small_error;
    assert(codec.decode(auto_small, threshold_small_decoded, threshold_small_error));
    assert(threshold_small_decoded.at(639u, 479u).r == 7u);
    const Image threshold_large(641u, 480u, Pixel{7u, 11u, 13u});
    const auto auto_large = codec.encode(threshold_large, exact_config);
    assert(auto_large[34u] == 1u && auto_large[35u] == 4u);
    Image threshold_large_decoded;
    std::string threshold_large_error;
    assert(codec.decode(auto_large, threshold_large_decoded, threshold_large_error));
    assert(threshold_large_decoded.at(640u, 479u).r == 7u);
    const auto forced_single = codec.encode(threshold_large, exact_config, nullptr, PartitionMode::Single);
    assert(forced_single[34u] == 0u && forced_single[35u] == 1u);
    const Image threshold_tall(640u, 481u, Pixel{7u, 11u, 13u});
    const auto auto_tall = codec.encode(threshold_tall, exact_config);
    assert(auto_tall[34u] == 1u && auto_tall[35u] == 4u);

    const Image max_extent(4096u, 1u, Pixel{51u, 61u, 71u});
    EncoderConfig max_block_config = exact_config;
    max_block_config.max_block_size = 4096u;
    const auto max_block_stream = codec.encode(max_extent, max_block_config, nullptr, PartitionMode::Single);
    Image max_extent_decoded;
    std::string max_extent_error;
    assert(codec.decode(max_block_stream, max_extent_decoded, max_extent_error));
    const Pixel max_extent_pixel = max_extent_decoded.at(4095u, 0u);
    assert(max_extent_pixel.r == 51u && max_extent_pixel.g == 61u && max_extent_pixel.b == 71u);

    const auto write_u32 = [](std::vector<uint8_t>& stream, uint32_t value) {
        for (uint32_t shift = 0; shift < 32u; shift += 8u) {
            stream.push_back(static_cast<uint8_t>((value >> shift) & 0xFFu));
        }
    };
    const auto write_u16 = [](std::vector<uint8_t>& stream, uint16_t value) {
        stream.push_back(static_cast<uint8_t>(value & 0xFFu));
        stream.push_back(static_cast<uint8_t>((value >> 8u) & 0xFFu));
    };
    std::vector<uint8_t> version_one = {0x38u, 0x34u, 0x50u, 0x4Du, 1u, 3u};
    write_u32(version_one, 1u);
    write_u32(version_one, 1u);
    for (int field = 0; field < 5; ++field) write_u32(version_one, 1u);
    write_u32(version_one, 1u);
    write_u16(version_one, 0u);
    write_u16(version_one, 0u);
    write_u16(version_one, 1u);
    version_one.insert(version_one.end(), {17u, 23u, 29u});
    Image legacy_one;
    std::string legacy_one_error;
    assert(codec.decode(version_one, legacy_one, legacy_one_error));
    assert(legacy_one.at(0u, 0u).r == 17u && legacy_one.at(0u, 0u).g == 23u);

    std::vector<uint8_t> version_two = {0x38u, 0x34u, 0x50u, 0x4Du, 2u, 3u};
    write_u32(version_two, 1u);
    write_u32(version_two, 1u);
    for (int field = 0; field < 5; ++field) write_u32(version_two, 1u);
    write_u32(version_two, 0u);
    write_u32(version_two, 1u);
    write_u16(version_two, 0u);
    write_u16(version_two, 0u);
    write_u16(version_two, 1u);
    version_two.insert(version_two.end(), {0u, 31u, 37u, 41u});
    Image legacy_two;
    std::string legacy_two_error;
    assert(codec.decode(version_two, legacy_two, legacy_two_error));
    assert(legacy_two.at(0u, 0u).b == 41u);

    std::cout << "Quadtree stack and codec tests passed.\n";
    return 0;
}
