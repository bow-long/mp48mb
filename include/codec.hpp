#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "frame.hpp"
#include "quadtree_stack.hpp"

namespace mp48mb {

enum class BlockStatsStrategy {
    Scalar,
    SIMD,
};

enum class PartitionMode {
    Auto,
    Single,
    Four,
};

struct EncoderConfig {
    uint32_t max_block_size = 256u;
    uint32_t min_block_size = 8u;
    uint32_t max_depth = 6u;
    uint32_t error_threshold = 24u;
    uint32_t max_dimension = 4096u;
    bool use_reference_stats = true;
    bool use_simd = true;
    BlockStatsStrategy stats_strategy = BlockStatsStrategy::SIMD;
};

struct LeafNode {
    uint16_t x = 0;
    uint16_t y = 0;
    uint16_t size = 0;
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

class FrameCodec {
public:
    std::vector<uint8_t> encode(const Image& image, const EncoderConfig& config = EncoderConfig{}) const;
    std::vector<uint8_t> encode(const Image& image, const EncoderConfig& config, const Image* previous_frame) const;
    std::vector<uint8_t> encode(const Image& image, const EncoderConfig& config, const Image* previous_frame, PartitionMode partition_mode) const;
    bool decode(const std::vector<uint8_t>& encoded, Image& image, std::string& error) const;
    bool decode(const std::vector<uint8_t>& encoded, Image& image, std::string& error, const Image* previous_frame) const;
    static uint32_t next_power_of_two(uint32_t value);
    static uint8_t log2_block_size(uint32_t value);
};

class FixedBlockRgb565Codec {
public:
    static std::vector<uint8_t> encode(const Image& image, uint32_t block_size = 8u);
    static bool decode(const std::vector<uint8_t>& encoded, Image& image, std::string& error);
};

struct BlockStats {
    uint64_t sum_r = 0;
    uint64_t sum_g = 0;
    uint64_t sum_b = 0;
    uint64_t sum_sq_r = 0;
    uint64_t sum_sq_g = 0;
    uint64_t sum_sq_b = 0;
    uint64_t pixel_count = 0;
    double mse = 0.0;
    Pixel avg = Pixel{0, 0, 0};
};

BlockStats compute_block_stats(const Image& image, uint32_t x, uint32_t y, uint32_t block_size);
BlockStats compute_block_stats(const Image& image, uint32_t x, uint32_t y, uint32_t block_size, BlockStatsStrategy strategy);
BlockStats compute_block_stats_scalar(const Image& image, uint32_t x, uint32_t y, uint32_t block_size);
BlockStats compute_block_stats_simd(const Image& image, uint32_t x, uint32_t y, uint32_t block_size);

}  // namespace mp48mb
