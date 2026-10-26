#include <chrono>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "codec.hpp"
#include "frame.hpp"

namespace fs = std::filesystem;

namespace {

mp48mb::EncoderConfig parse_config(int argc, char** argv) {
    mp48mb::EncoderConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) {
            continue;
        }
        const std::string value = argv[i + 1];
        if (key == "--max-block") {
            cfg.max_block_size = static_cast<uint32_t>(std::stoul(value));
        } else if (key == "--min-block") {
            cfg.min_block_size = static_cast<uint32_t>(std::stoul(value));
        } else if (key == "--max-depth") {
            cfg.max_depth = static_cast<uint32_t>(std::stoul(value));
        } else if (key == "--threshold") {
            cfg.error_threshold = static_cast<uint32_t>(std::stoul(value));
        }
    }
    return cfg;
}

const char* find_argument_value(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == key && i + 1 < argc) {
            return argv[i + 1];
        }
    }
    return nullptr;
}

mp48mb::PartitionMode parse_partition_mode(int argc, char** argv) {
    const char* value = find_argument_value(argc, argv, "--regions");
    if (value == nullptr || std::string(value) == "auto") {
        return mp48mb::PartitionMode::Auto;
    }
    if (std::string(value) == "single") {
        return mp48mb::PartitionMode::Single;
    }
    if (std::string(value) == "four") {
        return mp48mb::PartitionMode::Four;
    }
    throw std::invalid_argument("--regions must be auto, single, or four");
}

void print_usage(const char* argv0) {
    std::cout << "Usage:\n"
              << "  " << argv0 << " encode <input.ppm> <output.mpf> [--max-block 256 --min-block 8 --max-depth 6 --threshold 24] [--regions auto|single|four] [--prev previous.ppm]\n"
              << "  " << argv0 << " decode <input.mpf> <output.ppm> [--prev previous.ppm]\n"
              << "  " << argv0 << " encode-block8 <input.ppm> <output.mpb565>\n"
              << "  " << argv0 << " decode-block8 <input.mpb565> <output.ppm>\n"
              << "  " << argv0 << " serve (persistent binary frame backend)\n"
              << "  " << argv0 << " benchmark [width height]\n"
              << "  " << argv0 << " benchmark-block8 [width height]\n"
              << "  " << argv0 << " test\n";
}

mp48mb::Image make_test_image(uint32_t width, uint32_t height) {
    mp48mb::Image source(width, height);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const uint8_t r = static_cast<uint8_t>((x * 255u) / std::max<uint32_t>(1u, width - 1u));
            const uint8_t g = static_cast<uint8_t>((y * 255u) / std::max<uint32_t>(1u, height - 1u));
            const uint8_t b = static_cast<uint8_t>(((x + y) * 255u) / std::max<uint32_t>(1u, width + height - 2u));
            source.set_pixel(x, y, mp48mb::Pixel{r, g, b});
        }
    }
    return source;
}

struct BenchmarkResult {
    std::string name;
    std::size_t encoded_bytes = 0;
    double encode_ms = 0.0;
    double decode_ms = 0.0;
    double mse = 0.0;
    double psnr = 0.0;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}

double image_mse(const mp48mb::Image& source, const mp48mb::Image& reconstructed) {
    long double squared_error = 0.0;
    for (uint32_t y = 0; y < source.height(); ++y) {
        for (uint32_t x = 0; x < source.width(); ++x) {
            const auto a = source.at(x, y);
            const auto b = reconstructed.at(x, y);
            const int dr = static_cast<int>(a.r) - static_cast<int>(b.r);
            const int dg = static_cast<int>(a.g) - static_cast<int>(b.g);
            const int db = static_cast<int>(a.b) - static_cast<int>(b.b);
            squared_error += static_cast<long double>(dr * dr + dg * dg + db * db);
        }
    }
    return static_cast<double>(squared_error / (3.0L * source.size()));
}

bool read_exact(std::istream& input, char* data, std::size_t size) {
    input.read(data, static_cast<std::streamsize>(size));
    return input.gcount() == static_cast<std::streamsize>(size);
}

uint32_t decode_u32(const std::array<uint8_t, 4>& bytes) {
    return static_cast<uint32_t>(bytes[0])
        | (static_cast<uint32_t>(bytes[1]) << 8u)
        | (static_cast<uint32_t>(bytes[2]) << 16u)
        | (static_cast<uint32_t>(bytes[3]) << 24u);
}

void write_u32(std::ostream& output, uint32_t value) {
    const std::array<uint8_t, 4> bytes{
        static_cast<uint8_t>(value),
        static_cast<uint8_t>(value >> 8u),
        static_cast<uint8_t>(value >> 16u),
        static_cast<uint8_t>(value >> 24u),
    };
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void write_u64(std::ostream& output, uint64_t value) {
    std::array<uint8_t, 8> bytes{};
    for (unsigned int index = 0; index < 8u; ++index) {
        bytes[index] = static_cast<uint8_t>(value >> (index * 8u));
    }
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void write_server_error(const std::string& message) {
    std::cout.put(static_cast<char>(1));
    write_u32(std::cout, static_cast<uint32_t>(message.size()));
    std::cout.write(message.data(), static_cast<std::streamsize>(message.size()));
    std::cout.flush();
}

int run_server() {
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 || _setmode(_fileno(stdout), _O_BINARY) == -1) {
        throw std::runtime_error("Unable to switch backend protocol streams to binary mode");
    }
#endif

    mp48mb::FrameCodec codec;
    while (true) {
        std::array<uint8_t, 28> request_header{};
        std::cin.read(reinterpret_cast<char*>(request_header.data()), static_cast<std::streamsize>(request_header.size()));
        const std::streamsize header_bytes = std::cin.gcount();
        if (header_bytes == 0 && std::cin.eof()) {
            return 0;
        }
        if (header_bytes != static_cast<std::streamsize>(request_header.size())) {
            throw std::runtime_error("Truncated frame request header");
        }

        try {
            std::array<uint32_t, 7> fields{};
            for (std::size_t index = 0; index < fields.size(); ++index) {
                std::array<uint8_t, 4> bytes{};
                std::copy_n(request_header.begin() + static_cast<std::ptrdiff_t>(index * 4u), 4, bytes.begin());
                fields[index] = decode_u32(bytes);
            }
            const uint32_t width = fields[0];
            const uint32_t height = fields[1];
            const uint64_t pixel_count = static_cast<uint64_t>(width) * height;
            if (width == 0u || height == 0u || width > 4096u || height > 4096u
                || pixel_count > 16'800'000u) {
                throw std::invalid_argument("Frame dimensions exceed the backend resource limit");
            }
            if (fields[6] > 2u) {
                throw std::invalid_argument("Invalid partition mode in frame request");
            }

            std::vector<uint8_t> rgb(static_cast<std::size_t>(pixel_count) * 3u);
            if (!read_exact(std::cin, reinterpret_cast<char*>(rgb.data()), rgb.size())) {
                throw std::runtime_error("Truncated RGB frame data");
            }
            mp48mb::Image image(width, height);
            for (uint32_t y = 0; y < height; ++y) {
                for (uint32_t x = 0; x < width; ++x) {
                    const std::size_t offset = (static_cast<std::size_t>(y) * width + x) * 3u;
                    image.set_pixel(x, y, mp48mb::Pixel{rgb[offset], rgb[offset + 1u], rgb[offset + 2u]});
                }
            }

            mp48mb::EncoderConfig config;
            config.max_block_size = fields[2];
            config.min_block_size = fields[3];
            config.max_depth = fields[4];
            config.error_threshold = fields[5];
            const auto partition_mode = static_cast<mp48mb::PartitionMode>(fields[6]);

            const auto encode_start = std::chrono::steady_clock::now();
            const std::vector<uint8_t> encoded = codec.encode(image, config, nullptr, partition_mode);
            const auto encode_end = std::chrono::steady_clock::now();
            mp48mb::Image reconstructed;
            std::string error;
            const auto decode_start = std::chrono::steady_clock::now();
            if (!codec.decode(encoded, reconstructed, error)) {
                throw std::runtime_error("Frame decode failed: " + error);
            }
            const auto decode_end = std::chrono::steady_clock::now();

            const auto encode_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(encode_end - encode_start).count();
            const auto decode_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(decode_end - decode_start).count();
            std::cout.put(static_cast<char>(0));
            write_u64(std::cout, static_cast<uint64_t>(encode_ns));
            write_u64(std::cout, static_cast<uint64_t>(decode_ns));
            write_u32(std::cout, static_cast<uint32_t>(encoded.size()));
            std::cout.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
            for (uint32_t y = 0; y < height; ++y) {
                for (uint32_t x = 0; x < width; ++x) {
                    const auto pixel = reconstructed.at(x, y);
                    std::cout.put(static_cast<char>(pixel.r));
                    std::cout.put(static_cast<char>(pixel.g));
                    std::cout.put(static_cast<char>(pixel.b));
                }
            }
            std::cout.flush();
            if (!std::cout) {
                return 1;
            }
        } catch (const std::exception& error) {
            write_server_error(error.what());
            if (!std::cout) {
                return 1;
            }
        }
    }
}

BenchmarkResult benchmark_mode(
    const mp48mb::Image& source,
    mp48mb::PartitionMode mode,
    const std::string& name) {
    mp48mb::EncoderConfig cfg;
    cfg.max_block_size = 256u;
    cfg.min_block_size = 8u;
    cfg.max_depth = 12u;
    cfg.error_threshold = 24u;
    cfg.use_simd = false;
    cfg.stats_strategy = mp48mb::BlockStatsStrategy::Scalar;
    mp48mb::FrameCodec codec;

    auto warmup = codec.encode(source, cfg, nullptr, mode);
    mp48mb::Image warmup_decoded;
    std::string warmup_error;
    if (!codec.decode(warmup, warmup_decoded, warmup_error)) {
        throw std::runtime_error("Benchmark warm-up decode failed: " + warmup_error);
    }

    std::vector<double> encode_times;
    std::vector<double> decode_times;
    std::vector<uint8_t> final_stream;
    mp48mb::Image final_decoded;
    for (int trial = 0; trial < 5; ++trial) {
        const auto encode_start = std::chrono::steady_clock::now();
        final_stream = codec.encode(source, cfg, nullptr, mode);
        const auto encode_end = std::chrono::steady_clock::now();
        std::string error;
        const auto decode_start = std::chrono::steady_clock::now();
        if (!codec.decode(final_stream, final_decoded, error)) {
            throw std::runtime_error("Benchmark decode failed: " + error);
        }
        const auto decode_end = std::chrono::steady_clock::now();
        encode_times.push_back(std::chrono::duration<double, std::milli>(encode_end - encode_start).count());
        decode_times.push_back(std::chrono::duration<double, std::milli>(decode_end - decode_start).count());
    }

    BenchmarkResult result;
    result.name = name;
    result.encoded_bytes = final_stream.size();
    result.encode_ms = median(encode_times);
    result.decode_ms = median(decode_times);
    result.mse = image_mse(source, final_decoded);
    result.psnr = result.mse == 0.0 ? std::numeric_limits<double>::infinity()
        : 10.0 * std::log10(255.0 * 255.0 / result.mse);
    return result;
}

void benchmark_frame(uint32_t width, uint32_t height) {
    if (width == 0u || height == 0u || width > 4096u || height > 4096u) {
        throw std::invalid_argument("Benchmark dimensions must be between 1 and 4096");
    }
    const mp48mb::Image source = make_test_image(width, height);
    const BenchmarkResult single = benchmark_mode(source, mp48mb::PartitionMode::Single, "single");
    const BenchmarkResult four = benchmark_mode(source, mp48mb::PartitionMode::Four, "four-sequential");
    std::cout << "Benchmark frame size: " << width << "x" << height << "\n";
    const char* cpu = std::getenv("PROCESSOR_IDENTIFIER");
    std::cout << "CPU: " << (cpu != nullptr ? cpu : "not reported by environment") << "\n";
    std::cout << "Compiler: " << __VERSION__ << " | flags: -std=c++20 -O3 -Wall -Wextra -pedantic\n";
    std::cout << "Config: max block 256, min block 8, max depth 12, MSE threshold 24\n";
    std::cout << "Trials: 5 after one warm-up; median values; scalar block statistics\n";
    std::cout << "Mode, bytes, encode ms, decode ms, encode FPS, MSE, PSNR dB\n";
    for (const BenchmarkResult* result : {&single, &four}) {
        std::cout << result->name << ", " << result->encoded_bytes << ", "
                  << std::fixed << std::setprecision(3) << result->encode_ms << ", "
                  << result->decode_ms << ", "
                  << (1000.0 / result->encode_ms) << ", "
                  << result->mse << ", " << result->psnr << "\n";
    }
}

BenchmarkResult benchmark_fixed_block(const mp48mb::Image& source) {
    std::vector<uint8_t> warmup = mp48mb::FixedBlockRgb565Codec::encode(source);
    mp48mb::Image warmup_decoded;
    std::string error;
    if (!mp48mb::FixedBlockRgb565Codec::decode(warmup, warmup_decoded, error)) {
        throw std::runtime_error("Fixed-block benchmark warm-up decode failed: " + error);
    }

    std::vector<double> encode_times;
    std::vector<double> decode_times;
    std::vector<uint8_t> final_stream;
    mp48mb::Image final_decoded;
    for (int trial = 0; trial < 5; ++trial) {
        const auto encode_start = std::chrono::steady_clock::now();
        final_stream = mp48mb::FixedBlockRgb565Codec::encode(source);
        const auto encode_end = std::chrono::steady_clock::now();
        const auto decode_start = std::chrono::steady_clock::now();
        if (!mp48mb::FixedBlockRgb565Codec::decode(final_stream, final_decoded, error)) {
            throw std::runtime_error("Fixed-block benchmark decode failed: " + error);
        }
        const auto decode_end = std::chrono::steady_clock::now();
        encode_times.push_back(std::chrono::duration<double, std::milli>(encode_end - encode_start).count());
        decode_times.push_back(std::chrono::duration<double, std::milli>(decode_end - decode_start).count());
    }

    BenchmarkResult result;
    result.name = "rgb565-8x8";
    result.encoded_bytes = final_stream.size();
    result.encode_ms = median(encode_times);
    result.decode_ms = median(decode_times);
    result.mse = image_mse(source, final_decoded);
    result.psnr = result.mse == 0.0 ? std::numeric_limits<double>::infinity()
        : 10.0 * std::log10(255.0 * 255.0 / result.mse);
    return result;
}

void benchmark_fixed_block_frame(uint32_t width, uint32_t height) {
    if (width == 0u || height == 0u || width > 4096u || height > 4096u
        || static_cast<uint64_t>(width) * height > 16'800'000u) {
        throw std::invalid_argument("Benchmark dimensions must be within the 16.8-megapixel codec limit");
    }
    const mp48mb::Image source = make_test_image(width, height);
    const BenchmarkResult single = benchmark_mode(source, mp48mb::PartitionMode::Single, "quadtree-single");
    const BenchmarkResult four = benchmark_mode(source, mp48mb::PartitionMode::Four, "quadtree-four");
    const BenchmarkResult fixed = benchmark_fixed_block(source);
    const uint64_t block_count = static_cast<uint64_t>((width + 7u) / 8u) * ((height + 7u) / 8u);
    const uint64_t fixed_payload_bytes = block_count * 2u;

    std::cout << "Experimental 8x8 RGB565 benchmark: " << width << "x" << height << "\n";
    std::cout << "Trials: 5 after one warm-up; median timings; synthetic gradient; codec/file I/O excluded\n";
    std::cout << "Mode, payload bytes, total file bytes, encode ms, decode ms, encode FPS, MSE, PSNR dB\n";
    for (const BenchmarkResult* result : {&single, &four}) {
        std::cout << result->name << ", " << result->encoded_bytes << ", " << result->encoded_bytes << ", "
                  << std::fixed << std::setprecision(3) << result->encode_ms << ", "
                  << result->decode_ms << ", " << (1000.0 / result->encode_ms) << ", "
                  << result->mse << ", " << result->psnr << "\n";
    }
    std::cout << fixed.name << ", " << fixed_payload_bytes << ", " << fixed.encoded_bytes << ", "
              << std::fixed << std::setprecision(3) << fixed.encode_ms << ", "
              << fixed.decode_ms << ", " << (1000.0 / fixed.encode_ms) << ", "
              << fixed.mse << ", " << fixed.psnr << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 1;
        }

        const std::string command = argv[1];

        if (command == "serve") {
            return run_server();
        }

        if (command == "test") {
            std::cout << "The test target is built as build/test_quadtree; call make test.\n";
            return 0;
        }

        if (command == "benchmark") {
            const uint32_t width = argc >= 4 ? static_cast<uint32_t>(std::stoul(argv[2])) : 640u;
            const uint32_t height = argc >= 4 ? static_cast<uint32_t>(std::stoul(argv[3])) : 480u;
            benchmark_frame(width, height);
            return 0;
        }

        if (command == "benchmark-block8") {
            const uint32_t width = argc >= 4 ? static_cast<uint32_t>(std::stoul(argv[2])) : 1920u;
            const uint32_t height = argc >= 4 ? static_cast<uint32_t>(std::stoul(argv[3])) : 1080u;
            benchmark_fixed_block_frame(width, height);
            return 0;
        }

        if (command == "encode-block8") {
            if (argc < 4) {
                print_usage(argv[0]);
                return 1;
            }
            const mp48mb::Image image = mp48mb::Image::read_ppm(argv[2]);
            const std::vector<uint8_t> encoded = mp48mb::FixedBlockRgb565Codec::encode(image);
            std::ofstream output(argv[3], std::ios::binary);
            if (!output) {
                throw std::runtime_error(std::string("Unable to open output file: ") + argv[3]);
            }
            output.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
            if (!output) {
                throw std::runtime_error(std::string("Failed while writing output file: ") + argv[3]);
            }
            std::cout << "Encoded " << image.width() << "x" << image.height() << " frame with experimental 8x8 RGB565 blocks to "
                      << argv[3] << " (" << encoded.size() << " bytes including header)\n";
            return 0;
        }

        if (command == "decode-block8") {
            if (argc < 4) {
                print_usage(argv[0]);
                return 1;
            }
            std::ifstream input(argv[2], std::ios::binary);
            if (!input) {
                throw std::runtime_error(std::string("Unable to open input file: ") + argv[2]);
            }
            const std::vector<uint8_t> encoded(
                (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            mp48mb::Image decoded;
            std::string error;
            if (!mp48mb::FixedBlockRgb565Codec::decode(encoded, decoded, error)) {
                throw std::runtime_error("Failed fixed-block decode: " + error);
            }
            if (!decoded.write_ppm(argv[3])) {
                throw std::runtime_error(std::string("Failed while writing PPM output: ") + argv[3]);
            }
            std::cout << "Decoded experimental RGB565 frame to " << argv[3] << "\n";
            return 0;
        }

        if (command == "encode") {
            if (argc < 4) {
                print_usage(argv[0]);
                return 1;
            }
            const std::string input_path = argv[2];
            const std::string output_path = argv[3];
            mp48mb::Image image = mp48mb::Image::read_ppm(input_path);
            const auto config = parse_config(argc, argv);
            const auto partition_mode = parse_partition_mode(argc, argv);
            const char* prev_path = find_argument_value(argc, argv, "--prev");
            mp48mb::FrameCodec codec;
            std::vector<uint8_t> encoded;
            if (prev_path != nullptr) {
                const mp48mb::Image previous = mp48mb::Image::read_ppm(prev_path);
                encoded = codec.encode(image, config, &previous, partition_mode);
            } else {
                encoded = codec.encode(image, config, nullptr, partition_mode);
            }
            std::ofstream out(output_path, std::ios::binary);
            if (!out) {
                throw std::runtime_error("Unable to open output file: " + output_path);
            }
            out.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
            if (!out) {
                throw std::runtime_error("Failed while writing output file: " + output_path);
            }
            std::cout << "Encoded " << image.width() << "x" << image.height() << " frame to " << output_path << " (" << encoded.size() << " bytes)\n";
            return 0;
        }

        if (command == "decode") {
            if (argc < 4) {
                print_usage(argv[0]);
                return 1;
            }
            const std::string input_path = argv[2];
            const std::string output_path = argv[3];
            const char* prev_path = find_argument_value(argc, argv, "--prev");
            std::ifstream in(input_path, std::ios::binary);
            std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            mp48mb::Image decoded;
            std::string error;
            mp48mb::FrameCodec codec;
            if (prev_path != nullptr) {
                const mp48mb::Image previous = mp48mb::Image::read_ppm(prev_path);
                if (!codec.decode(bytes, decoded, error, &previous)) {
                    throw std::runtime_error("Failed decode: " + error);
                }
            } else if (!codec.decode(bytes, decoded, error)) {
                throw std::runtime_error("Failed decode: " + error);
            }
            decoded.write_ppm(output_path);
            std::cout << "Decoded output written to " << output_path << "\n";
            return 0;
        }

        print_usage(argv[0]);
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 2;
    }
}
