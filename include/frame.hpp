#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mp48mb {

struct Pixel {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

class Image {
public:
    Image() = default;
    Image(uint32_t width, uint32_t height, const Pixel& fill = Pixel{0, 0, 0})
        : width_(width), height_(height), pixels_(width * height, fill) {
    }

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    std::size_t size() const { return pixels_.size(); }

    const Pixel& at(uint32_t x, uint32_t y) const {
        if (x >= width_ || y >= height_) {
            throw std::out_of_range("Image pixel access out of bounds");
        }
        return pixels_[y * width_ + x];
    }

    Pixel& at(uint32_t x, uint32_t y) {
        if (x >= width_ || y >= height_) {
            throw std::out_of_range("Image pixel access out of bounds");
        }
        return pixels_[y * width_ + x];
    }

    Pixel at_clamped(uint32_t x, uint32_t y) const {
        if (width_ == 0 || height_ == 0) {
            return Pixel{0, 0, 0};
        }
        uint32_t cx = std::min(x, width_ - 1u);
        uint32_t cy = std::min(y, height_ - 1u);
        return pixels_[cy * width_ + cx];
    }

    void set_pixel(uint32_t x, uint32_t y, const Pixel& pixel) {
        if (x >= width_ || y >= height_) {
            throw std::out_of_range("Image pixel write out of bounds");
        }
        pixels_[y * width_ + x] = pixel;
    }

    void fill(const Pixel& pixel) {
        std::fill(pixels_.begin(), pixels_.end(), pixel);
    }

    void fill_gradient() {
        for (uint32_t y = 0; y < height_; ++y) {
            for (uint32_t x = 0; x < width_; ++x) {
                const uint8_t r = static_cast<uint8_t>((x * 255u) / std::max<uint32_t>(1u, width_ - 1u));
                const uint8_t g = static_cast<uint8_t>((y * 255u) / std::max<uint32_t>(1u, height_ - 1u));
                const uint8_t b = static_cast<uint8_t>((x + y) * 255u / std::max<uint32_t>(1u, width_ + height_ - 2u));
                pixels_[y * width_ + x] = Pixel{r, g, b};
            }
        }
    }

    bool write_ppm(const std::string& path) const {
        std::ofstream out(path, std::ios::binary | std::ios::out);
        if (!out.is_open()) {
            return false;
        }
        out << "P6\n" << width_ << ' ' << height_ << "\n255\n";
        for (const auto& px : pixels_) {
            out.put(static_cast<char>(px.r));
            out.put(static_cast<char>(px.g));
            out.put(static_cast<char>(px.b));
        }
        return true;
    }

    static Image read_ppm(const std::string& path) {
        std::ifstream in(path, std::ios::binary | std::ios::in);
        if (!in.is_open()) {
            throw std::runtime_error("Unable to open PPM image: " + path);
        }

        std::string magic;
        in >> magic;
        if (magic != "P6") {
            throw std::runtime_error("Unsupported PPM format; only P6 is supported");
        }

        std::string line;
        std::getline(in, line);
        while (in.peek() == '#' || in.peek() == ' ' || in.peek() == '\n' || in.peek() == '\r') {
            if (in.peek() == '#') {
                std::getline(in, line);
            } else {
                char c = static_cast<char>(in.get());
                (void)c;
            }
        }

        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t max_value = 0;
        in >> width >> height >> max_value;
        if (width == 0 || height == 0 || max_value == 0 || max_value > 255u) {
            throw std::runtime_error("Invalid PPM dimensions or max value");
        }

        char whitespace = static_cast<char>(in.get());
        (void)whitespace;

        Image image(width, height);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                unsigned char r = 0, g = 0, b = 0;
                in.read(reinterpret_cast<char*>(&r), 1);
                in.read(reinterpret_cast<char*>(&g), 1);
                in.read(reinterpret_cast<char*>(&b), 1);
                image.set_pixel(x, y, Pixel{r, g, b});
            }
        }
        return image;
    }

private:
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    std::vector<Pixel> pixels_;
};

}  // namespace mp48mb
