//
// Created by aichao on 2026/8/6.
// 轻量图片编解码实现：stb_image 优先，OpenCV fallback。
//

#include "vision/common/image_codec.h"

// stb_image / stb_image_write 单头文件库的实现单元
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb/stb_image_write.h"

#include <opencv2/opencv.hpp>
#include <cstring>
#include <algorithm>

namespace modeldeploy::vision {

    namespace {

        // 小写扩展名（去点）
        std::string normalize_ext(const std::string& ext) {
            std::string e;
            e.reserve(ext.size());
            for (char c : ext) {
                if (c == '.') continue;
                e.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            return e;
        }

        // stb_image 支持的解码格式
        bool stb_supported_decode(const std::string& ext) {
            const std::string e = normalize_ext(ext);
            return e == "jpg" || e == "jpeg" || e == "png" || e == "bmp";
        }

        // stb_image_write 支持的编码格式
        bool stb_supported_encode(const std::string& ext) {
            const std::string e = normalize_ext(ext);
            return e == "jpg" || e == "jpeg" || e == "png" || e == "bmp";
        }

        // stb 解码（RGB/RGBA/GRAY）→ BGR 或 BGRA（匹配 OpenCV 语义）
        bool decode_via_stb(const unsigned char* data, int len,
                            std::vector<uint8_t>& dst, int& width, int& height, int& channels) {
            int w = 0, h = 0, comp = 0;
            unsigned char* img = stbi_load_from_memory(data, len, &w, &h, &comp, 0);
            if (!img || w <= 0 || h <= 0) {
                if (img) stbi_image_free(img);
                return false;
            }
            width = w;
            height = h;
            // comp: stb 返回实际通道（1=gray, 3=RGB, 4=RGBA）
            // OpenCV 语义：灰度→1 通道 BGR=GRAY；RGB→BGR；RGBA→BGRA
            const size_t plane = static_cast<size_t>(w) * h;
            if (comp == 1) {
                channels = 1;
                dst.resize(plane);
                std::memcpy(dst.data(), img, plane);
            } else if (comp == 3) {
                channels = 3;
                dst.resize(plane * 3);
                // RGB → BGR
                const uint8_t* src = img;
                uint8_t* out = dst.data();
                for (size_t i = 0; i < plane; ++i) {
                    out[i * 3 + 0] = src[i * 3 + 2];
                    out[i * 3 + 1] = src[i * 3 + 1];
                    out[i * 3 + 2] = src[i * 3 + 0];
                }
            } else if (comp == 4) {
                channels = 4;
                dst.resize(plane * 4);
                // RGBA → BGRA
                const uint8_t* src = img;
                uint8_t* out = dst.data();
                for (size_t i = 0; i < plane; ++i) {
                    out[i * 4 + 0] = src[i * 4 + 2];
                    out[i * 4 + 1] = src[i * 4 + 1];
                    out[i * 4 + 2] = src[i * 4 + 0];
                    out[i * 4 + 3] = src[i * 4 + 3];
                }
            } else {
                stbi_image_free(img);
                return false;
            }
            stbi_image_free(img);
            return true;
        }

        // OpenCV fallback 解码（BGR 语义）
        bool decode_via_cv(const unsigned char* data, size_t len,
                           std::vector<uint8_t>& dst, int& width, int& height, int& channels) {
            cv::Mat mat;
            if (data) {
                mat = cv::imdecode(cv::Mat(1, static_cast<int>(len), CV_8UC1,
                                           const_cast<unsigned char*>(data)),
                                   cv::IMREAD_COLOR);
            }
            if (mat.empty()) return false;
            width = mat.cols;
            height = mat.rows;
            channels = mat.channels();
            const size_t bytes = static_cast<size_t>(width) * height * channels;
            dst.resize(bytes);
            std::memcpy(dst.data(), mat.data, bytes);
            return true;
        }

    } // namespace

    bool decode_image_file(const std::string& filename,
                           std::vector<uint8_t>& dst, int& width, int& height, int& channels) {
        // 读取文件字节
        std::vector<uint8_t> buf;
        {
            FILE* f = std::fopen(filename.c_str(), "rb");
            if (!f) return false;
            std::fseek(f, 0, SEEK_END);
            const long sz = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            if (sz <= 0) { std::fclose(f); return false; }
            buf.resize(static_cast<size_t>(sz));
            const size_t rd = std::fread(buf.data(), 1, static_cast<size_t>(sz), f);
            std::fclose(f);
            if (rd != static_cast<size_t>(sz)) return false;
        }
        if (buf.empty()) return false;
        return decode_image_memory(buf, dst, width, height, channels);
    }

    bool decode_image_memory(const std::vector<uint8_t>& buf,
                             std::vector<uint8_t>& dst, int& width, int& height, int& channels) {
        if (buf.empty()) return false;
        // 先试 stb_image（JPEG/PNG/BMP）
        if (decode_via_stb(buf.data(), static_cast<int>(buf.size()),
                           dst, width, height, channels)) {
            return true;
        }
        // fallback OpenCV
        return decode_via_cv(buf.data(), buf.size(), dst, width, height, channels);
    }

    bool encode_image_memory(const uint8_t* src, int width, int height, int channels,
                             const std::string& ext, std::vector<uint8_t>& dst) {
        if (!src || width <= 0 || height <= 0 || (channels != 1 && channels != 3 && channels != 4)) {
            return false;
        }
        const std::string e = normalize_ext(ext);
        // BGR → RGB（stb 需要 RGB）
        std::vector<uint8_t> rgb;
        const uint8_t* encode_src = src;
        if (channels == 3) {
            const size_t plane = static_cast<size_t>(width) * height;
            rgb.resize(plane * 3);
            for (size_t i = 0; i < plane; ++i) {
                rgb[i * 3 + 0] = src[i * 3 + 2];
                rgb[i * 3 + 1] = src[i * 3 + 1];
                rgb[i * 3 + 2] = src[i * 3 + 0];
            }
            encode_src = rgb.data();
        } else if (channels == 4) {
            const size_t plane = static_cast<size_t>(width) * height;
            rgb.resize(plane * 4);
            for (size_t i = 0; i < plane; ++i) {
                rgb[i * 4 + 0] = src[i * 4 + 2];
                rgb[i * 4 + 1] = src[i * 4 + 1];
                rgb[i * 4 + 2] = src[i * 4 + 0];
                rgb[i * 4 + 3] = src[i * 4 + 3];
            }
            encode_src = rgb.data();
        }

        // stb_image_write 内存编码
        bool ok = false;
        int comp = channels;
        if (channels == 1) {
            const int stride = width;
            const int len = stbi_write_png_to_func(
                [](void* ctx, void* data, int size) {
                    auto* v = static_cast<std::vector<uint8_t>*>(ctx);
                    const auto* p = static_cast<const uint8_t*>(data);
                    v->insert(v->end(), p, p + size);
                }, &dst, width, height, comp, encode_src, stride);
            ok = len != 0;
        } else if (e == "png") {
            const int len = stbi_write_png_to_func(
                [](void* ctx, void* data, int size) {
                    auto* v = static_cast<std::vector<uint8_t>*>(ctx);
                    const auto* p = static_cast<const uint8_t*>(data);
                    v->insert(v->end(), p, p + size);
                }, &dst, width, height, comp, encode_src, width * comp);
            ok = len != 0;
        } else if (e == "jpg" || e == "jpeg") {
            const int len = stbi_write_jpg_to_func(
                [](void* ctx, void* data, int size) {
                    auto* v = static_cast<std::vector<uint8_t>*>(ctx);
                    const auto* p = static_cast<const uint8_t*>(data);
                    v->insert(v->end(), p, p + size);
                }, &dst, width, height, comp, encode_src, 90);
            ok = len != 0;
        } else if (e == "bmp") {
            const int len = stbi_write_bmp_to_func(
                [](void* ctx, void* data, int size) {
                    auto* v = static_cast<std::vector<uint8_t>*>(ctx);
                    const auto* p = static_cast<const uint8_t*>(data);
                    v->insert(v->end(), p, p + size);
                }, &dst, width, height, comp, encode_src);
            ok = len != 0;
        }

        if (ok) return true;

        // OpenCV fallback
        dst.clear();
        cv::Mat mat(height, width, channels == 1 ? CV_8UC1 : channels == 3 ? CV_8UC3 : CV_8UC4,
                    const_cast<uint8_t*>(src));
        std::vector<uchar> cvbuf;
        if (cv::imencode(ext, mat, cvbuf)) {
            dst.assign(cvbuf.begin(), cvbuf.end());
            return true;
        }
        return false;
    }

    bool encode_image_file(const uint8_t* src, int width, int height, int channels,
                           const std::string& ext, const std::string& filename) {
        std::vector<uint8_t> buf;
        if (!encode_image_memory(src, width, height, channels, ext, buf)) {
            return false;
        }
        FILE* f = std::fopen(filename.c_str(), "wb");
        if (!f) return false;
        const size_t wr = std::fwrite(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        return wr == buf.size();
    }

} // namespace modeldeploy::vision
