//
// Created by aichao on 2026/8/6.
// 轻量图片编解码封装：优先使用 stb_image（单头文件，零依赖，JPEG/PNG/BMP），
// 不支持格式回退 OpenCV。解码统一输出 BGR HWC uint8（与 OpenCV cv::imread 语义一致）。
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace modeldeploy::vision {

    // 解码图片为 BGR HWC uint8（与 cv::imread 语义一致）。
    // 成功返回 true，输出到 dst；失败返回 false。
    // 支持格式：jpeg/png/bmp（stb_image）；其余回退 OpenCV。
    bool decode_image_file(const std::string& filename,
                           std::vector<uint8_t>& dst, int& width, int& height, int& channels);

    bool decode_image_memory(const std::vector<uint8_t>& buf,
                             std::vector<uint8_t>& dst, int& width, int& height, int& channels);

    // 编码图片。支持 jpg/png/bmp（stb_image）；其余回退 OpenCV。
    // src 为 BGR HWC uint8（与 cv::imwrite 语义一致）。
    bool encode_image_memory(const uint8_t* src, int width, int height, int channels,
                             const std::string& ext, std::vector<uint8_t>& dst);

    bool encode_image_file(const uint8_t* src, int width, int height, int channels,
                           const std::string& ext, const std::string& filename);

} // namespace modeldeploy::vision
