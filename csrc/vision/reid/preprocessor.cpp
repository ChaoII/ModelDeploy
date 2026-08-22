//
// Created for standalone pedestrian Re-ID (OSNet) preprocessing.
//

#include <cstring>

#include <opencv2/imgproc.hpp>

#include "core/md_log.h"
#include "vision/reid/preprocessor.h"

namespace modeldeploy::vision::reid {
    bool ReIDPreprocessor::run(const std::vector<ImageData>& images,
                               std::vector<Tensor>* output_tensors) {
        if (images.empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0."
                << std::endl;
            return false;
        }

        const int H = height_;  // 256
        const int W = width_;   // 128
        const int C = 3;
        const int channel_step = H * W;
        const int row_bytes = W * C;
        const size_t n = images.size();

        std::vector<float> input_data(n * static_cast<size_t>(C) * H * W);

        for (size_t b = 0; b < n; ++b) {
            cv::Mat src;
            if (!images[b].asMat(&src) || src.empty()) {
                MD_LOG_ERROR << "Failed to read input image as cv::Mat." << std::endl;
                return false;
            }
            if (src.channels() == 1) {
                cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
            } else if (src.channels() == 4) {
                cv::cvtColor(src, src, cv::COLOR_BGRA2BGR);
            }

            cv::Mat resized;
            cv::resize(src, resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);

            // OSNet: BGR -> RGB
            cv::Mat rgb;
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

            // ImageNet mean/std（与 classification/preprocessor.cpp 的 alpha/beta 一致）
            constexpr float mean[C] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
            constexpr float std_dev[C] = {0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f};

            const uint8_t* p = rgb.data;
            float* out = input_data.data() + b * static_cast<size_t>(C) * H * W;
            for (int h = 0; h < H; ++h) {
                const uint8_t* row = p + static_cast<size_t>(h) * row_bytes;
                for (int w = 0; w < W; ++w) {
                    const uint8_t* px = row + static_cast<size_t>(w) * C;
                    for (int c = 0; c < C; ++c) {
                        out[static_cast<size_t>(c) * channel_step +
                            static_cast<size_t>(h) * W + w] =
                            (static_cast<float>(px[c]) - mean[c]) / std_dev[c];
                    }
                }
            }
        }

        output_tensors->resize(1);
        (*output_tensors)[0] =
            Tensor({static_cast<int64_t>(n), C, H, W}, DataType::FP32, Device::CPU);
        std::memcpy((*output_tensors)[0].data(), input_data.data(),
                    input_data.size() * sizeof(float));
        return true;
    }
} // namespace modeldeploy::vision::reid
