//
// Created for MOT tracking ReID appearance extraction.
//

#include "vision/tracking/reid_extractor.h"
#include "core/md_log.h"
#include "core/tensor.h"
#include "core/enum_variables.h"

#include <cmath>
#include <cstring>
#include <opencv2/opencv.hpp>

namespace modeldeploy::vision::tracking {
    bool ReidExtractor::init(const std::string& onnx, const RuntimeOption& opt) {
        if (onnx.empty()) {
            return false;
        }
        runtime_option = opt;
        runtime_option.set_model_path(onnx);
        if (!init_runtime()) {
            MD_LOG_ERROR << "ReidExtractor: failed to initialize runtime for: " << onnx
                         << std::endl;
            return false;
        }
        initialized_ = true;
        return true;
    }

    bool ReidExtractor::is_initialized() const {
        return BaseModel::is_initialized();
    }

    std::vector<float> ReidExtractor::extract(const ImageData& patch) {
        if (!initialized_ || patch.empty()) {
            return {};
        }

        cv::Mat src;
        if (!patch.asMat(&src) || src.empty()) {
            return {};
        }

        if (src.channels() == 1) {
            cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
        } else if (src.channels() == 4) {
            cv::cvtColor(src, src, cv::COLOR_BGRA2BGR);
        }

        const int H = input_h_;
        const int W = input_w_;
        const int C = 3;

        cv::Mat resized;
        cv::resize(src, resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        constexpr float mean[C] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
        constexpr float std_dev[C] = {0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f};

        std::vector<float> input_data(static_cast<size_t>(C) * H * W);
        const uint8_t* p = rgb.data;
        const int channel_step = H * W;
        const int row_bytes = W * C;
        for (int h = 0; h < H; ++h) {
            const uint8_t* row = p + static_cast<size_t>(h) * row_bytes;
            for (int w = 0; w < W; ++w) {
                const uint8_t* px = row + static_cast<size_t>(w) * C;
                for (int c = 0; c < C; ++c) {
                    input_data[static_cast<size_t>(c) * channel_step + static_cast<size_t>(h) * W + w] =
                        (static_cast<float>(px[c]) - mean[c]) / std_dev[c];
                }
            }
        }

        std::vector<Tensor> input_tensors(1);
        input_tensors[0] = Tensor({1, C, H, W}, DataType::FP32, Device::CPU);
        std::memcpy(input_tensors[0].data(), input_data.data(),
                    input_data.size() * sizeof(float));
        input_tensors[0].set_name(get_input_info(0).name);

        std::vector<Tensor> output_tensors;
        if (!infer(input_tensors, &output_tensors) || output_tensors.empty()) {
            MD_LOG_ERROR << "ReidExtractor: inference failed." << std::endl;
            return {};
        }

        const Tensor& out = output_tensors[0];
        const size_t n = out.size();
        const void* od = out.data();
        if (od == nullptr || n == 0) {
            return {};
        }
        std::vector<float> embedding(n);
        std::memcpy(embedding.data(), od, n * sizeof(float));
        return l2_normalize(embedding);
    }

    std::vector<float> ReidExtractor::l2_normalize(const std::vector<float>& v) const {
        double sum = 0.0;
        for (const float x : v) {
            sum += static_cast<double>(x) * static_cast<double>(x);
        }
        if (sum <= 0.0) {
            return v; // zero norm -> unchanged, avoids NaN
        }
        const float norm = static_cast<float>(std::sqrt(sum));
        std::vector<float> out = v;
        for (float& x : out) {
            x /= norm;
        }
        return out;
    }

    void ReidExtractor::set_input_size(int h, int w) {
        if (h > 0) {
            input_h_ = h;
        }
        if (w > 0) {
            input_w_ = w;
        }
    }
} // namespace modeldeploy::vision::tracking
