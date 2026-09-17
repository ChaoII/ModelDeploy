//
// MediaPipe 手部关键点（21 点）独立模型封装实现。
//
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/md_log.h"
#include "vision/hand/hand_landmark.h"

namespace modeldeploy::vision::hand {
    HandLandmark::HandLandmark(const std::string& model_file, const RuntimeOption& option) {
        runtime_option = option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool HandLandmark::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize hand landmark runtime." << std::endl;
            return false;
        }
        return true;
    }

    bool HandLandmark::predict(const ImageData& img,
                               std::vector<KeyPointsResult>* results,
                               TimerArray* timer) {
        if (!results || !initialized_) return false;

        // 输入布局：MediaPipe handpose 为 NHWC [1,224,224,3]；同时兼容 NCHW 以便移植。
        const TensorInfo info = get_input_info(0);
        const std::vector<int>& shp = info.shape;
        if (shp.size() != 4) {
            MD_LOG_ERROR << "HandLandmark expects a 4-D input tensor." << std::endl;
            return false;
        }
        const bool nhwc = !(shp[1] == 3 && shp[3] != 3);
        const int H = nhwc ? shp[1] : shp[2];
        const int W = nhwc ? shp[2] : shp[3];
        const int C = 3;
        if (H <= 0 || W <= 0 || shp[nhwc ? 3 : 1] != C) {
            MD_LOG_ERROR << "HandLandmark unexpected input shape." << std::endl;
            return false;
        }

        if (timer) timer->pre_timer.start();
        cv::Mat src;
        if (!img.asMat(&src) || src.empty()) return false;
        if (src.channels() == 1) {
            cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
        } else if (src.channels() == 4) {
            cv::cvtColor(src, src, cv::COLOR_BGRA2BGR);
        }
        cv::Mat resized;
        cv::resize(src, resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        cv::Mat f32;
        rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);   // MediaPipe 输入：RGB [0,1]

        std::vector<int64_t> tshape = nhwc
            ? std::vector<int64_t>{1, H, W, C}
            : std::vector<int64_t>{1, C, H, W};
        Tensor input(tshape, DataType::FP32, Device::CPU);
        float* dst = static_cast<float*>(input.data());
        if (nhwc) {
            std::memcpy(dst, f32.data, static_cast<size_t>(H) * W * C * sizeof(float));
        } else {
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        dst[c * H * W + h * W + w] = f32.at<cv::Vec3f>(h, w)[c];
        }
        input.set_name(info.name);
        reused_input_tensors_.clear();
        reused_input_tensors_.push_back(std::move(input));
        if (timer) timer->pre_timer.stop();

        if (timer) timer->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "HandLandmark inference failed." << std::endl;
            return false;
        }
        if (timer) timer->infer_timer.stop();

        if (timer) timer->post_timer.start();
        // 输出识别：优先按图名（Identity=landmarks、Identity_1=handflag、Identity_3=world），
        // 名称缺失时按元素数回退（63=landmarks/world 各一，1=handflag/handedness 各一）。
        const Tensor* lm = nullptr;
        const Tensor* flag = nullptr;
        for (const auto& t : reused_output_tensors_) {
            const std::string& n = t.get_name();
            if (n == "Identity") lm = &t;
            else if (n == "Identity_1") flag = &t;
        }
        if (!lm || !flag) {
            lm = nullptr;
            flag = nullptr;
            for (const auto& t : reused_output_tensors_) {
                if (t.size() == 63) {
                    if (!lm) lm = &t;
                } else if (t.size() == 1 && !flag) {
                    flag = &t;
                }
            }
        }
        if (!lm || !flag || lm->size() < 63) {
            MD_LOG_ERROR << "HandLandmark missing landmark/handflag outputs." << std::endl;
            return false;
        }

        const float score = static_cast<const float*>(flag->data())[0];
        const float* p = static_cast<const float*>(lm->data());
        const float inv_w = static_cast<float>(img.width()) / static_cast<float>(W);
        const float inv_h = static_cast<float>(img.height()) / static_cast<float>(H);
        const float max_x = static_cast<float>(img.width());
        const float max_y = static_cast<float>(img.height());

        KeyPointsResult r;
        r.label_id = 0;
        r.score = score;
        float x0 = std::numeric_limits<float>::infinity();
        float y0 = std::numeric_limits<float>::infinity();
        float x1 = -std::numeric_limits<float>::infinity();
        float y1 = -std::numeric_limits<float>::infinity();
        r.keypoints.reserve(21);
        for (int i = 0; i < 21; ++i) {
            // 模型输出为 224 输入空间的像素坐标：线性回映射到传入图像像素坐标并钳制。
            const float x = std::clamp(p[i * 3 + 0] * inv_w, 0.f, max_x);
            const float y = std::clamp(p[i * 3 + 1] * inv_h, 0.f, max_y);
            const float z = p[i * 3 + 2];
            r.keypoints.emplace_back(x, y, z);
            x0 = std::min(x0, x); y0 = std::min(y0, y);
            x1 = std::max(x1, x); y1 = std::max(y1, y);
        }
        r.box = Rect2f(x0, y0, std::max(1.f, x1 - x0), std::max(1.f, y1 - y0));
        if (timer) timer->post_timer.stop();

        results->clear();
        results->push_back(std::move(r));
        return true;
    }

    std::unique_ptr<HandLandmark> HandLandmark::clone() const {
        auto ret = std::make_unique<HandLandmark>(*this);
        ret->set_runtime(ret->clone_runtime());
        return ret;
    }
} // namespace modeldeploy::vision::hand
