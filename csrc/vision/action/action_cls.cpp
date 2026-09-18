//
// PP-TSMv2（Kinetics-400, 16 帧）视频动作分类模型封装实现。
//
#include "vision/action/action_cls.h"

#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/md_log.h"

namespace modeldeploy::vision::action {

ActionCls::ActionCls(const std::string& model_file, const RuntimeOption& custom_option) {
    runtime_option = custom_option;
    runtime_option.set_model_path(model_file);
    initialized_ = initialize();
}

std::unique_ptr<ActionCls> ActionCls::clone() const {
    auto m = std::unique_ptr<ActionCls>(new ActionCls());
    m->set_runtime(const_cast<ActionCls*>(this)->clone_runtime());
    m->runtime_option = runtime_option;
    m->num_frames_ = num_frames_;
    m->input_h_ = input_h_;
    m->input_w_ = input_w_;
    m->initialized_ = initialized_;
    return m;
}

bool ActionCls::is_initialized() const { return initialized_; }

bool ActionCls::initialize() {
    if (!init_runtime()) {
        MD_LOG_ERROR << "ActionCls: failed to init runtime." << std::endl;
        return false;
    }
    // 以模型输入 shape 校正时序段数与空间尺寸：PP-TSMv2 为 [N, T, 3, H, W]
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;
        if (shp.size() == 5) {
            if (shp[1] > 0) num_frames_ = shp[1];
            if (shp[3] > 0) input_h_ = shp[3];
            if (shp[4] > 0) input_w_ = shp[4];
        }
    }
    return true;
}

bool ActionCls::assemble_clip(const std::vector<ImageData>& frames, int64_t T,
                              int64_t H, int64_t W, Tensor* out) {
    const size_t K = frames.size();
    if (K == 0 || T <= 0 || H <= 0 || W <= 0 || !out) return false;
    const int64_t C = 3;
    out->allocate({1, T, C, H, W}, DataType::FP32, Device::CPU);
    float* dst = static_cast<float*>(out->data());
    // ImageNet 归一化常量（RGB，与 PaddleVideo Normalize 一致）
    static const float kMean[3] = {0.485f, 0.456f, 0.406f};
    static const float kStd[3] = {0.229f, 0.224f, 0.225f};

    for (int64_t t = 0; t < T; ++t) {
        // 均匀采样（每段取 1 帧）：idx = t*K/T，钳制到有效范围
        int idx = (K > 1) ? static_cast<int>((t * static_cast<int64_t>(K)) / T) : 0;
        idx = std::clamp(idx, 0, static_cast<int>(K) - 1);
        const ImageData& f = frames[static_cast<size_t>(idx)];
        cv::Mat src;
        if (!f.asMat(&src) || src.empty()) return false;

        // 统一到 3 通道 BGR
        cv::Mat bgr;
        if (src.channels() == 1) cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
        else if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;

        // 依据 ImageData 格式决定是否 BGR→RGB（PKG_BGR_U8 需转换）
        const MdImageType ptype = f.type();
        const bool is_rgb = ptype == MdImageType::PKG_RGB_U8 || ptype == MdImageType::PLA_RGB_U8;
        cv::Mat rgb;
        if (is_rgb) rgb = bgr;
        else cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        // 短边缩放到 256（保持长宽比），并保证不短于目标裁剪边长
        const double scale = static_cast<double>(kShortSide) /
                             static_cast<double>(std::min(rgb.cols, rgb.rows));
        const int nw = std::max(static_cast<int>(std::lround(rgb.cols * scale)),
                                static_cast<int>(W));
        const int nh = std::max(static_cast<int>(std::lround(rgb.rows * scale)),
                                static_cast<int>(H));
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

        // 中心裁剪 H×W
        const int x0 = std::clamp((nw - static_cast<int>(W)) / 2, 0, nw - static_cast<int>(W));
        const int y0 = std::clamp((nh - static_cast<int>(H)) / 2, 0, nh - static_cast<int>(H));
        const cv::Mat crop = resized(cv::Rect(x0, y0, static_cast<int>(W), static_cast<int>(H)));

        // 写为 CHW：dst[t][c][h][w]
        float* dst_t = dst + t * C * H * W;
        for (int64_t c = 0; c < C; ++c)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w) {
                    const float v = static_cast<float>(
                        crop.at<cv::Vec3b>(static_cast<int>(h), static_cast<int>(w))[static_cast<int>(c)]) /
                        255.0f;
                    dst_t[(c * H + h) * W + w] = (v - kMean[c]) / kStd[c];
                }
    }
    return true;
}

bool ActionCls::preprocess(const std::vector<ImageData>& frames, std::vector<Tensor>* outputs) {
    outputs->resize(1);
    return assemble_clip(frames, num_frames_, input_h_, input_w_, &(*outputs)[0]);
}

bool ActionCls::postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores) {
    if (infer_result.empty() || !scores) return false;
    auto& t = infer_result[0];
    const float* p = static_cast<const float*>(t.data());
    const int64_t n = t.size();
    if (n <= 0) return false;
    scores->assign(p, p + n);
    return true;
}

bool ActionCls::predict(const std::vector<ImageData>& frames, std::vector<float>* scores) {
    if (frames.empty() || !scores) return false;
    if (!preprocess(frames, &reused_input_tensors_)) return false;
    for (int i = 0; i < static_cast<int>(reused_input_tensors_.size()); ++i)
        reused_input_tensors_[i].set_name(get_input_info(i).name);
    if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
    return postprocess(reused_output_tensors_, scores);
}

} // namespace modeldeploy::vision::action
