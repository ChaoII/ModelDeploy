#include "vision/action/tsn.h"
#include <algorithm>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include "core/md_log.h"

namespace modeldeploy::vision::action {

TSN::TSN(const std::string& model_file, const RuntimeOption& custom_option) {
    runtime_option = custom_option;
    runtime_option.set_model_path(model_file);
    initialized_ = initialize();
}

std::unique_ptr<TSN> TSN::clone() const {
    auto m = std::unique_ptr<TSN>(new TSN());
    m->set_runtime(const_cast<TSN*>(this)->clone_runtime());
    m->runtime_option = runtime_option;
    m->num_segments_ = num_segments_;
    m->initialized_ = initialized_;
    return m;
}

bool TSN::is_initialized() const { return initialized_; }

bool TSN::initialize() {
    if (!init_runtime()) {
        MD_LOG_ERROR << "TSN: failed to init runtime." << std::endl;
        return false;
    }
    // 若模型输入明确了时序段数，用 shape 覆盖 num_segments_
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;
        if (shp.size() == 4 && shp[1] > 0 && shp[1] % 3 == 0)
            num_segments_ = shp[1] / 3;          // [1, 3*T, H, W]
        else if (shp.size() == 5 && shp[2] > 0)
            num_segments_ = shp[2];              // [1, C, T, H, W]
    }
    return true;
}

bool TSN::assemble_frames(const std::vector<ImageData>& frames, int64_t T,
                          int64_t H, int64_t W, Tensor* out) {
    // 均匀采样 T 帧（不足则循环填充到 T）
    const size_t K = frames.size();
    if (K == 0 || T <= 0 || H <= 0 || W <= 0) return false;
    std::vector<int> idxs;
    idxs.reserve(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
        int i = (K > 1) ? static_cast<int>((t * K) / T) : 0;
        idxs.push_back(std::min(i, static_cast<int>(K - 1)));
    }

    const int64_t C = 3;
    out->allocate({1, C * T, H, W}, DataType::FP32, Device::CPU);
    float* dst = static_cast<float*>(out->data());
    for (int64_t t = 0; t < T; ++t) {
        const ImageData& f = frames[idxs[t]];
        cv::Mat src;
        if (!f.asMat(&src) || src.empty()) return false;
        cv::Mat rgb, resized;
        if (src.channels() == 3 && src.type() == CV_8UC3)
            cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
        else if (src.channels() == 1)
            cv::cvtColor(src, rgb, cv::COLOR_GRAY2RGB);
        else
            rgb = src;
        cv::resize(rgb, resized, cv::Size(static_cast<int>(W), static_cast<int>(H)));
        // 逐帧 CHW，[0,1] 归一化；拼到 [1, 3*T, H, W]
        float* dst_t = dst + t * C * H * W;
        for (int64_t c = 0; c < C; ++c)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w) {
                    const uint8_t v = resized.at<cv::Vec3b>(static_cast<int>(h), static_cast<int>(w))[static_cast<int>(c)];
                    dst_t[c * H * W + h * W + w] = static_cast<float>(v) / 255.0f;
                }
    }
    return true;
}

bool TSN::preprocess(const std::vector<ImageData>& frames, std::vector<Tensor>* outputs) {
    // 以模型输入 shape 为目标尺寸；缺省用 num_segments_ 与 224x224
    int64_t H = 224, W = 224;
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;
        if (shp.size() == 4 && shp[3] > 0) { H = shp[2]; W = shp[3]; }
        else if (shp.size() == 5 && shp[4] > 0) { H = shp[3]; W = shp[4]; }
    }
    outputs->resize(1);
    return assemble_frames(frames, num_segments_, H, W, &(*outputs)[0]);
}

bool TSN::postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores) {
    if (infer_result.empty()) return false;
    auto& t = infer_result[0];
    const float* p = static_cast<const float*>(t.data());
    const int64_t n = t.size();
    if (n <= 0) return false;
    scores->clear();
    scores->reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) scores->push_back(p[i]);
    return true;
}

bool TSN::predict(const std::vector<ImageData>& frames, std::vector<float>* scores) {
    if (frames.empty() || !scores) return false;
    if (!preprocess(frames, &reused_input_tensors_)) return false;
    for (int i = 0; i < static_cast<int>(reused_input_tensors_.size()); ++i)
        reused_input_tensors_[i].set_name(get_input_info(i).name);
    if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
    return postprocess(reused_output_tensors_, scores);
}

} // namespace modeldeploy::vision::action
