#include "vision/action/st_gcn.h"
#include "core/md_log.h"

namespace modeldeploy::vision::action {

StGcn::StGcn(const std::string& model_file, const RuntimeOption& custom_option) {
    runtime_option = custom_option;
    runtime_option.set_model_path(model_file);
    initialized_ = initialize();
}

std::unique_ptr<StGcn> StGcn::clone() const {
    auto m = std::unique_ptr<StGcn>(new StGcn());
    m->set_runtime(const_cast<StGcn*>(this)->clone_runtime());
    m->runtime_option = runtime_option;
    m->num_joints_ = num_joints_;
    m->feat_dim_ = feat_dim_;
    m->scale_ = scale_;
    m->initialized_ = initialized_;
    return m;
}

bool StGcn::is_initialized() const { return initialized_; }

bool StGcn::initialize() {
    if (!init_runtime()) {
        MD_LOG_ERROR << "StGcn: failed to init runtime." << std::endl;
        return false;
    }
    // 以模型输入 shape 探测关节数与维度
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;   // [1, C, T, V]
        if (shp.size() == 4) {
            if (shp[1] == 2 || shp[1] == 3) feat_dim_ = static_cast<int32_t>(shp[1]);
            num_joints_ = static_cast<int32_t>(shp[shp.size() - 1]);
        }
    }
    return true;
}

bool StGcn::assemble_skeleton(const KeyPointSeq& seq, int64_t V, int64_t C, Tensor* out) {
    const int64_t T = static_cast<int64_t>(seq.frames.size());
    if (T == 0 || C < 2 || V <= 0) return false;
    out->allocate({1, C, T, V}, DataType::FP32, Device::CPU);
    float* dst = static_cast<float*>(out->data());
    // 平移不变量：以首帧关节中心作参考（轻量；YAGNI 不做复杂骨架归一化）
    float cx = 127.0f, cy = 127.0f;   // 缺省依据图像尺寸；真实流水线用裁剪 box 中心覆盖
    for (int64_t t = 0; t < T; ++t) {
        const auto& fr = seq.frames[t];
        for (int64_t v = 0; v < V; ++v) {
            Point3f j = (v < static_cast<int64_t>(fr.size())) ? fr[v] : Point3f();
            float x = (j.x - cx) / 127.0f;   // 缩放到约 [-1,1]
            float y = (j.y - cy) / 127.0f;
            dst[(0 * T + t) * V + v] = x;    // C=0 => x
            if (C >= 2) dst[(1 * T + t) * V + v] = y;   // C=1 => y
            if (C >= 3) dst[(2 * T + t) * V + v] = 0.0f; // C=2 => z（2D 时为 0）
        }
    }
    return true;
}

bool StGcn::preprocess(const KeyPointSeq& seq, std::vector<Tensor>* outputs) {
    const int64_t T = static_cast<int64_t>(seq.frames.size());
    int64_t V = num_joints_, C = feat_dim_;
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;
        if (shp.size() == 4) { C = shp[1]; V = shp[3]; }
    }
    if (T == 0) return false;
    outputs->resize(1);
    return assemble_skeleton(seq, V, C, &(*outputs)[0]);
}

bool StGcn::postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores) {
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

bool StGcn::predict(const KeyPointSeq& seq, std::vector<float>* scores) {
    if (seq.frames.empty() || !scores) return false;
    if (!preprocess(seq, &reused_input_tensors_)) return false;
    for (int i = 0; i < static_cast<int>(reused_input_tensors_.size()); ++i)
        reused_input_tensors_[i].set_name(get_input_info(i).name);
    if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
    return postprocess(reused_output_tensors_, scores);
}

} // namespace modeldeploy::vision::action
