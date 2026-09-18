//
// ST-GCN 骨架动作分类（NTU-RGB+D 60）模型封装实现。
//
#include "vision/action/action_skeleton.h"

#include <algorithm>
#include <initializer_list>

#include "core/md_log.h"

namespace modeldeploy::vision::action {

ActionSkeleton::ActionSkeleton(const std::string& model_file, const RuntimeOption& custom_option) {
    runtime_option = custom_option;
    runtime_option.set_model_path(model_file);
    initialized_ = initialize();
}

std::unique_ptr<ActionSkeleton> ActionSkeleton::clone() const {
    auto m = std::unique_ptr<ActionSkeleton>(new ActionSkeleton());
    m->set_runtime(const_cast<ActionSkeleton*>(this)->clone_runtime());
    m->runtime_option = runtime_option;
    m->num_frames_ = num_frames_;
    m->num_joints_ = num_joints_;
    m->num_persons_ = num_persons_;
    m->feat_dim_ = feat_dim_;
    m->initialized_ = initialized_;
    return m;
}

bool ActionSkeleton::is_initialized() const { return initialized_; }

bool ActionSkeleton::initialize() {
    if (!init_runtime()) {
        MD_LOG_ERROR << "ActionSkeleton: failed to init runtime." << std::endl;
        return false;
    }
    // 以模型输入 shape 校正：[N, C, T, V, M]
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;
        if (shp.size() == 5) {
            if (shp[1] > 0) feat_dim_ = shp[1];
            if (shp[2] > 0) num_frames_ = shp[2];
            if (shp[3] > 0) num_joints_ = shp[3];
            if (shp[4] > 0) num_persons_ = shp[4];
        }
    }
    return true;
}

std::vector<Point3f> ActionSkeleton::coco17_to_ntu25(const std::vector<Point3f>& coco) {
    auto pt = [&](int i) -> Point3f {
        return (i >= 0 && static_cast<size_t>(i) < coco.size())
                   ? coco[static_cast<size_t>(i)]
                   : Point3f();
    };
    auto mean = [](std::initializer_list<Point3f> ps) {
        float x = 0.f, y = 0.f, z = 0.f;
        int n = 0;
        for (const auto& p : ps) {
            x += p.x;
            y += p.y;
            z += p.z;
            ++n;
        }
        return n > 0 ? Point3f(x / static_cast<float>(n), y / static_cast<float>(n),
                               z / static_cast<float>(n))
                     : Point3f();
    };

    // NTU 25 关节（0-based；1 spine_base / 21 spine_shoulder）← COCO 17 近似映射。
    // 手部细节关节（8/22/23/24）COCO 无对应点，用腕点近似；缺失处保留 0。
    std::vector<Point3f> out(25);
    out[0] = mean({pt(11), pt(12)});                  // spine base（髋中点）
    out[1] = mean({pt(5), pt(6), pt(11), pt(12)});    // spine mid
    out[2] = mean({pt(5), pt(6)});                    // neck
    out[3] = pt(0);                                   // head ← nose
    out[4] = pt(5);                                   // left shoulder
    out[5] = pt(7);                                   // left elbow
    out[6] = pt(9);                                   // left wrist
    out[7] = pt(9);                                   // left hand（近似）
    out[8] = pt(6);                                   // right shoulder
    out[9] = pt(8);                                   // right elbow
    out[10] = pt(10);                                 // right wrist
    out[11] = pt(10);                                 // right hand（近似）
    out[12] = pt(11);                                 // left hip
    out[13] = pt(13);                                 // left knee
    out[14] = pt(15);                                 // left ankle
    out[15] = pt(15);                                 // left foot（近似）
    out[16] = pt(12);                                 // right hip
    out[17] = pt(14);                                 // right knee
    out[18] = pt(16);                                 // right ankle
    out[19] = pt(16);                                 // right foot（近似）
    out[20] = mean({pt(5), pt(6)});                   // spine shoulder
    out[21] = pt(9);                                  // tip of left hand（近似）
    out[22] = pt(9);                                  // left thumb（近似）
    out[23] = pt(10);                                 // tip of right hand（近似）
    out[24] = pt(10);                                 // right thumb（近似）
    return out;
}

bool ActionSkeleton::assemble_skeleton(const KeyPointSeq& seq, int64_t C, int64_t T,
                                       int64_t V, int64_t M, Tensor* out) {
    if (T <= 0 || V <= 0 || M <= 0 || C <= 0 || !out) return false;
    if (static_cast<int64_t>(seq.frames.size()) < T) return false;
    out->allocate({1, C, T, V, M}, DataType::FP32, Device::CPU);
    float* dst = static_cast<float*>(out->data());
    for (int64_t t = 0; t < T; ++t) {
        const auto& fr = seq.frames[static_cast<size_t>(t)];
        for (int64_t v = 0; v < V; ++v) {
            for (int64_t m = 0; m < M; ++m) {
                const int64_t idx = m * V + v;   // 展平布局：人优先，每人 V 点
                Point3f p;
                if (idx < static_cast<int64_t>(fr.size())) p = fr[static_cast<size_t>(idx)];
                // 恒等预处理：通道 0/1/2 = x/y/score（官方 stgcn pipeline 为 Iden，不居中不归一化）
                const int64_t pos = v * M + m;
                if (C >= 1) dst[(0 * T + t) * V * M + pos] = p.x;
                if (C >= 2) dst[(1 * T + t) * V * M + pos] = p.y;
                if (C >= 3) dst[(2 * T + t) * V * M + pos] = p.z;
            }
        }
    }
    return true;
}

bool ActionSkeleton::preprocess(const KeyPointSeq& seq, std::vector<Tensor>* outputs) {
    int64_t C = feat_dim_, T = num_frames_, V = num_joints_, M = num_persons_;
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;
        if (shp.size() == 5) {
            if (shp[1] > 0) C = shp[1];
            if (shp[2] > 0) T = shp[2];
            if (shp[3] > 0) V = shp[3];
            if (shp[4] > 0) M = shp[4];
        }
    }
    outputs->resize(1);
    return assemble_skeleton(seq, C, T, V, M, &(*outputs)[0]);
}

bool ActionSkeleton::postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores) {
    if (infer_result.empty() || !scores) return false;
    auto& t = infer_result[0];
    const float* p = static_cast<const float*>(t.data());
    const int64_t n = t.size();
    if (n <= 0) return false;
    scores->assign(p, p + n);
    return true;
}

bool ActionSkeleton::predict(const KeyPointSeq& seq, std::vector<float>* scores) {
    if (seq.frames.empty() || !scores) return false;
    if (!preprocess(seq, &reused_input_tensors_)) return false;
    for (int i = 0; i < static_cast<int>(reused_input_tensors_.size()); ++i)
        reused_input_tensors_[i].set_name(get_input_info(i).name);
    if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
    return postprocess(reused_output_tensors_, scores);
}

} // namespace modeldeploy::vision::action
