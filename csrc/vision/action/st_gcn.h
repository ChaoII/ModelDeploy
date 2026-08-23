#pragma once

#include <memory>
#include <string>
#include <vector>
#include "base_model.h"
#include "runtime/runtime_option.h"
#include "vision/action/keypoint_seq.h"

namespace modeldeploy::vision::action {

/*! @brief ST-GCN 骨骼动作识别：骨骼序列 → 图卷积 → 类别 scores。
 *  输入 KeyPointSeq，输入张量 [1, C, T, V]（C=2 或 3）。上游由 UltralyticsPose 逐帧提关键点组序。
 */
class MODELDEPLOY_CXX_EXPORT StGcn : public BaseModel {
public:
    explicit StGcn(const std::string& model_file,
                   const RuntimeOption& custom_option = RuntimeOption());
    [[nodiscard]] std::string name() const override { return "StGcn"; }

    bool predict(const KeyPointSeq& seq, std::vector<float>* scores);
    [[nodiscard]] bool is_initialized() const;
    [[nodiscard]] std::unique_ptr<StGcn> clone() const;

    // 测试缝（纯计算）：骨骼序列 → [1, C, T, V]
    static bool assemble_skeleton(const KeyPointSeq& seq, int64_t V, int64_t C, Tensor* out);

protected:
    bool initialize();
    bool preprocess(const KeyPointSeq& seq, std::vector<Tensor>* outputs);
    bool postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores);

private:
    explicit StGcn() = default;   // 供 clone()
    int32_t num_joints_{18};
    int32_t feat_dim_{2};         // 2(x,y) 或 3(x,y,z)
    float scale_{1.0f};
};

} // namespace modeldeploy::vision::action
