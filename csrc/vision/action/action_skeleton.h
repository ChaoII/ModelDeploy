#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "base_model.h"
#include "runtime/runtime_option.h"
#include "vision/action/keypoint_seq.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision::action {

/*! @brief ST-GCN 骨架动作分类（NTU-RGB+D 60，cross-subject）。
 *
 *  与同目录的 action::StGcn（面向 [1,C,T,V] 旧口径）不同：本类对应 PaddleVideo 导出的
 *  STGCN_ntucs ONNX，输入为 rank-5 [N, C, T, V, M]（C=3 → x,y,score；T=300；V=25；M=2），
 *  输出 [N, 60] logits。预处理按官方 stgcn_ntucs.yaml 的 pipeline 为 **Iden（恒等）**：
 *  不做骨架居中/归一化，原样送入坐标（NTU 拓扑的邻接矩阵与 data_bn 已固化在图中）。
 *
 *  KeyPointSeq.frames[t] 为第 t 帧的**展平**骨骼点：按人优先排列，每人 V 个点，
 *  即 person p 的点位于下标 [p*V, (p+1)*V)。缺失的人补零。
 */
class MODELDEPLOY_CXX_EXPORT ActionSkeleton : public BaseModel {
public:
    explicit ActionSkeleton(const std::string& model_file,
                            const RuntimeOption& custom_option = RuntimeOption());
    [[nodiscard]] std::string name() const override { return "ActionSkeleton"; }

    bool predict(const KeyPointSeq& seq, std::vector<float>* scores);
    [[nodiscard]] bool is_initialized() const;
    [[nodiscard]] std::unique_ptr<ActionSkeleton> clone() const;

    // 测试缝（纯计算）：KeyPointSeq → [1, C, T, V, M]（恒等预处理）
    static bool assemble_skeleton(const KeyPointSeq& seq, int64_t C, int64_t T,
                                  int64_t V, int64_t M, Tensor* out);

    // 测试缝：COCO 17 点 → NTU 25 点（近似映射，缺失/手部细节关节用最近邻或均值填充）
    static std::vector<Point3f> coco17_to_ntu25(const std::vector<Point3f>& coco);

    bool postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores);

    [[nodiscard]] int64_t num_frames() const { return num_frames_; }
    [[nodiscard]] int64_t num_joints() const { return num_joints_; }
    [[nodiscard]] int64_t num_persons() const { return num_persons_; }
    [[nodiscard]] int64_t feat_dim() const { return feat_dim_; }

protected:
    bool initialize();
    bool preprocess(const KeyPointSeq& seq, std::vector<Tensor>* outputs);

private:
    explicit ActionSkeleton() = default;   // 供 clone()

    int64_t num_frames_{300};
    int64_t num_joints_{25};
    int64_t num_persons_{2};
    int64_t feat_dim_{3};
};

} // namespace modeldeploy::vision::action
