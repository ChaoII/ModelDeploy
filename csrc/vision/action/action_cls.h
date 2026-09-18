#pragma once

#include <memory>
#include <string>
#include <vector>

#include "base_model.h"
#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision::action {

/*! @brief PP-TSMv2（Kinetics-400, 16 帧）RGB 视频动作分类。
 *
 *  与同目录的 action::TSN 不同：本类对应 PaddleVideo 导出的 PP-TSMv2 ONNX，
 *  输入为 rank-5 NCHW 时序张量 [N, T, 3, H, W]（T=16，H=W=224），输出 [N, 400] logits。
 *  预处理严格对齐 PaddleVideo 的 Sampler/Scale/CenterCrop/Normalize：
 *    1) 将已缓冲帧序列均匀采样出 T 帧（每段取 1 帧）；
 *    2) 短边缩放到 256（保持长宽比）；
 *    3) 中心裁剪 H×W；
 *    4) 转 RGB、/255、ImageNet mean/std 归一化。
 *  本类只负责模型本身与预处理；跨帧滚动缓冲由调用方（InferenceEngine）维护。
 */
class MODELDEPLOY_CXX_EXPORT ActionCls : public BaseModel {
public:
    explicit ActionCls(const std::string& model_file,
                       const RuntimeOption& custom_option = RuntimeOption());
    [[nodiscard]] std::string name() const override { return "ActionCls"; }

    // 输入：抽帧后的 RGB/BGR 帧序列（任意尺寸）；输出：类别 logits（Kinetics-400 顺序）
    bool predict(const std::vector<ImageData>& frames, std::vector<float>* scores);
    [[nodiscard]] bool is_initialized() const;
    [[nodiscard]] std::unique_ptr<ActionCls> clone() const;

    // 测试缝（纯计算，无需已初始化 runtime）：
    // 均匀采样 T 帧 → 短边 256 → 中心裁剪 H×W → ImageNet 归一化 → [1, T, 3, H, W]
    static bool assemble_clip(const std::vector<ImageData>& frames,
                              int64_t T, int64_t H, int64_t W, Tensor* out);

    // 后处理透传（暴露供无权重单测直接调用）
    bool postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores);

    [[nodiscard]] int64_t num_frames() const { return num_frames_; }
    [[nodiscard]] int64_t input_height() const { return input_h_; }
    [[nodiscard]] int64_t input_width() const { return input_w_; }

    static constexpr int64_t kShortSide = 256;   // 缩放短边（PaddleVideo Scale(short_size=256)）

protected:
    bool initialize();
    bool preprocess(const std::vector<ImageData>& frames, std::vector<Tensor>* outputs);

private:
    explicit ActionCls() = default;   // 供 clone()

    int64_t num_frames_{16};
    int64_t input_h_{224};
    int64_t input_w_{224};
};

} // namespace modeldeploy::vision::action
