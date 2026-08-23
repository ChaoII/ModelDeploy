#pragma once

#include <memory>
#include <string>
#include <vector>
#include "base_model.h"
#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision::action {

/*! @brief TSN 动作识别：多帧 RGB → 时序聚合 → 类别 scores。
 *  默认 rank-4 uni-dimension 输入 [1, 3*T, H, W]（T 帧 RGB 逐帧 CHW 拼接，轻量平均聚合）。
 *  若模型为 rank-5 [1,3,T,H,W]，在集成回归（Task 8）按 get_input_info(0).shape 适配。
 */
class MODELDEPLOY_CXX_EXPORT TSN : public BaseModel {
public:
    explicit TSN(const std::string& model_file,
                 const RuntimeOption& custom_option = RuntimeOption());
    [[nodiscard]] std::string name() const override { return "TSN"; }

    // 输入：抽帧后的 RGB 帧序列（调用方用 video::VideoDecoder 抽帧给出，或任意历史帧列表）
    bool predict(const std::vector<ImageData>& frames, std::vector<float>* scores);
    [[nodiscard]] bool is_initialized() const;
    [[nodiscard]] std::unique_ptr<TSN> clone() const;

    // 测试缝（纯计算，无需已初始化 runtime）：均匀采样 T 帧 → resize → [0,1] 归一化 → 逐帧 CHW 拼成 [1, 3*T, H, W]
    static bool assemble_frames(const std::vector<ImageData>& frames,
                                int64_t T, int64_t H, int64_t W, Tensor* out);

    // 后处理透传（暴露供无权重单测直接调用）
    bool postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores);

protected:
    bool initialize();
    bool preprocess(const std::vector<ImageData>& frames, std::vector<Tensor>* outputs);

private:
    explicit TSN() = default;   // 供 clone()
    int64_t num_segments_{8};   // 默认时序段数（T，可用 get_input_info 覆盖）
};

} // namespace modeldeploy::vision::action
