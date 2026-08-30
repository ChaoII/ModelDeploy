#pragma once
#include <vector>
#include "core/tensor.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision::seg {
    /*! @brief Postprocessor for FastSAM（一次出 box + mask）。
     *  输入两输出：[box 头, N×(5+32)]（x1,y1,x2,y2,score + 32 个 mask 系数）与
     *  [mask_proto, 1×32×H×W]；掩码 = 系数×proto → sigmoid → 裁剪 resize 到 box → 阈值。
     */
    class MODELDEPLOY_CXX_EXPORT FastSamPostprocessor {
    public:
        FastSamPostprocessor();

        bool run(std::vector<Tensor>& tensors,
                 std::vector<std::vector<InstanceSegResult>>* results,
                 const std::vector<LetterBoxRecord>& letter_box_records) const;

        /// 目标类别数（FastSAM 只需前景，=0 且用输出内嵌 score）
        void set_conf_threshold(const float& conf) { conf_threshold_ = conf; }
        [[nodiscard]] float get_conf_threshold() const { return conf_threshold_; }

        void set_nms_threshold(const float& nms) { nms_threshold_ = nms; }
        [[nodiscard]] float get_nms_threshold() const { return nms_threshold_; }

        void set_mask_threshold(const float& mask) { mask_threshold_ = mask; }
        [[nodiscard]] float get_mask_threshold() const { return mask_threshold_; }

    protected:
        float conf_threshold_{0.3f};
        float nms_threshold_{0.5f};
        float mask_threshold_{0.5f};
    };
} // namespace modeldeploy::vision::seg
