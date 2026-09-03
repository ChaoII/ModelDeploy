//
// Created by aichao on 2025/2/24.
//

#pragma once
#include "core/md_decl.h"
#include "vision/common/result.h"
#include "core/tensor.h"

namespace modeldeploy::vision::classification {
    /*! @brief Postprocessor object for YOLOv5Cls serials model.
    */
    class MODELDEPLOY_CXX_EXPORT ClassificationPostprocessor {
    public:
        /** \brief Create a postprocessor instance for YOLOv5Cls serials model
        */
        ClassificationPostprocessor();
        /** \brief Process the result of runtime and fill to ClassifyResult structure
        *
        * \param[in] tensors The inference result from runtime
        * \param[in] results The output result of classification
        * \return true if the postprocess successful, otherwise false
        */
        bool run(std::vector<Tensor>& tensors,
                 std::vector<ClassifyResult>* results) const;

        /// Set topk, default 1
        void set_top_k(const int& top_k) {top_k_ = top_k;}

        /// Get topk, default 1
        [[nodiscard]] int get_top_k() const { return top_k_; }

        // Set multi_label, default false
        void set_multi_label(const bool& multi_label) {multi_label_ = multi_label;}

        /// Get multi_label, default false
        [[nodiscard]] bool get_multi_label() const { return multi_label_; }

        // 自动判别单/多标签分类：单标签模型输出为 softmax，全类概率和约等于 1；
        // 多标签模型各维为独立概率，全类和可显著大于 1。当 auto_multi_label 开启时，
        // 按全类概率和是否超过阈值(默认1.5)自动决定本次按单标签还是多标签后处理。
        // 若同时显式 set_multi_label，则以显式设置优先。
        void set_multi_label_auto(const bool& enable,
                                  const float& threshold = 1.5f) {
            auto_multi_label_ = enable;
            auto_multi_label_thresh_ = threshold;
        }
        [[nodiscard]] bool get_multi_label_auto() const { return auto_multi_label_; }

    protected:
        int top_k_;
        bool multi_label_ = false;
        bool auto_multi_label_ = false;
        float auto_multi_label_thresh_ = 1.5f;
    };
}
