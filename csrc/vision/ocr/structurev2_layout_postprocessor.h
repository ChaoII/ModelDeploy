//
// Created by aichao on 2025/3/21.
//

#pragma once

#include "vision/common/result.h"
#include "core/tensor.h"


namespace modeldeploy::vision::ocr
{
    /*! @brief Postprocessor object for PaddleDet serials model.
     */
    class MODELDEPLOY_CXX_EXPORT StructureV2LayoutPostprocessor {
    public:
        StructureV2LayoutPostprocessor() = default;

        /** \brief Process the result of runtime and fill to batch DetectionResult
         *
         * \param[in] tensors The inference result from runtime
         * \param[in] results The output result of layout detection
         * \param[in] batch_layout_img_info The image info of input images,
         *            {{image width, image height, resize width, resize height},...}
         * \return true if the postprocess successed, otherwise false
         */
        bool run(const std::vector<Tensor>& tensors,
                 std::vector<std::vector<DetectionResult>>* results,
                 const std::vector<std::array<int, 4>>& batch_layout_img_info);

        /// Set score_threshold_ for layout detection postprocess, default is 0.4
        void set_score_threshold(float score_threshold) { score_threshold_ = score_threshold; }

        /// Set nms_threshold_ for layout detection postprocess, default is 0.5
        void set_nms_threshold(float nms_threshold) { nms_threshold_ = nms_threshold; }

        /// Set num_class_ for layout detection postprocess, default is 5
        void set_num_class(int num_class) { num_class_ = num_class; }

        /// Set fpn_stride_ for layout detection postprocess, default is {8, 16, 32, 64}
        void set_fpn_stride(const std::vector<int>& fpn_stride) { fpn_stride_ = fpn_stride; }

        /// Set reg_max_ for layout detection postprocess, default is 8
        void set_reg_max(int reg_max) { reg_max_ = reg_max; } // should private ?
        /// Get score_threshold_ of layout detection postprocess, default is 0.4
        [[nodiscard]] float get_score_threshold() const { return score_threshold_; }

        /// Get nms_threshold_ of layout detection postprocess, default is 0.5
        [[nodiscard]] float get_nms_threshold() const { return nms_threshold_; }

        /// Get num_class_ of layout detection postprocess, default is 5
        [[nodiscard]] int get_num_class() const { return num_class_; }

        /// Get fpn_stride_ of layout detection postprocess, default is {8, 16, 32, 64}
        [[nodiscard]] std::vector<int> get_fpn_stride() const { return fpn_stride_; }

        /// Get reg_max_ of layout detection postprocess, default is 8
        [[nodiscard]] int get_reg_max() const { return reg_max_; }

    private:
        Rect2f dis_pred_to_bbox(const std::vector<float>& bbox_pred, int x, int y,
                                int stride, int resize_w, int resize_h, int reg_max);

        bool single_batch_postprocessor(const std::vector<Tensor>& single_batch_tensors,
                                        const std::array<int, 4>& layout_img_info,
                                        std::vector<DetectionResult>* result);

        void set_single_batch_external_data(const std::vector<Tensor>& tensors,
                                            std::vector<Tensor>& single_batch_tensors,
                                            size_t batch_idx);

        std::vector<int> fpn_stride_ = {8, 16, 32, 64};
        float score_threshold_ = 0.4;
        float nms_threshold_ = 0.5;
        int num_class_ = 5;
        int reg_max_ = 8;
    };
} // namespace modeldeploy::vision::ocr
