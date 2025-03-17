//
// Created by aichao on 2025/2/21.
//

#pragma once
#include "csrc/core/md_decl.h"
#include "csrc/core/md_tensor.h"
#include "csrc/vision/ocr/utils/ocr_postprocess_op.h"

namespace modeldeploy::vision::ocr {
    class MODELDEPLOY_CXX_EXPORT DBDetectorPostprocessor {
    public:
        /** \brief Process the result of runtime and fill to results structure
         *
         * \param[in] tensors The inference result from runtime
         * \param[in] results The output result of detector
         * \param[in] batch_det_img_info The detector_preprocess result
         * \return true if the postprocess successes, otherwise false
         */
        bool apply(const std::vector<MDTensor>& tensors,
                   std::vector<std::vector<std::array<int, 8>>>* results,
                   const std::vector<std::array<int, 4>>& batch_det_img_info);

        /// Set det_db_thresh for the detection postprocess, default is 0.3
        void set_det_db_thresh(const double det_db_thresh) { det_db_thresh_ = det_db_thresh; }
        /// Get det_db_thresh of the detection postprocess
        [[nodiscard]] double get_det_db_thresh() const { return det_db_thresh_; }

        /// Set det_db_box_thresh for the detection postprocess, default is 0.6
        void set_det_db_box_thresh(const double det_db_box_thresh) {
            det_db_box_thresh_ = det_db_box_thresh;
        }

        /// Get det_db_box_thresh of the detection postprocess
        [[nodiscard]] double get_det_db_box_thresh() const { return det_db_box_thresh_; }

        /// Set det_db_unclip_ratio for the detection postprocess, default is 1.5
        void set_det_db_unclip_ratio(const double det_db_unclip_ratio) {
            det_db_unclip_ratio_ = det_db_unclip_ratio;
        }

        /// Get det_db_unclip_ratio_ of the detection postprocess
        [[nodiscard]] double get_det_db_unclip_ratio() const { return det_db_unclip_ratio_; }

        /// Set det_db_score_mode for the detection postprocess, default is 'slow'
        void set_det_db_score_mode(const std::string& det_db_score_mode) {
            det_db_score_mode_ = det_db_score_mode;
        }

        /// Get det_db_score_mode_ of the detection postprocess
        [[nodiscard]] std::string get_det_db_score_mode() const { return det_db_score_mode_; }

        /// Set use_dilation for the detection postprocess, default is fasle
        void set_use_dilation(const int use_dilation) { use_dilation_ = use_dilation; }
        /// Get use_dilation of the detection postprocess
        [[nodiscard]] int get_use_dilation() const { return use_dilation_; }

    private:
        bool single_batch_postprocessor(const float* out_data, int n2, int n3,
                                        const std::array<int, 4>& det_img_info,
                                        std::vector<std::array<int, 8>>* boxes_result);
        double det_db_thresh_ = 0.3;
        double det_db_box_thresh_ = 0.6;
        double det_db_unclip_ratio_ = 1.5;
        std::string det_db_score_mode_ = "slow";
        bool use_dilation_ = false;
        PostProcessor util_post_processor_;
    };
} // namespace ocr
