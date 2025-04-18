//
// Created by aichao on 2025/3/21.
//

#pragma once

#include "csrc/core/tensor.h"
#include "csrc/core/md_decl.h"
#include "csrc/vision/ocr/utils/ocr_postprocess_op.h"


namespace modeldeploy::vision::ocr {
    /*! @brief Postprocessor object for DBDetector serials model.
     */
    class MODELDEPLOY_CXX_EXPORT StructureV2TablePostprocessor {
    public:
        StructureV2TablePostprocessor();

        /** \brief Create a postprocessor instance for Recognizer serials model
         *
         * \param[in] label_path The path of label_dict
         */
        explicit StructureV2TablePostprocessor(const std::string &dict_path);

        /** \brief Process the result of runtime and fill to RecognizerResult
         *
         * \param[in] tensors The inference result from runtime
         * \param[in] texts The output text results of recognizer
         * \param[in] rec_scores The output score results of recognizer
         * \return true if the postprocess successed, otherwise false
         */
        bool run(const std::vector<Tensor> &tensors,
                 std::vector<std::vector<std::array<int, 8>>> *bbox_batch_list,
                 std::vector<std::vector<std::string>> *structure_batch_list,
                 const std::vector<std::array<int, 4>> &batch_det_img_info);

    private:
        PostProcessor util_post_processor_;

        bool single_batch_post_processor(const float *structure_probs,
                                         const float *bbox_preds,
                                         size_t slice_dim,
                                         size_t prob_dim,
                                         size_t box_dim,
                                         int img_width,
                                         int img_height,
                                         std::vector<std::array<int, 8>> *boxes_result,
                                         std::vector<std::string> *structure_list_result);

        bool merge_no_span_structure{true};
        std::vector<std::string> dict_character;
        std::vector<std::string> td_tokens{"<td>", "<td", "<td></td>"};
        std::map<std::string, int> dict;
        int ignore_beg_token_idx{};
        int ignore_end_token_idx{};
        int dict_end_idx{};
        bool initialized_ = false;
    };
} // namespace modeldeploy::vision::ocr

