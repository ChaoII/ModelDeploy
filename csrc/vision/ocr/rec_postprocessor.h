//
// Created by aichao on 2025/2/21.
//


#pragma once

#include "csrc/vision/ocr/utils/ocr_postprocess_op.h"
#include "../../core/md_tensor.h"

namespace modeldeploy::vision::ocr {
    /*! @brief Postprocessor object for Recognizer serials model.
     */
    class RecognizerPostprocessor {
    public:
        RecognizerPostprocessor();
        /** \brief Create a postprocessor instance for Recognizer serials model
         *
         * \param[in] label_path The path of label_dict
         */
        explicit RecognizerPostprocessor(const std::string& label_path);

        /** \brief Process the result of runtime and fill to RecognizerResult
         *
         * \param[in] tensors The inference result from runtime
         * \param[in] texts The output text results of recognizer
         * \param[in] rec_scores The output score results of recognizer
         * \return true if the postprocess successed, otherwise false
         */
        bool run(const std::vector<MDTensor>& tensors,
                 std::vector<std::string>* texts, std::vector<float>* rec_scores);

        bool run(const std::vector<MDTensor>& tensors,
                 std::vector<std::string>* texts, std::vector<float>* rec_scores,
                 size_t start_index, size_t total_size,
                 const std::vector<int>& indices);

    private:
        bool single_batch_postprocessor(const float* out_data,
                                      const std::vector<int64_t>& output_shape,
                                      std::string* text, float* rec_score);
        bool initialized_ = false;
        std::vector<std::string> label_list_;
    };
}
