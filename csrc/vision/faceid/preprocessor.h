//
// Created by aichao on 2025/3/24.
//

#pragma once
#include "csrc/core/md_decl.h"
#include "csrc/vision/common/result.h"
#include "csrc/core/md_tensor.h"


namespace modeldeploy::vision::faceid {
    /*! @brief Preprocessor object for AdaFace serials model.
     */
    class MODELDEPLOY_CXX_EXPORT AdaFacePreprocessor {
    public:
        /** \brief Create a preprocessor instance for AdaFace serials model
         */
        AdaFacePreprocessor() = default;

        /** \brief Process the input image and prepare input tensors for runtime
         *
         * \param[in] images The input image data list, all the elements are returned by cv::imread()
         * \param[in] outputs The output tensors which will feed in runtime
         * \return true if the preprocess successed, otherwise false
         */
        bool Run(std::vector<cv::Mat>* images, std::vector<MDTensor>* outputs);

        /// Get Size
        std::vector<int> GetSize() { return size_; }

        /// Set size.
        void SetSize(std::vector<int>& size) { size_ = size; }

    protected:
        bool Preprocess(cv::Mat* mat, MDTensor* output);
        // Argument for image preprocessing step, tuple of (width, height),
        // decide the target size after resize, default (248, 248)
        std::vector<int> size_{248, 248};
    };
} // namespace modeldeploy::vision::faceid
