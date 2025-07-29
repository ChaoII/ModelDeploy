//
// Created by aichao on 2025/5/30.
//

#pragma once
#include "core/tensor.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision::detection {
    /*! @brief Preprocessor object for YOLOv5Seg serials model.
    */
    class MODELDEPLOY_CXX_EXPORT UltralyticsObbPreprocessor {
    public:
        /// Create a preprocessor instance for YOLOv5Seg serials model
        UltralyticsObbPreprocessor();
        /** \brief Process the input image and prepare input tensors for runtime
        *
        * \param[in] images The input image data list, all the elements are returned by cv::imread()
        * \param[in] outputs The output tensors which will feed in runtime
        * \param[in] letter_box_records The shape info list, record input_shape and output_shape
        * \return true if the preprocess successed, otherwise false
        */
        bool run(std::vector<ImageData>* images, std::vector<Tensor>* outputs,
                 std::vector<LetterBoxRecord>* letter_box_records) const;

        /// Set target size, tuple of (width, height), default size = {640, 640}
        void set_size(const std::vector<int>& size) { size_ = size; }

        /// Get target size, tuple of (width, height), default size = {640, 640}
        [[nodiscard]] std::vector<int> get_size() const { return size_; }

        /// Set padding value, size should be the same as channels
        void set_padding_value(const std::vector<float>& padding_value) {
            padding_value_ = padding_value;
        }

        /// Get padding value, size should be the same as channels
        [[nodiscard]] std::vector<float> get_padding_value() const { return padding_value_; }

        void use_cuda_preproc() { use_cuda_preproc_ = true; }

    protected:
        bool preprocess(ImageData* image, Tensor* output,
                        LetterBoxRecord* letter_box_record) const;
        bool use_cuda_preproc_ = false;
        std::vector<int> size_;
        std::vector<float> padding_value_;
    };
}
