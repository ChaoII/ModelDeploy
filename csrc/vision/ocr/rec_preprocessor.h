//
// Created by aichao on 2025/2/21.
//


#pragma once
#include "core/tensor.h"

namespace modeldeploy::vision::ocr {
    class MODELDEPLOY_CXX_EXPORT RecognizerPreprocessor final {
    public:
        ~RecognizerPreprocessor() = default;
        RecognizerPreprocessor();

        /** \brief Process the input image and prepare input tensors for runtime
         *
         * \param[in] images The input data list, all the elements are FDMat
         * \param[in] outputs The output tensors which will be fed into runtime
         * \param start_index
         * \param end_index
         * \param indices
         * \return true if the preprocess successes, otherwise false
         */
        bool run(const std::vector<ImageData>* images, std::vector<Tensor>* outputs,
                 size_t start_index, size_t end_index,
                 const std::vector<int>& indices) const;

        /** \brief Implement the virtual function of ProcessorManager, Apply() is the
         *  body of Run(). Apply() contains the main logic of preprocessing, Run() is
         *  called by users to execute preprocessing
         *
         * \param[in] image_batch The input image batch
         * \param[in] outputs The output tensors which will feed in runtime
         * \return true if the preprocess successed, otherwise false
         */
        bool apply(const std::vector<ImageData>* image_batch, std::vector<Tensor>* outputs) const;

        /// Set static_shape_infer is true or not. When deploy PP-OCR
        /// on hardware which can not support dynamic input shape very well,
        /// like Huawei Ascned, static_shape_infer needs to to be true.
        void set_static_shape_infer(bool static_shape_infer) {
            static_shape_infer_ = static_shape_infer;
        }

        /// Get static_shape_infer of the recognition preprocess
        [[nodiscard]] bool get_static_shape_infer() const { return static_shape_infer_; }

        /// Set preprocess normalize parameters, please call this API to customize
        /// the normalize parameters, otherwise it will use the default normalize
        /// parameters.
        void set_normalize(const std::vector<float>& mean,
                           const std::vector<float>& std,
                           bool is_scale) {
            mean_ = mean;
            std_ = std;
            is_scale_ = is_scale;
        }

        /// Set rec_image_shape for the recognition preprocess
        void set_rec_image_shape(const std::vector<int>& rec_image_shape) {
            rec_image_shape_ = rec_image_shape;
        }

        /// Get rec_image_shape for the recognition preprocess
        std::vector<int> get_rec_image_shape() { return rec_image_shape_; }

    private:
        std::vector<int> rec_image_shape_ = {3, 48, 320};
        bool static_shape_infer_ = false;
        std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
        std::vector<float> std_ = {0.5f, 0.5f, 0.5f};
        bool is_scale_ = true;
        std::vector<float> pad_value_ = {127, 127, 127};
    };
} // namespace ocr
