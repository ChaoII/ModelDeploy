//
// Created by aichao on 2025/2/21.
//

#pragma once

#include "../common/processors/normalize.h"
#include "../common/processors/resize.h"
#include "../common/processors/pad.h"
#include "../common/processors/hwc2chw.h"

namespace modeldeploy::vision::ocr {
    class ClassifierPreprocessor {
    public:
        virtual ~ClassifierPreprocessor() = default;
        ClassifierPreprocessor();

        /** \brief Process the input image and prepare input tensors for runtime
         *
         * \param[in] images The input data list, all the elements are FDMat
         * \param[in] outputs The output tensors which will be fed into runtime
         * \param start_index
         * \param end_index
         * \return true if the preprocess successed, otherwise false
         */
        bool Run(std::vector<cv::Mat>* images, std::vector<MDTensor>* outputs,
                 size_t start_index, size_t end_index);

        /** \brief Implement the virtual function of ProcessorManager, Apply() is the
         *  body of Run(). Apply() contains the main logic of preprocessing, Run() is
         *  called by users to execute preprocessing
         *
         * \param[in] image_batch The input image batch
         * \param[in] outputs The output tensors which will feed in runtime
         * \return true if the preprocess successed, otherwise false
         */
        virtual bool Apply(std::vector<cv::Mat>* image_batch, std::vector<MDTensor>* outputs);

        /// Set preprocess normalize parameters, please call this API to customize
        /// the normalize parameters, otherwise it will use the default normalize
        /// parameters.
        void set_normalize(const std::vector<float>& mean,
                          const std::vector<float>& std,
                          bool is_scale) {
            normalize_op_ = std::make_shared<Normalize>(mean, std, is_scale);
        }

        /// Set cls_image_shape for the classification preprocess
        void set_cls_image_shape(const std::vector<int>& cls_image_shape) {
            cls_image_shape_ = cls_image_shape;
        }

        /// Get cls_image_shape for the classification preprocess
        std::vector<int> get_cls_image_shape() const { return cls_image_shape_; }

    private:
        void ocr_classifier_resize_image(cv::Mat* mat,
                                      const std::vector<int>& cls_image_shape);

        std::vector<int> cls_image_shape_ = {3, 48, 192};
        std::shared_ptr<Resize> resize_op_;
        std::shared_ptr<Pad> pad_op_;
        std::shared_ptr<Normalize> normalize_op_;
        std::shared_ptr<HWC2CHW> hwc2chw_op_;
    };
} // namespace ocr
