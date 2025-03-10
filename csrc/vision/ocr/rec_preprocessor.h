//
// Created by aichao on 2025/2/21.
//


#pragma once
#include "csrc/core/md_tensor.h"
#include "csrc/vision/common/processors/pad.h"
#include "csrc/vision/common/processors/cast.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/normalize.h"
#include "csrc/vision/common/processors/normalize_and_permute.h"

namespace modeldeploy::vision::ocr {
    class MODELDEPLOY_CXX_EXPORT RecognizerPreprocessor final {
    public:
        virtual ~RecognizerPreprocessor() = default;
        RecognizerPreprocessor();

        /** \brief Process the input image and prepare input tensors for runtime
         *
         * \param[in] images The input data list, all the elements are FDMat
         * \param[in] outputs The output tensors which will be fed into runtime
         * \return true if the preprocess successes, otherwise false
         */
        bool Run(std::vector<cv::Mat>* images, std::vector<MDTensor>* outputs,
                 size_t start_index, size_t end_index,
                 const std::vector<int>& indices);

        /** \brief Implement the virtual function of ProcessorManager, Apply() is the
         *  body of Run(). Apply() contains the main logic of preprocessing, Run() is
         *  called by users to execute preprocessing
         *
         * \param[in] image_batch The input image batch
         * \param[in] outputs The output tensors which will feed in runtime
         * \return true if the preprocess successed, otherwise false
         */
        bool Apply(std::vector<cv::Mat>* image_batch, std::vector<MDTensor>* outputs);

        /// Set static_shape_infer is true or not. When deploy PP-OCR
        /// on hardware which can not support dynamic input shape very well,
        /// like Huawei Ascned, static_shape_infer needs to to be true.
        void set_static_shape_infer(bool static_shape_infer) {
            static_shape_infer_ = static_shape_infer;
        }

        /// Get static_shape_infer of the recognition preprocess
        bool get_static_shape_infer() const { return static_shape_infer_; }

        /// Set preprocess normalize parameters, please call this API to customize
        /// the normalize parameters, otherwise it will use the default normalize
        /// parameters.
        void set_normalize(const std::vector<float>& mean,
                           const std::vector<float>& std,
                           bool is_scale) {
            normalize_permute_op_ =
                std::make_shared<NormalizeAndPermute>(mean, std, is_scale);
            normalize_op_ = std::make_shared<Normalize>(mean, std, is_scale);
        }

        /// Set rec_image_shape for the recognition preprocess
        void set_rec_image_shape(const std::vector<int>& rec_image_shape) {
            rec_image_shape_ = rec_image_shape;
        }

        /// Get rec_image_shape for the recognition preprocess
        std::vector<int> get_rec_image_shape() { return rec_image_shape_; }

        /// This function will disable normalize in preprocessing step.
        void disable_normalize() { disable_permute_ = true; }
        /// This function will disable hwc2chw in preprocessing step.
        void disable_permute() { disable_normalize_ = true; }

    private:
        void ocr_recognizer_resize_image(cv::Mat* mat, float max_wh_ratio,
                                         const std::vector<int>& rec_image_shape,
                                         bool static_shape_infer);
        // for recording the switch of hwc2chw
        bool disable_permute_ = false;
        // for recording the switch of normalize
        bool disable_normalize_ = false;
        std::vector<int> rec_image_shape_ = {3, 48, 320};
        bool static_shape_infer_ = false;
        std::shared_ptr<Resize> resize_op_;
        std::shared_ptr<Pad> pad_op_;
        std::shared_ptr<NormalizeAndPermute> normalize_permute_op_;
        std::shared_ptr<Normalize> normalize_op_;
        std::shared_ptr<HWC2CHW> hwc2chw_op_;
        std::shared_ptr<Cast> cast_op_;
    };
} // namespace ocr
