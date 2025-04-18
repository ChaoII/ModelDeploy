//
// Created by aichao on 2025/3/21.
//
#pragma once

#include "csrc/core/tensor.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/normalize_and_permute.h"


namespace modeldeploy::vision::ocr {
    /*! @brief Preprocessor object for DBDetector serials model.
     */
    class MODELDEPLOY_CXX_EXPORT StructureV2LayoutPreprocessor {
    public:
        StructureV2LayoutPreprocessor();

        /** \brief Process the input image and prepare input tensors for runtime
         *
         * \param[in] image_batch The input image batch
         * \param[in] outputs The output tensors which will feed in runtime
         * \return true if the preprocess successed, otherwise false
         */
        virtual bool run(std::vector<cv::Mat> *image_batch, std::vector<Tensor> *outputs);

        /// Set preprocess normalize parameters, please call this API to customize
        /// the normalize parameters, otherwise it will use the default normalize
        /// parameters.
        void set_normalize(const std::vector<float> &mean,
                           const std::vector<float> &std,
                           bool is_scale) {
            normalize_permute_op_ =
                    std::make_shared<NormalizeAndPermute>(mean, std, is_scale);
        }

        /// Get the image info of the last batch, return a list of array
        /// {image width, image height, resize width, resize height}
        const std::vector<std::array<int, 4>> *get_batch_layout_image_info() {
            return &batch_layout_img_info_;
        }

        /// This function will disable normalize in preprocessing step.
        void disable_normalize() { disable_permute_ = true; }

        /// This function will disable hwc2chw in preprocessing step.
        void disable_permute() { disable_normalize_ = true; }

        /// Set image_shape for the detection preprocess.
        /// This api is usually used when you retrain the model.
        /// Generally, you do not need to use it.
        void set_layout_image_shape(const std::vector<int> &image_shape) {
            layout_image_shape_ = image_shape;
        }

        /// Get cls_image_shape for the classification preprocess
        [[nodiscard]] std::vector<int> get_layout_image_shape() const { return layout_image_shape_; }

        /// Set static_shape_infer is true or not. When deploy PP-StructureV2
        /// on hardware which can not support dynamic input shape very well,
        /// like Huawei Ascned, static_shape_infer needs to to be true.
        void set_static_shape_infer(bool static_shape_infer) {
            static_shape_infer_ = static_shape_infer;
        }

        /// Get static_shape_infer of the recognition preprocess
        bool get_static_shape_infer() const { return static_shape_infer_; }

    private:
        bool resize_layout_image(cv::Mat *img, int resize_w, int resize_h);

        std::array<int, 4> get_layout_image_info(cv::Mat *img);

        // for recording the switch of hwc2chw
        bool disable_permute_ = false;
        // for recording the switch of normalize
        bool disable_normalize_ = false;
        std::vector<std::array<int, 4>> batch_layout_img_info_;
        std::shared_ptr<Resize> resize_op_;
        std::shared_ptr<NormalizeAndPermute> normalize_permute_op_;
        std::vector<int> layout_image_shape_ = {3, 800, 608}; // c,h,w
        // default true for pp-structurev2-layout model, backbone picodet.
        bool static_shape_infer_ = true;

    };
} // namespace modeldeploy::vision::ocr
