//
// Created by aichao on 2025/3/21.
//
#pragma once

#include "core/tensor.h"



namespace modeldeploy::vision::ocr {
    /*! @brief Preprocessor object for DBDetector serials model.
     */
    class MODELDEPLOY_CXX_EXPORT StructureV2LayoutPreprocessor {
    public:
        virtual ~StructureV2LayoutPreprocessor() = default;
        StructureV2LayoutPreprocessor();

        /** \brief Process the input image and prepare input tensors for runtime
         *
         * \param[in] image_batch The input image batch
         * \param[in] outputs The output tensors which will feed in runtime
         * \return true if the preprocess successed, otherwise false
         */
        virtual bool run(std::vector<ImageData>* image_batch, std::vector<Tensor>* outputs);

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

        /// Get the image info of the last batch, return a list of array
        /// {image width, image height, resize width, resize height}
        const std::vector<std::array<int, 4>>* get_batch_layout_image_info() const {
            return &batch_layout_img_info_;
        }

        /// Set image_shape for the detection preprocess.
        /// This api is usually used when you retrain the model.
        /// Generally, you do not need to use it.
        void set_layout_image_shape(const std::vector<int>& image_shape) {
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
        std::array<int, 4> get_layout_image_info(ImageData* image);

        std::vector<std::array<int, 4>> batch_layout_img_info_;
        std::vector<int> layout_image_shape_ = {3, 800, 608}; // c,h,w
        std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
        std::vector<float> std_ = {0.229f, 0.224f, 0.225f};
        bool is_scale_ = true;

        // default true for pp-structurev2-layout model, backbone picodet.
        bool static_shape_infer_ = true;
    };
} // namespace modeldeploy::vision::ocr
