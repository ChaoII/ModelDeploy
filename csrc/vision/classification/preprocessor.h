//
// Created by aichao on 2025/2/24.
//

#pragma once

#include "core/md_decl.h"
#include "core/tensor.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision::classification {
    /*! @brief Preprocessor object for YOLOv5Cls serials model.
    */
    class MODELDEPLOY_CXX_EXPORT UltralyticsClsPreprocessor {
    public:
        /** \brief Create a preprocessor instance for YOLOv5Cls serials model
        */
        UltralyticsClsPreprocessor();

        /** \brief Process the input image and prepare input tensors for runtime
       *
       * \param[in] images The input image data list, all the elements are returned by cv::imread()
       * \param[in] outputs The output tensors which will feed in runtime
       * \return true if the preprocess successfully, otherwise false
       */
        bool run(std::vector<ImageData>* images, std::vector<Tensor>* outputs) const;

        /// Set target size, tuple of (width, height), default size = {224, 224}
        void set_size(const std::vector<int>& size) { size_ = size; }

        /// Get target size, tuple of (width, height), default size = {224, 224}
        [[nodiscard]] std::vector<int> get_size() const { return size_; }

        /// enable center crop, for almost classification model such as same width and height
        /// but for person attribute model, width and height is {192, 256}
        void enable_center_crop() { enable_center_crop_ = true; }

        void disable_center_crop() { enable_center_crop_ = false; }


    protected:
        bool preprocess(ImageData* image, Tensor* output) const;

        // target size, tuple of (width, height), default size = {224, 224}
        // person attribute rec size = {192, 256}
        std::vector<int> size_;
        // default true, for cls...
        bool enable_center_crop_ = true;
    };
}
