//
// Created for standalone pedestrian Re-ID (OSNet) preprocessing.
//

#pragma once

#include <utility>

#include "core/md_decl.h"
#include "core/tensor.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision::reid {
    /*! @brief Preprocessor for OSNet pedestrian Re-ID model.
     *  OSNet 输入布局：256(H) x 128(W)，BGR->RGB，ImageNet 归一化，NCHW。
     */
    class MODELDEPLOY_CXX_EXPORT ReIDPreprocessor {
    public:
        ReIDPreprocessor() = default;

        /** \brief Process a batch of images into an NCHW FP32 input tensor.
         *  \param[in] images Input image list (BGR, returned by cv::imread())
         *  \param[in] output_tensors The output tensors which will feed the runtime
         *  \return true if the preprocess succeeded, otherwise false
         */
        bool run(const std::vector<ImageData>& images,
                 std::vector<Tensor>* output_tensors);

        /// Set target size (w, h)
        void set_size(int w, int h) { width_ = w; height_ = h; }

        /// Get target size (w, h)
        std::pair<int, int> size() const { return {width_, height_}; }

    private:
        int width_ = 128;    // internal CHW; OSNet input 256x128 (HxW)
        int height_ = 256;
    };
} // namespace modeldeploy::vision::reid
