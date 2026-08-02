//
// Created by aichao on 2025/6/10.
//

#include "core/md_log.h"
#include "vision/lpr/lpr_rec/preprocessor.h"

namespace modeldeploy::vision::lpr {
    LprRecPreprocessor::LprRecPreprocessor() {
        size_ = {168, 48};
    }

    bool LprRecPreprocessor::preprocess(ImageData* image, Tensor* output) const {
        // preprocess steps
        // 1. Resize
        // 2. convert_and_permute(swap_rb=true)
        const int src_w = image->width();
        const int src_h = image->height();
        const float scale_x = static_cast<float>(size_[0]) / src_w;  // 168/src_w
        const float scale_y = static_cast<float>(size_[1]) / src_h;  // 48/src_h
        const std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector<float> beta = {-0.588f, -0.588f, -0.588f};
        if (!backend_->fused_preprocess(*image, output, size_,
                                        0.0f, 0.0f, scale_x, scale_y,
                                        alpha, beta, true, 0.0f)) return false;
        return true;
    }

    bool LprRecPreprocessor::run(
        std::vector<ImageData>* images, std::vector<Tensor>* outputs) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        outputs->resize(1);
        if (images->size() == 1) {
            // 单图：直接写进持久 outputs[0]，其 allocate 跨帧复用显存 buffer
            preprocess(&(*images)[0], &(*outputs)[0]);
            return true;
        }
        // Concat all the preprocessed data to a batch tensor
        std::vector<Tensor> tensors(images->size());
        for (size_t i = 0; i < images->size(); ++i) {
            preprocess(&(*images)[i], &tensors[i]);
        }
        if (tensors.size() == 1) {
            (*outputs)[0] = std::move(tensors[0]);
        }
        else {
            (*outputs)[0] = std::move(Tensor::concat(tensors, 0));
        }
        return true;
    }
}
