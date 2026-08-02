//
// Created by aichao on 2025/3/24.
//

#include "core/md_log.h"
#include "vision/face/face_gender/preprocessor.h"


namespace modeldeploy::vision::face {
    bool SeetaFaceGenderPreprocessor::preprocess(ImageData* image, Tensor* output) const {
        // 1. Resize
        // 2. Cast (uint8->float, 不缩放)
        // 3. HWC2CHW
        const int src_w = image->width();
        const int src_h = image->height();
        const float scale_x = static_cast<float>(size_[0]) / src_w;  // 112/src_w
        const float scale_y = static_cast<float>(size_[1]) / src_h;
        if (!backend_->fused_preprocess(*image, output, size_,
                                        0.0f, 0.0f, scale_x, scale_y,
                                        {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                                        false, 0.0f)) return false;
        return true;
    }

    bool SeetaFaceGenderPreprocessor::run(std::vector<ImageData>* images,
                                          std::vector<Tensor>* outputs) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        outputs->resize(1);
        // Concat all the preprocessed data to a batch tensor
        std::vector<Tensor> tensors(images->size());
        for (size_t i = 0; i < images->size(); ++i) {
            if (!preprocess(&(*images)[i], &tensors[i])) {
                MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
                return false;
            }
        }
        if (tensors.size() == 1) {
            (*outputs)[0] = std::move(tensors[0]);
        }
        else {
            (*outputs)[0] = std::move(Tensor::concat(tensors, 0));
        }
        return true;
    }
} // namespace modeldeploy::vision::face_rec
