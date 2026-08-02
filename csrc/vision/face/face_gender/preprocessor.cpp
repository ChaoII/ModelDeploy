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
        if (images->size() == 1) {
            // 单图：直接写进持久 outputs[0]，其 allocate 跨帧复用显存 buffer
            if (!preprocess(&(*images)[0], &(*outputs)[0])) {
                MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
                return false;
            }
            return true;
        }
        // 整批一次 fused kernel
        const int n = static_cast<int>(images->size());
        std::vector<float> oxs(n, 0.0f), oys(n, 0.0f), sxs(n), sys(n);
        for (int i = 0; i < n; ++i) {
            sxs[i] = static_cast<float>(size_[0]) / (*images)[i].width();
            sys[i] = static_cast<float>(size_[1]) / (*images)[i].height();
        }
        if (!backend_->fused_preprocess_batch(*images, &(*outputs)[0], size_,
                                              oxs, oys, sxs, sys,
                                              {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                                              false, 0.0f)) {
            MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::face_rec
