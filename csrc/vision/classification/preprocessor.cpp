//
// Created by aichao on 2025/2/24.
//

#include "core/md_log.h"
#include "vision/classification/preprocessor.h"

namespace modeldeploy::vision::classification {
    ClassificationPreprocessor::ClassificationPreprocessor() {
        size_ = {224, 224}; //{h,w}
    }

    bool ClassificationPreprocessor::preprocess(ImageData* image, Tensor* output) const {
        // yolov8-cls's preprocess steps
        // 1. CenterCrop
        // 2. Resize
        // 2. Normalize
        if (image->width() <= 0 || image->height() <= 0) {
            return false;
        }
        const int src_w = image->width();
        const int src_h = image->height();
        const int dst_w = size_[0];  // 224
        const int dst_h = size_[1];  // 224
        float origin_x = 0.0f, origin_y = 0.0f;
        float scale_x, scale_y;
        if (enable_center_crop_) {
            const int crop = std::min(src_w, src_h);
            // 各轴独立 scale：crop 正方形 -> 非均匀 resize 到 {dst_w, dst_h}
            const float scale_x = static_cast<float>(dst_w) / crop;
            const float scale_y = static_cast<float>(dst_h) / crop;
            // center_crop 映射 src = (dst - origin)/scale，crop 区域在 src 中偏移 offset，故 origin = -offset*scale（负数）
            origin_x = -static_cast<float>(src_w - crop) / 2.0f * scale_x;
            origin_y = -static_cast<float>(src_h - crop) / 2.0f * scale_y;
        } else {
            scale_x = static_cast<float>(dst_w) / src_w;
            scale_y = static_cast<float>(dst_h) / src_h;
        }
        const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
        const std::vector<float> std = {0.229f, 0.224f, 0.225f};
        std::vector<float> alpha(3), beta(3);
        for (int c = 0; c < 3; ++c) {
            alpha[c] = 1.0f / (255.0f * std[c]);  // convert(1/255) + normalize(scale=false)
            beta[c] = -mean[c] / std[c];
        }
        if (!backend_->fused_preprocess(*image, output, {dst_w, dst_h},
                                        origin_x, origin_y, scale_x, scale_y,
                                        alpha, beta, true, 0.0f)) return false;
        return true;
    }

    bool ClassificationPreprocessor::run(
        std::vector<ImageData>* images, std::vector<Tensor>* outputs) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0."
                << std::endl;
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
}
