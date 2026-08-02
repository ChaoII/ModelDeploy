//
// Created by aichao on 2025/2/24.
//

#include "core/md_log.h"
#include "vision/classification/preprocessor.h"

namespace modeldeploy::vision::classification {
    namespace {
        // alpha/beta 由模型配置决定，batch 内共享，仅算一次
        const std::vector<float>& cls_alpha() {
            static const std::vector<float> v = {1.0f / (255.0f * 0.229f),
                                                 1.0f / (255.0f * 0.224f),
                                                 1.0f / (255.0f * 0.225f)};
            return v;
        }
        const std::vector<float>& cls_beta() {
            static const std::vector<float> v = {-0.485f / 0.229f,
                                                 -0.456f / 0.224f,
                                                 -0.406f / 0.225f};
            return v;
        }
        // 计算 fused 映射参数（crop/resize）
        void calc_params(int src_w, int src_h, int dst_w, int dst_h,
                         bool center_crop, float* ox, float* oy,
                         float* sx, float* sy) {
            if (center_crop) {
                const int crop = std::min(src_w, src_h);
                *sx = static_cast<float>(dst_w) / crop;
                *sy = static_cast<float>(dst_h) / crop;
                *ox = -static_cast<float>(src_w - crop) / 2.0f * (*sx);
                *oy = -static_cast<float>(src_h - crop) / 2.0f * (*sy);
            } else {
                *sx = static_cast<float>(dst_w) / src_w;
                *sy = static_cast<float>(dst_h) / src_h;
                *ox = 0.0f;
                *oy = 0.0f;
            }
        }
    }

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
        const int dst_w = size_[0];  // 224
        const int dst_h = size_[1];  // 224
        float ox, oy, sx, sy;
        calc_params(image->width(), image->height(), dst_w, dst_h,
                    enable_center_crop_, &ox, &oy, &sx, &sy);
        if (!backend_->fused_preprocess(*image, output, {dst_w, dst_h},
                                        ox, oy, sx, sy,
                                        cls_alpha(), cls_beta(), true, 0.0f)) return false;
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
        // 整批一次 fused kernel，避免 N 次 launch + concat
        const int dst_w = size_[0];
        const int dst_h = size_[1];
        const int n = static_cast<int>(images->size());
        std::vector<float> oxs(n), oys(n), sxs(n), sys(n);
        for (int i = 0; i < n; ++i) {
            calc_params((*images)[i].width(), (*images)[i].height(), dst_w, dst_h,
                        enable_center_crop_, &oxs[i], &oys[i], &sxs[i], &sys[i]);
        }
        if (!backend_->fused_preprocess_batch(*images, &(*outputs)[0],
                                              {dst_w, dst_h},
                                              oxs, oys, sxs, sys,
                                              cls_alpha(), cls_beta(), true, 0.0f)) {
            MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
            return false;
        }
        return true;
    }
}
