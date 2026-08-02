//
// Created by aichao on 2025/3/21.
//

#include "core/md_log.h"
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/ocr/structurev2_layout_preprocessor.h"


namespace modeldeploy::vision::ocr {
    StructureV2LayoutPreprocessor::StructureV2LayoutPreprocessor() {
    }

    std::array<int, 4> StructureV2LayoutPreprocessor::get_layout_image_info(ImageData* image) {
        if (static_shape_infer_) {
            return {
                image->width(), image->height(), layout_image_shape_[2],
                layout_image_shape_[1]
            };
        }
        MD_LOG_ERROR << "not support dynamic shape inference now!" << std::endl;
        return {
            image->width(), image->height(), layout_image_shape_[2], layout_image_shape_[1]
        };
    }


    bool StructureV2LayoutPreprocessor::run(std::vector<ImageData>* image_batch,
                                            std::vector<Tensor>* outputs) {
        batch_layout_img_info_.clear();
        const int n = static_cast<int>(image_batch->size());
        batch_layout_img_info_.resize(n);
        // alpha/beta 由 mean/std 决定，batch 内共享，仅算一次
        std::vector<float> alpha(3), beta(3);
        for (int c = 0; c < 3; ++c) {
            alpha[c] = 1.0f / (255.0f * std_[c]);
            beta[c] = -mean_[c] / std_[c];
        }
        outputs->resize(1);
        if (n == 1) {
            ImageData* image = &image_batch->at(0);
            batch_layout_img_info_[0] = get_layout_image_info(image);
            const int dst_w = batch_layout_img_info_[0][2];
            const int dst_h = batch_layout_img_info_[0][3];
            const float scale_x = static_cast<float>(dst_w) / image->width();
            const float scale_y = static_cast<float>(dst_h) / image->height();
            if (!backend_->fused_preprocess(*image, &(*outputs)[0], {dst_w, dst_h},
                                            0.0f, 0.0f, scale_x, scale_y,
                                            alpha, beta, false, 0.0f)) return false;
            return true;
        }
        // 整批一次 fused kernel（batch 内 dst 尺寸须统一）
        int dst_w = -1, dst_h = -1;
        std::vector<float> oxs(n, 0.0f), oys(n, 0.0f), sxs(n), sys(n);
        for (int i = 0; i < n; ++i) {
            ImageData* image = &image_batch->at(i);
            batch_layout_img_info_[i] = get_layout_image_info(image);
            if (dst_w < 0) {
                dst_w = batch_layout_img_info_[i][2];
                dst_h = batch_layout_img_info_[i][3];
            } else if (dst_w != batch_layout_img_info_[i][2] ||
                       dst_h != batch_layout_img_info_[i][3]) {
                MD_LOG_ERROR << "batch with mixed dst sizes not supported, use single run"
                    << std::endl;
                return false;
            }
            sxs[i] = static_cast<float>(dst_w) / image->width();
            sys[i] = static_cast<float>(dst_h) / image->height();
        }
        if (!backend_->fused_preprocess_batch(*image_batch, &(*outputs)[0], {dst_w, dst_h},
                                              oxs, oys, sxs, sys,
                                              alpha, beta, false, 0.0f)) return false;
        return true;
    }
} // namespace modeldeploy::vision::ocr
