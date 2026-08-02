//
// Created by aichao on 2025/3/21.
//

#include "core/md_log.h"
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/ocr/structurev2_table_preprocessor.h"


namespace modeldeploy::vision::ocr {
    StructureV2TablePreprocessor::StructureV2TablePreprocessor() {
    }


    bool StructureV2TablePreprocessor::run(std::vector<ImageData>* images,
                                           std::vector<Tensor>* outputs,
                                           const size_t start_index, size_t end_index,
                                           const std::vector<int>& indices) {
        if (images->size() == 0 || end_index <= start_index ||
            end_index > images->size()) {
            MD_LOG_ERROR << "images->size() or index error. Correct is: 0 <= start_index < "
                "end_index <= images->size()" << std::endl;
            return false;
        }

        std::vector<ImageData> mats(end_index - start_index);
        for (size_t i = start_index; i < end_index; ++i) {
            size_t real_index = i;
            if (indices.size() != 0) {
                real_index = indices[i];
            }
            mats[i - start_index] = images->at(real_index);
        }
        return run(&mats, outputs);
    }

    bool StructureV2TablePreprocessor::run(std::vector<ImageData>* image_batch,
                                           std::vector<Tensor>* outputs) {
        batch_det_img_info_.clear();
        const int n = static_cast<int>(image_batch->size());
        batch_det_img_info_.resize(n);
        // alpha/beta 由 mean/std 决定，batch 内共享，仅算一次
        std::vector<float> alpha(3), beta(3);
        for (int c = 0; c < 3; ++c) {
            alpha[c] = 1.0f / (255.0f * std_[c]);
            beta[c] = -mean_[c] / std_[c];
        }
        outputs->resize(1);
        if (n == 1) {
            const auto& image = image_batch->at(0);
            const int src_w = image.width();
            const int src_h = image.height();
            const float ratio = max_len / (std::max(static_cast<float>(src_h), static_cast<float>(src_w)) * 1.0f);
            const int resize_h = static_cast<int>(static_cast<float>(src_h) * ratio);
            const int resize_w = static_cast<int>(static_cast<float>(src_w) * ratio);
            const float scale_x = static_cast<float>(resize_w) / src_w;
            const float scale_y = static_cast<float>(resize_h) / src_h;
            if (!backend_->fused_preprocess(image, &(*outputs)[0], {max_len, max_len},
                                            0.0f, 0.0f, scale_x, scale_y,
                                            alpha, beta, false, pad_value_[0])) return false;
            batch_det_img_info_[0] = {src_w, src_h, resize_w, resize_h};
            return true;
        }
        // 整批一次 fused kernel（每图独立 resize，右侧/下侧 pad）
        std::vector<float> oxs(n, 0.0f), oys(n, 0.0f), sxs(n), sys(n);
        for (int i = 0; i < n; ++i) {
            const auto& image = image_batch->at(i);
            const int src_w = image.width();
            const int src_h = image.height();
            const float ratio = max_len / (std::max(static_cast<float>(src_h), static_cast<float>(src_w)) * 1.0f);
            const int resize_h = static_cast<int>(static_cast<float>(src_h) * ratio);
            const int resize_w = static_cast<int>(static_cast<float>(src_w) * ratio);
            sxs[i] = static_cast<float>(resize_w) / src_w;
            sys[i] = static_cast<float>(resize_h) / src_h;
            batch_det_img_info_[i] = {src_w, src_h, resize_w, resize_h};
        }
        if (!backend_->fused_preprocess_batch(*image_batch, &(*outputs)[0], {max_len, max_len},
                                              oxs, oys, sxs, sys,
                                              alpha, beta, false, pad_value_[0])) return false;
        return true;
    }
} // namespace modeldeploy::vision::ocr
