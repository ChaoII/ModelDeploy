//
// Created by aichao on 2025/2/21.
//

#include "core/md_log.h"
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/ocr/rec_preprocessor.h"

namespace modeldeploy::vision::ocr {
    RecognizerPreprocessor::RecognizerPreprocessor() {
    }

    bool RecognizerPreprocessor::run(const std::vector<ImageData>& images,
                                     std::vector<Tensor>* outputs,
                                     const size_t start_index, const size_t end_index,
                                     const std::vector<int>& indices) const {
        if (images.empty() || end_index <= start_index || end_index > images.size()) {
            MD_LOG_ERROR << "images->size() or index error. Correct is: 0 <= start_index < "
                "end_index <= images->size()" << std::endl;;
            return false;
        }

        std::vector<ImageData> real_images(end_index - start_index);
        for (size_t i = start_index; i < end_index; ++i) {
            size_t real_index = i;
            if (!indices.empty()) {
                real_index = indices[i];
            }
            real_images[i - start_index] = images.at(real_index);
        }
        return apply(real_images, outputs);
    }

    bool RecognizerPreprocessor::apply(const std::vector<ImageData>& image_batch,
                                       std::vector<Tensor>* outputs) const {
        const int img_h = rec_image_shape_[1];
        const int img_w = rec_image_shape_[2];
        float max_wh_ratio = static_cast<float>(img_w) * 1.0f / static_cast<float>(img_h);
        for (const auto& image : image_batch) {
            float ori_wh_ratio = static_cast<float>(image.width()) * 1.0f / static_cast<float>(image.height());
            max_wh_ratio = std::max(max_wh_ratio, ori_wh_ratio);
        }
        // batch 统一输出宽：所有图 pad 到同一 max_w，保证 concat 时 shape 一致
        const int batch_max_w = static_shape_infer_
            ? img_w
            : static_cast<int>(static_cast<float>(img_h) * max_wh_ratio);
        // alpha/beta 由模型配置决定，batch 内共享，仅算一次
        const float s = is_scale_ ? 255.0f : 1.0f;
        std::vector<float> alpha(3), beta(3);
        for (int c = 0; c < 3; ++c) {
            alpha[c] = 1.0f / (s * std_[c]);
            beta[c] = -mean_[c] / std_[c];
        }
        // pad 127 在归一化后的值（旧：resize->pad(127 raw)->fuse_normalize）
        const float pad_norm = pad_value_[0] * alpha[0] + beta[0];

        const int n = static_cast<int>(image_batch.size());
        if (n == 1) {
            const ImageData& image = image_batch[0];
            outputs->resize(1);
            int resize_w;
            if (!static_shape_infer_) {
                const float ratio = static_cast<float>(image.width()) / static_cast<float>(image.height());
                if (std::ceil(img_h * ratio) > batch_max_w) resize_w = batch_max_w;
                else resize_w = static_cast<int>(ceilf(static_cast<float>(img_h) * ratio));
            } else {
                resize_w = img_w;
            }
            const float scale_x = static_cast<float>(resize_w) / image.width();
            const float scale_y = static_cast<float>(img_h) / image.height();
            if (!backend_->fused_preprocess(image, &(*outputs)[0], {batch_max_w, img_h},
                                            0.0f, 0.0f, scale_x, scale_y,
                                            alpha, beta, true, pad_norm)) return false;
            return true;
        }
        // 整批一次 fused kernel（每图独立 resize_w -> scale_x，右侧 pad）
        std::vector<float> oxs(n, 0.0f), oys(n, 0.0f), sxs(n), sys(n);
        for (int i = 0; i < n; ++i) {
            const ImageData& image = image_batch[i];
            int resize_w;
            if (!static_shape_infer_) {
                const float ratio = static_cast<float>(image.width()) / static_cast<float>(image.height());
                if (std::ceil(img_h * ratio) > batch_max_w) resize_w = batch_max_w;
                else resize_w = static_cast<int>(ceilf(static_cast<float>(img_h) * ratio));
            } else {
                resize_w = img_w;
            }
            sxs[i] = static_cast<float>(resize_w) / image.width();
            sys[i] = static_cast<float>(img_h) / image.height();
        }
        // Only have 1 output Tensor.
        outputs->resize(1);
        if (!backend_->fused_preprocess_batch(image_batch, &(*outputs)[0], {batch_max_w, img_h},
                                              oxs, oys, sxs, sys,
                                              alpha, beta, true, pad_norm)) return false;
        return true;
    }
}
