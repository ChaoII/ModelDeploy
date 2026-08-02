//
// Created by aichao on 2025/2/21.
//

#include "core/md_log.h"
#include "vision/ocr/cls_preprocessor.h"
#include "vision/ocr/utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    ClassifierPreprocessor::ClassifierPreprocessor() {
    }

    bool ClassifierPreprocessor::run(const std::vector<ImageData>& images,
                                     std::vector<Tensor>* outputs,
                                     const size_t start_index, const size_t end_index) {
        if (static_cast<int>(images.size()) == 0 || end_index <= start_index ||
            end_index > images.size()) {
            MD_LOG_ERROR << "images->size() or index error. Correct is: 0 <= start_index < "
                "end_index <= images->size()" << std::endl;
            return false;
        }
        std::vector<ImageData> mats(end_index - start_index);
        for (size_t i = start_index; i < end_index; ++i) {
            mats[i - start_index] = images.at(i);
        }
        return apply(mats, outputs);
    }

    bool ClassifierPreprocessor::apply(const std::vector<ImageData>& image_batch,
                                       std::vector<Tensor>* outputs) {
        std::vector<Tensor> tensors;
        tensors.reserve(image_batch.size());
        for (auto& image : image_batch) {
            const int img_h = cls_image_shape_[1];
            const int img_w = cls_image_shape_[2];
            const int src_w = image.width();
            const int src_h = image.height();
            const float ratio = static_cast<float>(src_w) / static_cast<float>(src_h);
            int resize_w;
            if (ceilf(static_cast<float>(img_h) * ratio) > static_cast<float>(img_w)) resize_w = img_w;
            else resize_w = static_cast<int>(ceilf(static_cast<float>(img_h) * ratio));
            const float scale_x = static_cast<float>(resize_w) / src_w;
            const float scale_y = static_cast<float>(img_h) / src_h;
            Tensor t;
            const std::vector<float> alpha = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
            const std::vector<float> beta = {-1.0f, -1.0f, -1.0f};
            if (!backend_->fused_preprocess(image, &t, {resize_w, img_h},
                                            0.0f, 0.0f, scale_x, scale_y,
                                            alpha, beta, false, 0.0f)) return false;
            tensors.emplace_back(std::move(t));
        }
        // Only have 1 output tensor.
        outputs->resize(1);
        (*outputs)[0] = Tensor::concat(tensors, 0);
        return true;
    }
}
