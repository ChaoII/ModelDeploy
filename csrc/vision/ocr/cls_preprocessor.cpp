//
// Created by aichao on 2025/2/21.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/ocr/cls_preprocessor.h"

#include <vision/common/processors/normalize_and_permute.h>

#include "vision/common/processors/pad.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/normalize.h"
#include "vision/common/processors/hwc2chw.h"
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
            const float ratio = static_cast<float>(image.width()) / static_cast<float>(image.height());
            int resize_w;
            if (ceilf(static_cast<float>(img_h) * ratio) > static_cast<float>(img_w))
                resize_w = img_w;
            else
                resize_w = static_cast<int>(ceilf(static_cast<float>(img_h) * ratio));
            ImageData resized;
            if (!backend_->resize(image, &resized, resize_w, img_h)) return false;
            ImageData normed;
            if (!backend_->normalize(resized, &normed, mean_, std_, is_scale_, false)) return false;
            ImageData padded = normed;
            if (normed.width() < cls_image_shape_[2]) {
                if (!backend_->pad(normed, &padded, 0, 0, 0, cls_image_shape_[2] - normed.width(), 0.0f)) return false;
            }
            Tensor t;
            if (!backend_->hwc2chw(padded, &t)) return false;
            t.expand_dim(0);
            tensors.emplace_back(std::move(t));
        }
        // Only have 1 output tensor.
        outputs->resize(1);
        (*outputs)[0] = Tensor::concat(tensors, 0);
        return true;
    }
}
