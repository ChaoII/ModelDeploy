//
// Created by aichao on 2025/3/21.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/pad.h"
#include "vision/common/processors/normalize.h"
#include "vision/common/processors/hwc2chw.h"
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
        batch_det_img_info_.resize(image_batch->size());
        std::vector<Tensor> tensors;
        tensors.reserve(image_batch->size());
        for (size_t i = 0; i < image_batch->size(); ++i) {
            const auto& image = image_batch->at(i);
            const auto width = static_cast<float>(image.width());
            const auto height = static_cast<float>(image.height());
            const float ratio = max_len / (std::max(height, width) * 1.0f);
            const int resize_h = static_cast<int>(height * ratio);
            const int resize_w = static_cast<int>(width * ratio);
            ImageData resized;
            if (!backend_->resize(image, &resized, resize_w, resize_h)) return false;
            ImageData normed;
            if (!backend_->normalize(resized, &normed, mean_, std_, is_scale_, false)) return false;
            ImageData padded;
            if (!backend_->pad(normed, &padded, 0, max_len - resize_h, 0, max_len - resize_w, pad_value_[0])) return false;
            Tensor t;
            if (!backend_->hwc2chw(padded, &t)) return false;
            t.expand_dim(0);
            tensors.emplace_back(std::move(t));
            batch_det_img_info_[i] = {
                static_cast<int>(width), static_cast<int>(height), resize_w,
                resize_h
            };
        }

        // Only have 1 output Tensor.
        outputs->resize(1);
        (*outputs)[0] = Tensor::concat(tensors, 0);
        return true;
    }
} // namespace modeldeploy::vision::ocr
