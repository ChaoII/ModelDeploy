//
// Created by aichao on 2025/2/21.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/ocr/rec_preprocessor.h"
#include "vision/common/processors/pad.h"
#include "vision/common/processors/cast.h"

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
        std::vector<ImageData> images;
        images.reserve(image_batch.size());
        for (auto& image : image_batch) {
            ImageData processed_image;
            if (!static_shape_infer_) {
                // 单个batch中的最大
                const auto max_w = static_cast<int>(static_cast<float>(img_h) * max_wh_ratio);
                const float ratio = static_cast<float>(image.width()) / static_cast<float>(image.height());
                int resize_w;
                if (std::ceilf(img_h * ratio) > max_w) {
                    resize_w = max_w;
                }
                else {
                    resize_w = static_cast<int>(ceilf(static_cast<float>(img_h) * ratio));
                }
                processed_image = image.resize(resize_w, img_h)
                                       .pad(0, 0, 0, max_w - resize_w, 127.0f);
            }
            else {
                processed_image = image.resize(img_w, img_h);
            }
            processed_image = processed_image.fuse_normalize_and_permute(mean_, std_, is_scale_);
            images.emplace_back(processed_image);
        }
        // Only have 1 output Tensor.
        outputs->resize(1);
        ImageData::images_to_tensor(images, &(*outputs)[0]);
        return true;
    }
}
