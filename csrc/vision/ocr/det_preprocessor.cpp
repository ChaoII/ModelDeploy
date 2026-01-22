//
// Created by aichao on 2025/2/21.
//

#include "vision/utils.h"
#include "vision/ocr/det_preprocessor.h"
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/normalize_and_permute.h"

namespace modeldeploy::vision::ocr {
    std::array<int, 4> DBDetectorPreprocessor::ocr_detector_get_info(
        const ImageData* image, const int max_size_len) const {
        const int w = image->width();
        const int h = image->height();
        if (static_img_size_.size() == 2) {
            return {w, h, static_img_size_[0], static_img_size_[1]};
        }
        float ratio = 1.0f;
        const int max_wh = std::max(w, h);
        if (max_wh > max_size_len) {
            ratio = static_cast<float>(max_size_len) / max_wh;
        }
        // 最小图片尺寸为32
        int resize_h = std::max(static_cast<int>(h * ratio), 32);
        int resize_w = std::max(static_cast<int>(w * ratio), 32);
        // 如果尺寸不是32的倍数，则取最接近的32的倍数向上取整
        resize_h = (resize_h + 31) / 32 * 32;
        resize_w = (resize_w + 31) / 32 * 32;
        return {w, h, resize_w, resize_h};
    }

    DBDetectorPreprocessor::DBDetectorPreprocessor() {
    }


    bool DBDetectorPreprocessor::apply(const std::vector<ImageData>& image_batch,
                                       std::vector<Tensor>* outputs) {
        // 组batch找到当前batch最大的宽高组织
        int max_resize_w = 0;
        int max_resize_h = 0;
        batch_info_.resize(image_batch.size());
        for (size_t i = 0; i < image_batch.size(); ++i) {
            const ImageData* mat = &image_batch.at(i);
            batch_info_[i] = ocr_detector_get_info(mat, max_side_len_);
            max_resize_w = std::max(max_resize_w, batch_info_[i][2]);
            max_resize_h = std::max(max_resize_h, batch_info_[i][3]);
            batch_info_[i][2] = max_resize_w;
            batch_info_[i][3] = max_resize_h;
        }
        std::vector<ImageData> processed_images;
        processed_images.reserve(image_batch.size());
        for (size_t i = 0; i < image_batch.size(); ++i) {
            ImageData image = image_batch.at(i);
            auto processed_image = image
                                   .letter_box({max_resize_w, max_resize_h}, 0.0f)
                                   .fuse_normalize_and_permute(mean_, std_);
            processed_images.push_back(processed_image);
        }
        outputs->resize(1);
        ImageData::images_to_tensor(processed_images, &(*outputs)[0]);
        return true;
    }
}
