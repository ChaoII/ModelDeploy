//
// Created by aichao on 2025/2/21.
//

#include "vision/utils.h"
#include "vision/ocr/det_preprocessor.h"
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/pad.h"
#include "vision/common/processors/normalize_and_permute.h"

namespace modeldeploy::vision::ocr {
    std::array<int, 4> DBDetectorPreprocessor::ocr_detector_get_info(
        const ImageData* image, const int max_size_len) const {
        cv::Mat img;
        image->to_mat(&img);
        const int w = img.cols;
        const int h = img.rows;
        if (static_shape_infer_) {
            return {w, h, det_image_shape_[2], det_image_shape_[1]};
        }
        float ratio = 1.0f;
        if (const int max_wh = w >= h ? w : h; max_wh > max_size_len) {
            if (h > w) {
                ratio = static_cast<float>(max_size_len) / static_cast<float>(h);
            }
            else {
                ratio = static_cast<float>(max_size_len) / static_cast<float>(w);
            }
        }
        int resize_h = static_cast<int>(static_cast<float>(h) * ratio);
        int resize_w = static_cast<int>(static_cast<float>(w) * ratio);
        resize_h = std::max(static_cast<int>(std::round(static_cast<float>(resize_h) / 32) * 32), 32);
        resize_w = std::max(static_cast<int>(std::round(static_cast<float>(resize_w) / 32) * 32), 32);
        return {w, h, resize_w, resize_h};
    }

    DBDetectorPreprocessor::DBDetectorPreprocessor() {
    }


    bool DBDetectorPreprocessor::apply(std::vector<ImageData>* image_batch,
                                       std::vector<Tensor>* outputs) {
        int max_resize_w = 0;
        int max_resize_h = 0;
        batch_det_img_info_.clear();
        batch_det_img_info_.resize(image_batch->size());
        for (size_t i = 0; i < image_batch->size(); ++i) {
            const ImageData* mat = &image_batch->at(i);
            batch_det_img_info_[i] = ocr_detector_get_info(mat, max_side_len_);
            max_resize_w = std::max(max_resize_w, batch_det_img_info_[i][2]);
            max_resize_h = std::max(max_resize_h, batch_det_img_info_[i][3]);
        }
        std::vector<ImageData> processed_images;
        for (size_t i = 0; i < image_batch->size(); ++i) {
            ImageData image = image_batch->at(i);
            auto s = image.resize(batch_det_img_info_[i][2], batch_det_img_info_[i][3]);
            s = s.pad(0, max_resize_h - batch_det_img_info_[i][3], 0, max_resize_w - batch_det_img_info_[i][2], 0);
            s = s.fuse_normalize_and_permute({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});
            processed_images.push_back(s);
        }
        outputs->resize(1);
        ImageData::images_to_tensor(processed_images, &(*outputs)[0]);
        return true;
    }
}
