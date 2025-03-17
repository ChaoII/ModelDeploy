//
// Created by aichao on 2025/2/21.
//

#include "./det_preprocessor.h"

#include "./utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    std::array<int, 4> DBDetectorPreprocessor::ocr_detector_get_info(
        const cv::Mat* img, const int max_size_len) const {
        const int w = img->cols;
        const int h = img->rows;
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
        resize_op_ = std::make_shared<Resize>(-1, -1);
        std::vector<float> value = {0, 0, 0};
        pad_op_ = std::make_shared<Pad>(0, 0, 0, 0, value);
        normalize_permute_op_ = std::make_shared<NormalizeAndPermute>(
            std::vector({0.485f, 0.456f, 0.406f}),
            std::vector({0.229f, 0.224f, 0.225f}), true);
    }

    bool DBDetectorPreprocessor::resize_image(cv::Mat* img, const int resize_w, const int resize_h,
                                              const int max_resize_w, const int max_resize_h) const {
        resize_op_->SetWidthAndHeight(resize_w, resize_h);
        (*resize_op_)(img);
        pad_op_->SetPaddingSize(0, max_resize_h - resize_h, 0,
                                max_resize_w - resize_w);
        (*pad_op_)(img);
        return true;
    }

    bool DBDetectorPreprocessor::apply(std::vector<cv::Mat>* image_batch,
                                       std::vector<MDTensor>* outputs) {
        int max_resize_w = 0;
        int max_resize_h = 0;
        batch_det_img_info_.clear();
        batch_det_img_info_.resize(image_batch->size());
        for (size_t i = 0; i < image_batch->size(); ++i) {
            const cv::Mat* mat = &image_batch->at(i);
            batch_det_img_info_[i] = ocr_detector_get_info(mat, max_side_len_);
            max_resize_w = std::max(max_resize_w, batch_det_img_info_[i][2]);
            max_resize_h = std::max(max_resize_h, batch_det_img_info_[i][3]);
        }
        for (size_t i = 0; i < image_batch->size(); ++i) {
            cv::Mat* mat = &image_batch->at(i);
            resize_image(mat, batch_det_img_info_[i][2], batch_det_img_info_[i][3],
                         max_resize_w, max_resize_h);
            (*normalize_permute_op_)(mat);
        }
        outputs->resize(1);
        utils::mats_to_tensor(*image_batch, &(*outputs)[0]);
        return true;
    }
}
