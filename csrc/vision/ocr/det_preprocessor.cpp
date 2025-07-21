//
// Created by aichao on 2025/2/21.
//

#include "vision/utils.h"
#include "vision/ocr/det_preprocessor.h"
#include "vision/ocr/utils/ocr_utils.h"

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
        resize_op_ = std::make_shared<Resize>(-1, -1);
        std::vector<float> value = {0, 0, 0};
        pad_op_ = std::make_shared<Pad>(0, 0, 0, 0, value);
        normalize_permute_op_ = std::make_shared<NormalizeAndPermute>(
            std::vector({0.485f, 0.456f, 0.406f}),
            std::vector({0.229f, 0.224f, 0.225f}), true);
    }

    bool DBDetectorPreprocessor::resize_image(ImageData* image, const int resize_w, const int resize_h,
                                              const int max_resize_w, const int max_resize_h) const {
        cv::Mat img;
        image->to_mat(&img);
        resize_op_->set_width_and_height(resize_w, resize_h);
        (*resize_op_)(&img);
        pad_op_->set_padding_size(0, max_resize_h - resize_h, 0,
                                  max_resize_w - resize_w);
        (*pad_op_)(&img);
        return true;
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
        cv::Mat _images;
        for (size_t i = 0; i < image_batch->size(); ++i) {
            ImageData* image = &image_batch->at(i);
            resize_image(image, batch_det_img_info_[i][2], batch_det_img_info_[i][3],
                         max_resize_w, max_resize_h);
            cv::Mat mat;
            image->to_mat(&mat);
            (*normalize_permute_op_)(&mat);
            _images.push_back(mat);
        }
        outputs->resize(1);
        utils::mats_to_tensor(_images, &(*outputs)[0]);
        return true;
    }
}
