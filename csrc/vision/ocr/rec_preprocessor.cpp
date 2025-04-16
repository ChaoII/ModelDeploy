//
// Created by aichao on 2025/2/21.
//


#include "rec_preprocessor.h"

#include <csrc/core/md_log.h>


#include "./utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    RecognizerPreprocessor::RecognizerPreprocessor() {
        resize_op_ = std::make_shared<Resize>(-1, -1);
        std::vector<float> value = {127, 127, 127};
        pad_op_ = std::make_shared<Pad>(0, 0, 0, 0, value);
        std::vector mean = {0.5f, 0.5f, 0.5f};
        std::vector std = {0.5f, 0.5f, 0.5f};
        normalize_permute_op_ =
            std::make_shared<NormalizeAndPermute>(mean, std, true);
        normalize_op_ = std::make_shared<Normalize>(mean, std, true);
        hwc2chw_op_ = std::make_shared<HWC2CHW>();
        cast_op_ = std::make_shared<Cast>("float");
    }

    void RecognizerPreprocessor::ocr_recognizer_resize_image(
        cv::Mat* mat, const float max_wh_ratio,
        const std::vector<int>& rec_image_shape, const bool static_shape_infer) const {
        const int img_h = rec_image_shape[1];
        int img_w = rec_image_shape[2];

        if (!static_shape_infer) {
            img_w = static_cast<int>(static_cast<float>(img_h) * max_wh_ratio);
            const float ratio = static_cast<float>(mat->cols) / static_cast<float>(mat->rows);
            int resize_w;
            if (ceilf(static_cast<float>(img_h) * ratio) > static_cast<float>(img_w)) {
                resize_w = img_w;
            }
            else {
                resize_w = static_cast<int>(ceilf(static_cast<float>(img_h) * ratio));
            }
            resize_op_->SetWidthAndHeight(resize_w, img_h);
            (*resize_op_)(mat);
            pad_op_->SetPaddingSize(0, 0, 0, img_w - mat->cols);
            (*pad_op_)(mat);
        }
        else {
            if (mat->cols >= img_w) {
                // Resize W to 320
                resize_op_->SetWidthAndHeight(img_w, img_h);
                (*resize_op_)(mat);
            }
            else {
                resize_op_->SetWidthAndHeight(mat->cols, img_h);
                (*resize_op_)(mat);
                // Pad to 320
                pad_op_->SetPaddingSize(0, 0, 0, img_w - mat->cols);
                (*pad_op_)(mat);
            }
        }
    }

    bool RecognizerPreprocessor::run(const std::vector<cv::Mat>* images,
                                     std::vector<Tensor>* outputs,
                                     const size_t start_index, const size_t end_index,
                                     const std::vector<int>& indices) const {
        if (images->empty() || end_index <= start_index ||
            end_index > images->size()) {
            MD_LOG_ERROR << "images->size() or index error. Correct is: 0 <= start_index < "
                "end_index <= images->size()" << std::endl;;
            return false;
        }

        std::vector<cv::Mat> mats(end_index - start_index);
        for (size_t i = start_index; i < end_index; ++i) {
            size_t real_index = i;
            if (!indices.empty()) {
                real_index = indices[i];
            }
            mats[i - start_index] = images->at(real_index);
        }
        return apply(&mats, outputs);
    }

    bool RecognizerPreprocessor::apply(std::vector<cv::Mat>* image_batch,
                                       std::vector<Tensor>* outputs) const {
        const int img_h = rec_image_shape_[1];
        const int img_w = rec_image_shape_[2];
        float max_wh_ratio = static_cast<float>(img_w) * 1.0f / static_cast<float>(img_h);
        for (const auto& mat : *image_batch) {
            float ori_wh_ratio = static_cast<float>(mat.cols) * 1.0f / static_cast<float>(mat.rows);
            max_wh_ratio = std::max(max_wh_ratio, ori_wh_ratio);
        }

        for (auto& image : *image_batch) {
            cv::Mat* mat = &image;
            ocr_recognizer_resize_image(mat, max_wh_ratio,
                                        rec_image_shape_, static_shape_infer_);
            (*normalize_permute_op_)(mat);
        }
        // Only have 1 output Tensor.
        outputs->resize(1);
        utils::mats_to_tensor(*image_batch, &(*outputs)[0]);
        return true;
    }
}
