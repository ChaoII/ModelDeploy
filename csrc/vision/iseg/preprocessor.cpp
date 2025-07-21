//
// Created by aichao on 2025/4/14.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/iseg/preprocessor.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/pad.h"
#include "vision/common/processors/convert_and_permute.h"

namespace modeldeploy::vision::detection {
    UltralyticsSegPreprocessor::UltralyticsSegPreprocessor() {
        size_ = {640, 640};
        padding_value_ = {114.0, 114.0, 114.0};
        is_mini_pad_ = false;
        is_no_pad_ = false;
        is_scale_up_ = true;
        stride_ = 32;
    }

    void UltralyticsSegPreprocessor::letter_box(ImageData* image, LetterBoxRecord* letter_box_record) const {
        cv::Mat mat;
        image->to_mat(&mat);

        letter_box_record->ipt_h = static_cast<float>(mat.rows);
        letter_box_record->ipt_w = static_cast<float>(mat.cols);
        auto scale = std::min(size_[1] * 1.0 / mat.rows, size_[0] * 1.0 / mat.cols);
        if (!is_scale_up_) {
            scale = std::min(scale, 1.0);
        }
        int resize_h = static_cast<int>(round(mat.rows * scale));
        int resize_w = static_cast<int>(round(mat.cols * scale));
        int pad_w = size_[0] - resize_w;
        int pad_h = size_[1] - resize_h;
        if (is_mini_pad_) {
            pad_h = pad_h % stride_;
            pad_w = pad_w % stride_;
        }
        else if (is_no_pad_) {
            pad_h = 0;
            pad_w = 0;
            resize_h = size_[1];
            resize_w = size_[0];
        }
        if (std::fabs(scale - 1.0f) > 1e-06) {
            Resize::apply(&mat, resize_w, resize_h);
        }
        if (pad_h > 0 || pad_w > 0) {
            const float half_h = static_cast<float>(pad_h) * 1.0f / 2;
            const int top = static_cast<int>(round(half_h - 0.1));
            const int bottom = static_cast<int>(round(half_h + 0.1));
            const float half_w = static_cast<float>(pad_w) * 1.0f / 2;
            const int left = static_cast<int>(round(half_w - 0.1));
            const int right = static_cast<int>(round(half_w + 0.1));
            Pad::apply(&mat, top, bottom, left, right, padding_value_);
        }
        letter_box_record->out_h = static_cast<float>(mat.rows);
        letter_box_record->out_w = static_cast<float>(mat.cols);
        letter_box_record->pad_h = static_cast<float>(pad_h) / 2.0f;
        letter_box_record->pad_w = static_cast<float>(pad_w) / 2.0f;
        letter_box_record->scale = static_cast<float>(scale);
    }

    bool UltralyticsSegPreprocessor::preprocess(
        ImageData* image, Tensor* output,
        LetterBoxRecord* letter_box_record) const {
        cv::Mat mat;
        image->to_mat(&mat);

        // Record the shape of image and the shape of preprocessed image
        // yolov5seg's preprocess steps
        // 1. letterbox
        // 2. convert_and_permute(swap_rb=true)
        letter_box(image, letter_box_record);
        const std::vector alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector beta = {0.0f, 0.0f, 0.0f};
        ConvertAndPermute::apply(&mat, alpha, beta, true);
        utils::mat_to_tensor(mat, output);
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool UltralyticsSegPreprocessor::run(
        std::vector<ImageData>* images, std::vector<Tensor>* outputs,
        std::vector<LetterBoxRecord>* letter_box_records) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        letter_box_records->resize(images->size());
        outputs->resize(1);
        // Concat all the preprocessed data to a batch tensor
        std::vector<Tensor> tensors(images->size());
        for (size_t i = 0; i < images->size(); ++i) {
            preprocess(&(*images)[i], &tensors[i], &(*letter_box_records)[i]);
        }
        if (tensors.size() == 1) {
            (*outputs)[0] = std::move(tensors[0]);
        }
        else {
            (*outputs)[0] = std::move(Tensor::concat(tensors, 0));
        }
        return true;
    }
}
