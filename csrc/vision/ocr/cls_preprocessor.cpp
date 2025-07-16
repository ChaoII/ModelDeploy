//
// Created by aichao on 2025/2/21.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/ocr/cls_preprocessor.h"
#include "vision/ocr/utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    ClassifierPreprocessor::ClassifierPreprocessor() {
        resize_op_ = std::make_shared<Resize>(-1, -1);
        std::vector<float> value = {0, 0, 0};
        pad_op_ = std::make_shared<Pad>(0, 0, 0, 0, value);
        normalize_op_ =
            std::make_shared<Normalize>(std::vector({0.5f, 0.5f, 0.5f}),
                                        std::vector({0.5f, 0.5f, 0.5f}), true);
        hwc2chw_op_ = std::make_shared<HWC2CHW>();
    }

    void ClassifierPreprocessor::ocr_classifier_resize_image(
        cv::Mat* mat, const std::vector<int>& cls_image_shape) const {
        const int img_h = cls_image_shape[1];
        const int img_w = cls_image_shape[2];
        const float ratio = static_cast<float>(mat->cols) / static_cast<float>(mat->rows);
        int resize_w;
        if (ceilf(static_cast<float>(img_h) * ratio) > static_cast<float>(img_w))
            resize_w = img_w;
        else
            resize_w = static_cast<int>(ceilf(static_cast<float>(img_h) * ratio));

        resize_op_->set_width_and_height(resize_w, img_h);
        (*resize_op_)(mat);
    }

    bool ClassifierPreprocessor::run(const std::vector<cv::Mat>* images,
                                     std::vector<Tensor>* outputs,
                                     const size_t start_index, const size_t end_index) {
        if (static_cast<int>(images->size()) == 0 || end_index <= start_index ||
            end_index > images->size()) {
            MD_LOG_ERROR << "images->size() or index error. Correct is: 0 <= start_index < "
                "end_index <= images->size()" << std::endl;
            return false;
        }
        std::vector<cv::Mat> mats(end_index - start_index);
        for (size_t i = start_index; i < end_index; ++i) {
            mats[i - start_index] = images->at(i);
        }
        return apply(&mats, outputs);
    }

    bool ClassifierPreprocessor::apply(std::vector<cv::Mat>* image_batch,
                                       std::vector<Tensor>* outputs) {
        for (auto& image : *image_batch) {
            cv::Mat* mat = &image;
            ocr_classifier_resize_image(mat, cls_image_shape_);
            (*normalize_op_)(mat);
            std::vector<float> value = {0, 0, 0};
            if (mat->cols < cls_image_shape_[2]) {
                pad_op_->set_padding_size(0, 0, 0, cls_image_shape_[2] - mat->cols);
                (*pad_op_)(mat);
            }
            (*hwc2chw_op_)(mat);
        }
        // Only have 1 output tensor.
        outputs->resize(1);
        // Get the NCHW tensor
        utils::mats_to_tensor(*image_batch, &(*outputs)[0]);

        return true;
    }
}
