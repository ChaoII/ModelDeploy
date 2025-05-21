//
// Created by aichao on 2025/3/21.
//

#include "csrc/vision/ocr/structurev2_table_preprocessor.h"

#include <csrc/core/md_log.h>

#include "csrc/vision/ocr/utils/ocr_utils.h"


namespace modeldeploy::vision::ocr {
    StructureV2TablePreprocessor::StructureV2TablePreprocessor() {
        resize_op_ = std::make_shared<Resize>(-1, -1);
        std::vector<float> value = {0, 0, 0};
        pad_op_ = std::make_shared<Pad>(0, 0, 0, 0, value);

        std::vector mean = {0.485f, 0.456f, 0.406f};
        std::vector std = {0.229f, 0.224f, 0.225f};
        normalize_op_ = std::make_shared<Normalize>(mean, std, true);
        hwc2chw_op_ = std::make_shared<HWC2CHW>();
    }

    void StructureV2TablePreprocessor::structure_v2_table_resize_image(cv::Mat* mat,
                                                                       const int batch_idx) {
        const auto width = static_cast<float>(mat->cols);
        const auto height = static_cast<float>(mat->rows);
        const float ratio = max_len / (std::max(height, width) * 1.0);
        const int resize_h = static_cast<int>(height * ratio);
        const int resize_w = static_cast<int>(width * ratio);

        resize_op_->set_width_and_height(resize_w, resize_h);
        (*resize_op_)(mat);

        (*normalize_op_)(mat);
        pad_op_->set_padding_size(0, max_len - resize_h, 0,
                                  max_len - resize_w);
        (*pad_op_)(mat);
        (*hwc2chw_op_)(mat);
        batch_det_img_info_[batch_idx] = {
            static_cast<int>(width), static_cast<int>(height), resize_w,
            resize_h
        };
    }

    bool StructureV2TablePreprocessor::run(std::vector<cv::Mat>* images,
                                           std::vector<Tensor>* outputs,
                                           size_t start_index, size_t end_index,
                                           const std::vector<int>& indices) {
        if (images->size() == 0 || end_index <= start_index ||
            end_index > images->size()) {
            MD_LOG_ERROR << "images->size() or index error. Correct is: 0 <= start_index < "
                "end_index <= images->size()" << std::endl;
            return false;
        }

        std::vector<cv::Mat> mats(end_index - start_index);
        for (size_t i = start_index; i < end_index; ++i) {
            size_t real_index = i;
            if (indices.size() != 0) {
                real_index = indices[i];
            }
            mats[i - start_index] = images->at(real_index);
        }
        return run(&mats, outputs);
    }

    bool StructureV2TablePreprocessor::run(std::vector<cv::Mat>* image_batch,
                                           std::vector<Tensor>* outputs) {
        batch_det_img_info_.clear();
        batch_det_img_info_.resize(image_batch->size());
        for (size_t i = 0; i < image_batch->size(); ++i) {
            cv::Mat* mat = &(image_batch->at(i));
            structure_v2_table_resize_image(mat, i);
        }

        // Only have 1 output Tensor.
        outputs->resize(1);
        // Get the NCHW tensor
        utils::mats_to_tensor(*image_batch, &(*outputs)[0]);
        return true;
    }
} // namespace modeldeploy::vision::ocr
