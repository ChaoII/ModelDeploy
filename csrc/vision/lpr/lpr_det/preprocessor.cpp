//
// Created by aichao on 2025/6/10.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/lpr/lpr_det/preprocessor.h"

namespace modeldeploy::vision::lpr {
    LprDetPreprocessor::LprDetPreprocessor() {
        size_ = {640, 640};
        padding_value_ = {114.0, 114.0, 114.0};
        is_mini_pad_ = false;
        is_no_pad_ = false;
        is_scale_up_ = true;
        stride_ = 32;
    }


    bool LprDetPreprocessor::preprocess(const ImageData* image, Tensor* output, LetterBoxRecord* letter_box_record) const {
        // yolov8's preprocess steps
        // 1. letterbox
        // 2. convert_and_permute(swap_rb=true)
        const int src_w = image->width();
        const int src_h = image->height();
        *letter_box_record = utils::cal_letter_box_param({src_w, src_h}, size_);
        const float pad_x = static_cast<float>(letter_box_record->pad_w);
        const float pad_y = static_cast<float>(letter_box_record->pad_h);
        const float scale = letter_box_record->scale;
        const std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector<float> beta = {0.0f, 0.0f, 0.0f};
        if (!backend_->fused_preprocess(*image, output, size_,
                                        pad_x, pad_y, scale, scale,
                                        alpha, beta, true, padding_value_[0])) return false;
        return true;
    }

    bool LprDetPreprocessor::run(
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
            // 修改了数据，并生成一个tensor,并记录预处理的一些参数，便于在后处理中还原
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
