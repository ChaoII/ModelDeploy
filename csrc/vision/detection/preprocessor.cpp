//
// Created by aichao on 2025/2/20.
//

#include "core/md_log.h"
#include "vision/detection/preprocessor.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/pad.h"
#include "vision/common/processors/convert_and_permute.h"

namespace modeldeploy::vision::detection {
    UltralyticsPreprocessor::UltralyticsPreprocessor() {
        size_ = {640, 640};
        padding_value_ = {114.0, 114.0, 114.0};
        is_mini_pad_ = false;
        is_no_pad_ = false;
        is_scale_up_ = true;
        stride_ = 32;
    }


    bool UltralyticsPreprocessor::preprocess(ImageData* image, Tensor* output,
                                             LetterBoxRecord* letter_box_record) const {
        // yolov8's preprocess steps
        // 1. letterbox
        // 2. convert_and_permute(swap_rb=true)
        cv::Mat mat;
        image->to_mat(&mat);
        utils::letter_box(&mat, size_, is_scale_up_, is_mini_pad_, is_no_pad_,
                          padding_value_, stride_, letter_box_record);
        const std::vector alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector beta = {0.0f, 0.0f, 0.0f};
        ConvertAndPermute::apply(&mat, alpha, beta, true);
        utils::mat_to_tensor(mat, output, true);
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool UltralyticsPreprocessor::run(
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
