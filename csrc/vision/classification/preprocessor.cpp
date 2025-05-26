//
// Created by aichao on 2025/2/24.
//

#include "preprocessor.h"

#include <csrc/core/md_log.h>

#include "../common/processors/center_crop.h"
#include "../common/processors/resize.h"
#include "../common/processors/convert.h"
#include "../common/processors/color_space_convert.h"
#include "../common/processors/normalize_and_permute.h"

namespace modeldeploy::vision::classification {
    YOLOv5ClsPreprocessor::YOLOv5ClsPreprocessor() {
        size_ = {224, 224}; //{h,w}
    }

    bool YOLOv5ClsPreprocessor::preprocess(
        cv::Mat* mat, Tensor* output,
        std::map<std::string, std::array<float, 2>>* im_info) const {
        if (mat->empty()) {
            MD_LOG_ERROR << "Input image is empty." << std::endl;
            return false;
        }

        // Record the shape of image and the shape of preprocessed image
        (*im_info)["input_shape"] = {
            static_cast<float>(mat->rows),
            static_cast<float>(mat->cols)
        };

        // process after image load
        double ratio = (size_[0] * 1.0) / std::max(static_cast<float>(mat->rows),
                                                   static_cast<float>(mat->cols));
        // yolov5cls's preprocess steps
        // 1. CenterCrop
        // 2. Normalize
        // CenterCrop
        const int crop_size = std::min(mat->rows, mat->cols);
        CenterCrop::apply(mat, crop_size, crop_size);
        Resize::apply(mat, size_[0], size_[1], -1, -1, cv::INTER_LINEAR);
        // Normalize
        BGR2RGB::apply(mat);
        const std::vector alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector beta = {0.0f, 0.0f, 0.0f};
        Convert::apply(mat, alpha, beta);
        const std::vector mean = {0.485f, 0.456f, 0.406f};
        const std::vector std = {0.229f, 0.224f, 0.225f};
        NormalizeAndPermute::apply(mat, mean, std, false);

        // Record output shape of preprocessed image
        (*im_info)["output_shape"] = {
            static_cast<float>(mat->rows),
            static_cast<float>(mat->cols)
        };

        utils::mat_to_tensor(*mat, output);
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool YOLOv5ClsPreprocessor::run(
        std::vector<cv::Mat>* images, std::vector<Tensor>* outputs,
        std::vector<std::map<std::string, std::array<float, 2>>>* ims_info) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0."
                << std::endl;
            return false;
        }
        ims_info->resize(images->size());
        outputs->resize(1);
        // Concat all the preprocessed data to a batch tensor
        std::vector<Tensor> tensors(images->size());
        for (size_t i = 0; i < images->size(); ++i) {
            if (!preprocess(&(*images)[i], &tensors[i], &(*ims_info)[i])) {
                MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
                return false;
            }
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
