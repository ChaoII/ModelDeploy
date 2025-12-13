//
// Created by aichao on 2025/2/24.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/classification/preprocessor.h"
#include "vision/common/processors/center_crop.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/convert.h"
#include "vision/common/processors/color_space_convert.h"
#include "vision/common/processors/normalize_and_permute.h"

namespace modeldeploy::vision::classification {
    ClassificationPreprocessor::ClassificationPreprocessor() {
        size_ = {224, 224}; //{h,w}
    }

    bool ClassificationPreprocessor::preprocess(ImageData* image, Tensor* output) const {
        // yolov8-cls's preprocess steps
        // 1. CenterCrop
        // 2. Resize
        // 2. Normalize
        if (image->width() <= 0 || image->height() <= 0) {
            return false;
        }
        cv::Mat mat;
        image->to_mat(&mat);
        if (enable_center_crop_) {
            const int crop_size = std::min(mat.rows, mat.cols);
            CenterCrop::apply(&mat, crop_size, crop_size);
        }
        Resize::apply(&mat, size_[0], size_[1]);
        // Normalize
        BGR2RGB::apply(&mat);
        const std::vector alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector beta = {0.0f, 0.0f, 0.0f};
        Convert::apply(&mat, alpha, beta);
        const std::vector mean = {0.485f, 0.456f, 0.406f};
        const std::vector std = {0.229f, 0.224f, 0.225f};
        NormalizeAndPermute::apply(&mat, mean, std, false);
        utils::mat_to_tensor(mat, output);
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool ClassificationPreprocessor::run(
        std::vector<ImageData>* images, std::vector<Tensor>* outputs) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0."
                << std::endl;
            return false;
        }
        outputs->resize(1);
        // Concat all the preprocessed data to a batch tensor
        std::vector<Tensor> tensors(images->size());
        for (size_t i = 0; i < images->size(); ++i) {
            if (!preprocess(&(*images)[i], &tensors[i])) {
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
