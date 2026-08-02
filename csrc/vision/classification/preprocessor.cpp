//
// Created by aichao on 2025/2/24.
//

#include "core/md_log.h"
#include "vision/classification/preprocessor.h"

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
        ImageData tmp;
        if (enable_center_crop_) {
            const int crop_size = std::min(image->height(), image->width());
            if (!backend_->center_crop(*image, &tmp, crop_size, crop_size)) return false;
        } else {
            tmp = *image;
        }
        ImageData resized;
        if (!backend_->resize(tmp, &resized, size_[0], size_[1])) return false;
        ImageData rgb;
        if (!backend_->convert_to(resized, &rgb, "RGB")) return false;
        ImageData scaled;
        const std::vector alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector beta = {0.0f, 0.0f, 0.0f};
        if (!backend_->convert(rgb, &scaled, alpha, beta)) return false;
        const std::vector mean = {0.485f, 0.456f, 0.406f};
        const std::vector std = {0.229f, 0.224f, 0.225f};
        // scale=false：convert 已做过 1/255，normalize 不再缩放（与原 NormalizeAndPermute scale=false 一致）
        if (!backend_->normalize_and_permute(scaled, output, mean, std, false)) return false;
        output->expand_dim(0);
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
