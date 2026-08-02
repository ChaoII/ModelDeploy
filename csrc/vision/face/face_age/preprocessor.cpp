//
// Created by aichao on 2025/3/24.
//

#include "core/md_log.h"
#include "vision/face/face_age/preprocessor.h"


namespace modeldeploy::vision::face {
    bool SeetaFaceAgePreprocessor::preprocess(const ImageData& image, Tensor* output) const {
        // 经过人脸对齐后[256, 256]的图像,不需要BGR2RGB，不需要Normalize
        // 1. CenterCrop [256,256]->[248,248]
        // 2. Cast (uint8->float, 不缩放)
        // 3. HWC2CHW
        if (image.empty() || image.channels() != 3) {
            MD_LOG_ERROR << "The input image must be a color image." << std::endl;
            return false;
        }
        const ImageData* in = &image;
        ImageData resized_buf;
        bool already_target = false;
        if (image.height() == 256 && image.width() == 256) {
            // 直接用，下一步 center_crop 248
        } else if (image.height() == size_[0] && image.width() == size_[1]) {
            // 已 248：恒等直通（仅 cast + hwc2chw）
            already_target = true;
            MD_LOG_WARN << "the width and height is already to " << size_[0] << " and  " << size_[1] << std::endl;
        } else {
            // resize 到 256
            MD_LOG_WARN << "the size of shape must be 256, ensure use face alignment? "
                "now, resize to 256 and may loss predict precision." << std::endl;
            if (!backend_->resize(image, &resized_buf, 256, 256)) return false;
            in = &resized_buf;
        }
        float origin, scale;
        if (already_target) {
            origin = 0.0f;
            scale = 1.0f;
        } else {
            // center_crop 248 from 256：dst -> src = (dst + 4) * (src_dim/256)
            scale = 256.0f / in->width();
            origin = -4.0f;
        }
        if (!backend_->fused_preprocess(*in, output, size_,
                                        origin, origin, scale, scale,
                                        {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                                        false, 0.0f)) return false;
        return true;
    }

    bool SeetaFaceAgePreprocessor::run(const std::vector<ImageData>& images,
                                       std::vector<Tensor>* outputs) const {
        if (images.empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        outputs->resize(1);
        if (images.size() == 1) {
            // 单图：直接写进持久 outputs[0]，其 allocate 跨帧复用显存 buffer
            if (!preprocess(images[0], &(*outputs)[0])) {
                MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
                return false;
            }
            return true;
        }
        // Concat all the preprocessed data to a batch tensor
        std::vector<Tensor> tensors(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
            if (!preprocess(images[i], &tensors[i])) {
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
} // namespace modeldeploy::vision::face_rec
