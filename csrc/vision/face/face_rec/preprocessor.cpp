//
// Created by aichao on 2025/3/24.
//

#include "core/md_log.h"
#include "vision/face/face_rec/preprocessor.h"


namespace modeldeploy::vision::face {
    bool SeetaFaceIDPreprocessor::preprocess(ImageData* image, Tensor* output) const {
        // 经过人脸对齐后[256, 256]的图像
        // 1. Resize 到 [256,256]（若非 256）
        // 2. CenterCrop [256,256]->[248,248]
        // 3. BGR2RGB
        // 4. Cast (uint8->float, 不缩放)
        // 5. HWC2CHW
        const ImageData* in = image;
        ImageData resized_buf;
        if (image->width() != 256 || image->height() != 256) {
            MD_LOG_WARN <<
                "the size of shape must be 256, ensure use face alignment? "
                "now, resize to 256 and may loss precision" << std::endl;
            if (!backend_->resize(*image, &resized_buf, 256, 256)) return false;
            in = &resized_buf;
        }
        const int src2_w = in->width();
        const float scale = 256.0f / src2_w;
        const float origin = -4.0f;
        if (!backend_->fused_preprocess(*in, output, size_,  // {248, 248}
                                        origin, origin, scale, scale,
                                        {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                                        true, 0.0f)) return false;  // swap_rb=true (BGR2RGB)
        return true;
    }

    bool SeetaFaceIDPreprocessor::run(std::vector<ImageData>* images,
                                      std::vector<Tensor>* outputs) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        outputs->resize(1);
        // Concat all the preprocessed data to a batch tensor
        std::vector<Tensor> tensors(images->size());
        for (size_t i = 0; i < images->size(); ++i) {
            // 修改了数据，并生成一个tensor,并记录预处理的一些参数，便于在后处理中还原
            preprocess(&(*images)[i], &tensors[i]);
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
