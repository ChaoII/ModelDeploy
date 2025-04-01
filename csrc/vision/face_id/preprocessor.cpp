//
// Created by aichao on 2025/3/24.
//
#include "csrc/vision/face_id/preprocessor.h"

#include <csrc/core/md_log.h>

#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/color_space_convert.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"
#include "csrc/vision/common/processors/center_crop.h"


namespace modeldeploy::vision::face {
    bool SeetaFaceIDPreprocessor::preprocess(cv::Mat* mat, MDTensor* output) {
        // 经过人脸对齐后[256, 256]的图像
        // 1. CenterCrop [256,256]->[248,248]
        // 2. BGR2RGB
        // 3. HWC2CHW
        // 4. Cast
        if (mat->rows != 256 || mat->cols != 256) {
            MD_LOG_WARN <<
                "the size of shape must be 256, ensure use face alignment? "
                "now, resize to 256 and may loss precision" << std::endl;
            Resize::Run(mat, 256, 256);
        }
        CenterCrop::Run(mat, size_[0], size_[1]);
        BGR2RGB::Run(mat);
        HWC2CHW::Run(mat);
        Cast::Run(mat, "float");
        if (!utils::mat_to_tensor(*mat, output)) {
            MD_LOG_ERROR << "Failed to binding mat to tensor." << std::endl;
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool SeetaFaceIDPreprocessor::run(std::vector<cv::Mat>* images,
                                      std::vector<MDTensor>* outputs) {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        if (images->size() != 1) {
            MD_LOG_ERROR << "Only support batch = 1 now." << std::endl;
        }
        outputs->resize(1);
        // Concat all the preprocessed data to a batch tensor
        std::vector<MDTensor> tensors(images->size());
        for (size_t i = 0; i < images->size(); ++i) {
            if (!preprocess(&(*images)[i], &tensors[i])) {
                MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
                return false;
            }
        }
        (*outputs)[0] = std::move(tensors[0]);
        return true;
    }
} // namespace modeldeploy::vision::face_id
