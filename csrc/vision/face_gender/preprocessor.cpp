//
// Created by aichao on 2025/3/24.
//
#include "csrc/vision/face_gender/preprocessor.h"
#include <csrc/core/md_log.h>
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"


namespace modeldeploy::vision::faceid {
    bool SeetaFaceGenderPreprocessor::preprocess(cv::Mat* mat, MDTensor* output) {
        // 1. Resize
        // 2. HWC2CHW
        // 3. Cast
        Resize::Run(mat, size_[0], size_[1]);
        // BGR2RGB::Run(mat); 前处理不需要转换为RGB
        HWC2CHW::Run(mat);
        Cast::Run(mat, "float");
        if (!utils::mat_to_tensor(*mat, output)) {
            MD_LOG_ERROR("Failed to binding mat to tensor.");
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool SeetaFaceGenderPreprocessor::run(std::vector<cv::Mat>* images,
                                          std::vector<MDTensor>* outputs) {
        if (images->empty()) {
            MD_LOG_ERROR("The size of input images should be greater than 0.");
            return false;
        }
        if (images->size() != 1) {
            MD_LOG_ERROR("Only support batch = 1 now.");
        }
        outputs->resize(1);
        // Concat all the preprocessed data to a batch tensor
        std::vector<MDTensor> tensors(images->size());
        for (size_t i = 0; i < images->size(); ++i) {
            if (!preprocess(&(*images)[i], &tensors[i])) {
                MD_LOG_ERROR("Failed to preprocess input image.");
                return false;
            }
        }
        (*outputs)[0] = std::move(tensors[0]);
        return true;
    }
} // namespace modeldeploy::vision::face_id
