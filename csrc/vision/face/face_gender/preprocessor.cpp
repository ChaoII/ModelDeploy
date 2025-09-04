//
// Created by aichao on 2025/3/24.
//

#include "vision/face/face_gender/preprocessor.h"
#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/hwc2chw.h"
#include "vision/common/processors/cast.h"


namespace modeldeploy::vision::face {
    bool SeetaFaceGenderPreprocessor::preprocess(ImageData* image, Tensor* output) const {
        // 1. Resize
        // 2. HWC2CHW
        // 3. Cast
        cv::Mat mat;
        image->to_mat(&mat);
        Resize::apply(&mat, size_[0], size_[1]);
        // BGR2RGB::Run(mat); 前处理不需要转换为RGB
        HWC2CHW::apply(&mat);
        Cast::apply(&mat, "float");
        if (!utils::mat_to_tensor(mat, output)) {
            MD_LOG_ERROR << "Failed to binding mat to tensor." << std::endl;
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool SeetaFaceGenderPreprocessor::run(std::vector<ImageData>* images,
                                          std::vector<Tensor>* outputs) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
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
} // namespace modeldeploy::vision::face_rec
