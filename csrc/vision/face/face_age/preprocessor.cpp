//
// Created by aichao on 2025/3/24.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/color_space_convert.h"
#include "vision/common/processors/hwc2chw.h"
#include "vision/common/processors/cast.h"
#include "vision/common/processors/center_crop.h"
#include "vision/face/face_age/preprocessor.h"


namespace modeldeploy::vision::face {
    bool SeetaFaceAgePreprocessor::preprocess(ImageData* image, Tensor* output) const {
        // 经过人脸对齐后[256, 256]的图像
        // 1. CenterCrop [256,256]->[248,248]
        // 2. HWC2CHW
        // 3. Cast
        cv::Mat mat;
        image->to_mat(&mat);
        if (mat.rows == 256 && mat.cols == 256) {
            CenterCrop::apply(image, size_[0], size_[1]);
        }
        else if (mat.rows == size_[0] && mat.cols == size_[1]) {
            MD_LOG_WARN << "the width and height is already to " << size_[0] << " and  " << size_[1] << std::endl;
        }
        else {
            MD_LOG_WARN << "the size of shape must be 256, ensure use face alignment? "
                "now, resize to 256 and may loss predict precision." << std::endl;
            Resize::apply(&mat, 256, 256);
            CenterCrop::apply(image, size_[0], size_[1]);
        }
        // BGR2RGB::Run(mat); 前处理不需要转换为RGB
        HWC2CHW::apply(&mat);
        Cast::apply(image, "float");
        if (!utils::mat_to_tensor(mat, output)) {
            MD_LOG_ERROR << "Failed to binding mat to tensor." << std::endl;
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool SeetaFaceAgePreprocessor::run(std::vector<ImageData>* images,
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
