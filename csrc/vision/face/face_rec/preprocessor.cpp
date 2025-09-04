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
#include "vision/face/face_rec/preprocessor.h"


namespace modeldeploy::vision::face {
    bool SeetaFaceIDPreprocessor::preprocess(ImageData* image, Tensor* output) const {
        // 经过人脸对齐后[256, 256]的图像
        // 1. CenterCrop [256,256]->[248,248]
        // 2. BGR2RGB
        // 3. HWC2CHW
        // 4. Cast
        cv::Mat mat;
        image->to_mat(&mat);
        if (mat.rows != 256 || mat.cols != 256) {
            MD_LOG_WARN <<
                "the size of shape must be 256, ensure use face alignment? "
                "now, resize to 256 and may loss precision" << std::endl;
            Resize::apply(&mat, 256, 256);
        }
        CenterCrop::apply(&mat, size_[0], size_[1]);
        BGR2RGB::apply(&mat);
        HWC2CHW::apply(&mat);
        Cast::apply(&mat, "float");
        if (!utils::mat_to_tensor(mat, output)) {
            MD_LOG_ERROR << "Failed to binding mat to tensor." << std::endl;
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
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
