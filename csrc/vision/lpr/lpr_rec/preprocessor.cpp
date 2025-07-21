//
// Created by aichao on 2025/6/10.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/lpr/lpr_rec/preprocessor.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/pad.h"
#include "vision/common/processors/convert_and_permute.h"

namespace modeldeploy::vision::lpr {
    LprRecPreprocessor::LprRecPreprocessor() {
        size_ = {168, 48};
    }

    bool LprRecPreprocessor::preprocess(ImageData* image, Tensor* output) const {
        // preprocess steps
        // 1. Resize
        // 2. convert_and_permute(swap_rb=true)
        cv::Mat mat;
        image->to_mat(&mat);
        Resize::apply(&mat, size_[0], size_[1]);
        const std::vector alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector beta = {-0.588f, -0.588f, -0.588f};
        ConvertAndPermute::apply(&mat, alpha, beta, true);
        utils::mat_to_tensor(mat, output);
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool LprRecPreprocessor::run(
        std::vector<ImageData>* images, std::vector<Tensor>* outputs) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        outputs->resize(1);
        // Concat all the preprocessed data to a batch tensor
        std::vector<Tensor> tensors(images->size());
        for (size_t i = 0; i < images->size(); ++i) {
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
}
