//
// Created by aichao on 2025/3/24.
//
#include "csrc/vision/faceid/preprocessor.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/color_space_convert.h"
#include "csrc/vision/common/processors/convert.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"


namespace modeldeploy::vision::faceid {
    AdaFacePreprocessor::AdaFacePreprocessor() {
        // parameters for preprocess
        size_ = {248, 248};
        alpha_ = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
        beta_ = {-1.f, -1.f, -1.f}; // RGB
        permute_ = true;
    }

    bool AdaFacePreprocessor::Preprocess(cv::Mat* mat, MDTensor* output) {
        // face recognition model's preprocess steps in insightface
        // reference: insightface/recognition/arcface_torch/inference.py
        // 1. Resize
        // 2. BGR2RGB
        // 3. Convert(opencv style) or Normalize
        // 4. HWC2CHW
        int resize_w = size_[0];
        int resize_h = size_[1];
        if (resize_h != mat->rows || resize_w != mat->cols) {
            Resize::Run(mat, resize_w, resize_h);
        }
        if (permute_) {
            BGR2RGB::Run(mat);
        }

        Convert::Run(mat, alpha_, beta_);
        HWC2CHW::Run(mat);
        Cast::Run(mat, "float");

        utils::mat_to_tensor(*mat, output);
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool AdaFacePreprocessor::Run(std::vector<cv::Mat>* images,
                                  std::vector<MDTensor>* outputs) {
        if (images->empty()) {
            std::cerr << "The size of input images should be greater than 0."
                << std::endl;
            return false;
        }
        if (images->size() != 1) {
            std::cerr << "Only support batch = 1 now.";
        }
        outputs->resize(1);
        // Concat all the preprocessed data to a batch tensor
        std::vector<MDTensor> tensors(images->size());
        for (size_t i = 0; i < images->size(); ++i) {
            if (!Preprocess(&(*images)[i], &tensors[i])) {
                std::cerr << "Failed to preprocess input image." << std::endl;
                return false;
            }
        }
        (*outputs)[0] = std::move(tensors[0]);
        return true;
    }
} // namespace modeldeploy::vision::faceid
