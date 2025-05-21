//
// Created by aichao on 2025/3/26.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"
#include "csrc/vision/common/processors/center_crop.h"
#include "csrc/vision/face/face_as/face_as_first.h"


namespace modeldeploy::vision::face {
    SeetaFaceAsFirst::SeetaFaceAsFirst(const std::string& model_file,
                                       const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = Initialize();
    }

    bool SeetaFaceAsFirst::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy backend." << std::endl;
            return false;
        }
        return true;
    }

    bool SeetaFaceAsFirst::preprocess(cv::Mat* mat, Tensor* output) {
        if (mat->rows == 256 && mat->cols == 256) {
            CenterCrop::apply(mat, size_[0], size_[1]);
        }
        else if (mat->rows == size_[0] && mat->cols == size_[1]) {
            MD_LOG_WARN << "the width and height is already to [" << size_[0] << "," << size_[1] << "] " << std::endl;
        }
        else {
            MD_LOG_WARN << "the size of shape must be 256, ensure use face alignment? "
                "now, resize to 256 and may loss predict precision" << std::endl;
            Resize::apply(mat, 256, 256);
            CenterCrop::apply(mat, size_[0], size_[1]);
        }
        cv::cvtColor(*mat, *mat, cv::COLOR_BGR2YCrCb);
        HWC2CHW::apply(mat);
        Cast::apply(mat, "float");
        if (!utils::mat_to_tensor(*mat, output)) {
            MD_LOG_ERROR << "Failed to binding mat to tensor." << std::endl;
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }


    bool SeetaFaceAsFirst::postprocess(const std::vector<Tensor>& infer_result, float* result) {
        const Tensor& infer_result_ = infer_result[0];
        *result = static_cast<const float*>(infer_result_.data())[1];
        return true;
    }

    bool SeetaFaceAsFirst::predict(cv::Mat& im, float* result) {
        std::vector<Tensor> input_tensors(1);
        if (!preprocess(&im, &input_tensors[0])) {
            MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
            return false;
        }
        input_tensors[0].set_name(get_input_info(0).name);
        std::vector<Tensor> output_tensors;
        if (!infer(input_tensors, &output_tensors)) {
            MD_LOG_ERROR << "Failed to inference." << std::endl;
            return false;
        }
        postprocess(output_tensors, result);
        return true;
    }
}
