//
// Created by aichao on 2025/3/26.
//

#include <csrc/core/md_log.h>

#include "csrc/vision/utils.h"
#include "csrc/utils/utils.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/color_space_convert.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"
#include "csrc/vision/common/processors/convert.h"
#include "csrc/vision/face/face_as/face_as_second.h"


namespace modeldeploy::vision::face {
    SeetaFaceAsSecond::SeetaFaceAsSecond(const std::string& model_file,
                                         const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.model_file = model_file;
        initialized_ = Initialize();
    }

    bool SeetaFaceAsSecond::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        return true;
    }

    bool SeetaFaceAsSecond::preprocess(cv::Mat* mat, Tensor* output) {
        const std::vector alpha_ = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
        const std::vector beta_ = {-1.0f, -1.0f, -1.0f};

        Resize::apply(mat, size_[0], size_[1]);
        Convert::apply(mat, alpha_, beta_);
        HWC2CHW::apply(mat);
        Cast::apply(mat, "float");

        if (!utils::mat_to_tensor(*mat, output)) {
            MD_LOG_ERROR << "Failed to binding mat to tensor." << std::endl;
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }


    bool SeetaFaceAsSecond::postprocess(
        const std::vector<Tensor>& infer_result, std::vector<std::tuple<int, float>>* result) {
        const auto& class_predictions = infer_result[0];
        const auto& box_encodings = infer_result[1];
        const size_t size = box_encodings.shape()[1];
        result->resize(size);
        for (int i = 0; i < size; ++i) {
            // 获取类别预测
            std::vector<float> class_pred;
            for (int j = 1; j < class_predictions.shape()[2]; ++j) {
                class_pred.push_back(class_predictions.at({0, i, j, 0}));
            }
            int label = argmax(class_pred) + 1;
            float score = class_predictions.at({0, i, label, 0});
            if (score < 0.8) {
                continue;
            }
            result->at(i) = {label, score};
        }
        return true;
    }

    bool SeetaFaceAsSecond::predict(cv::Mat& im, std::vector<std::tuple<int, float>>* result) {
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
