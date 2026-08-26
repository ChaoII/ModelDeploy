//
// Created by aichao on 2025/3/26.
//

#include <core/md_log.h>

#include "vision/face/face_as/face_as_second.h"


namespace modeldeploy::vision::face {
    SeetaFaceAsSecond::SeetaFaceAsSecond(const std::string& model_file,
                                         const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = Initialize();
    }

    bool SeetaFaceAsSecond::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        backend_ = create_processor_backend(runtime_option.device, runtime_option.backend,
                                            runtime_option.device_id);
        return true;
    }

    bool SeetaFaceAsSecond::preprocess(ImageData* image, Tensor* output) {
        // Resize(直接拉伸到 size_) + Convert(alpha=1/128, beta=-1) + HWC2CHW + Cast(float)
        // 全融合为单步 SIMD kernel。映射 src=(dst-origin)/scale，拉伸时 scale=src/dst。
        const int src_w = image->width();
        const int src_h = image->height();
        const float scale_x = static_cast<float>(src_w) / size_[0];
        const float scale_y = static_cast<float>(src_h) / size_[1];
        const std::vector<float> alpha = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
        const std::vector<float> beta = {-1.0f, -1.0f, -1.0f};
        if (!backend_->fused_preprocess_common(*image, output, size_,
                                        0.0f, 0.0f, scale_x, scale_y,
                                        alpha, beta, false, 0.0f)) {
            MD_LOG_ERROR << "Failed to fused preprocess." << std::endl;
            return false;
        }
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

    bool SeetaFaceAsSecond::predict(const ImageData& image, std::vector<std::tuple<int, float>>* result) {
        std::vector<Tensor> input_tensors(1);
        auto _image = image;
        if (!preprocess(&_image, &input_tensors[0])) {
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

    std::unique_ptr<SeetaFaceAsSecond> SeetaFaceAsSecond::clone() const {
        auto clone_model = std::make_unique<SeetaFaceAsSecond>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }
}

