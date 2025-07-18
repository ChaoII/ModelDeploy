//
// Created by aichao on 2025/2/24.
//


#include "core/md_log.h"
#include "vision/classification/ultralytics_cls.h"


namespace modeldeploy::vision::classification {
    UltralyticsCls::UltralyticsCls(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool UltralyticsCls::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        return true;
    }

    bool UltralyticsCls::predict(const ImageData& im, ClassifyResult* result) {
        std::vector<ClassifyResult> results;
        if (!batch_predict({im}, &results)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool UltralyticsCls::batch_predict(const std::vector<ImageData>& images, std::vector<ClassifyResult>* results) {
        std::vector<cv::Mat> _images;
        for (const auto& image : images) {
            cv::Mat image_;
            image.to_mat(&image_);
            _images.push_back(image_);
        }
        if (!preprocessor_.run(&_images, &reused_input_tensors_)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }

        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }

        if (!postprocessor_.run(reused_output_tensors_, results)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        return true;
    }
} // namespace classification
