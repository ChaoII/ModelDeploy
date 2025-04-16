//
// Created by aichao on 2025/3/21.
//

#include "csrc/vision/ocr/structurev2_layout.h"
#include "csrc/vision/ocr/utils/ocr_utils.h"


namespace modeldeploy::vision::ocr {
    StructureV2Layout::StructureV2Layout(const std::string& model_file,
                                         const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = Initialize();
    }

    bool StructureV2Layout::Initialize() {
        if (!init_runtime()) {
            std::cerr << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        return true;
    }


    bool StructureV2Layout::predict(cv::Mat* im, DetectionResult* result) {
        return predict(*im, result);
    }

    bool StructureV2Layout::predict(const cv::Mat& im, DetectionResult* result) {
        std::vector<DetectionResult> results;
        if (!batch_predict({im}, &results)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool StructureV2Layout::batch_predict(const std::vector<cv::Mat>& images,
                                          std::vector<DetectionResult>* results) {
        std::vector<cv::Mat> images_ = images;
        if (!preprocessor_.run(&images_, &reused_input_tensors_)) {
            std::cerr << "Failed to preprocess input image." << std::endl;
            return false;
        }
        auto batch_layout_img_info = preprocessor_.get_batch_layout_image_info();

        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            std::cerr << "Failed to inference by runtime." << std::endl;
            return false;
        }

        if (!postprocessor_.run(reused_output_tensors_, results,
                                *batch_layout_img_info)) {
            std::cerr << "Failed to postprocess the inference results." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::ocr
