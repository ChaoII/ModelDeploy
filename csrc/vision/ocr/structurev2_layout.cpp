//
// Created by aichao on 2025/3/21.
//


#include "core/md_log.h"
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/ocr/structurev2_layout.h"


namespace modeldeploy::vision::ocr {
    StructureV2Layout::StructureV2Layout(const std::string& model_file,
                                         const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = Initialize();
    }

    bool StructureV2Layout::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        return true;
    }


    bool StructureV2Layout::predict(const ImageData& image, std::vector<DetectionResult>* result) {
        std::vector<std::vector<DetectionResult>> results;
        if (!batch_predict({image}, &results)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool StructureV2Layout::batch_predict(const std::vector<ImageData>& images,
                                          std::vector<std::vector<DetectionResult>>* results) {
        std::vector<ImageData> _images = images;
        if (!preprocessor_.run(&_images, &reused_input_tensors_)) {
            MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
            return false;
        }
        const auto batch_layout_img_info = preprocessor_.get_batch_layout_image_info();

        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }

        if (!postprocessor_.run(reused_output_tensors_, results, *batch_layout_img_info)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::ocr
