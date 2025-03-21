//
// Created by aichao on 2025/3/21.
//
#include "csrc/vision/ocr//structurev2_ser_vi_layoutxlm.h"


namespace modeldeploy::vision::ocr {
    StructureV2SERViLayoutXLMModel::StructureV2SERViLayoutXLMModel(
        const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = Initialize();
    }


    bool StructureV2SERViLayoutXLMModel::Initialize() {
        if (!init_runtime()) {
            std::cerr << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::ocr
