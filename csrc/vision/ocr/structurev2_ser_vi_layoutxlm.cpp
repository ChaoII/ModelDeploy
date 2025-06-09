//
// Created by aichao on 2025/3/21.
//
#include "csrc/vision/ocr//structurev2_ser_vi_layoutxlm.h"

#include <csrc/core/md_log.h>


namespace modeldeploy::vision::ocr {
    StructureV2SERViLayoutXLMModel::StructureV2SERViLayoutXLMModel(
            const std::string &model_file, const RuntimeOption &custom_option) {
        runtime_option = custom_option;
        runtime_option.model_file = model_file;
        initialized_ = initialize();
    }


    bool StructureV2SERViLayoutXLMModel::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::ocr
