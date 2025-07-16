//
// Created by aichao on 2025/3/21.
//

#include "opencv2/opencv.hpp"
#include "vision/ocr/structurev2_table.h"

#include <core/md_log.h>

#include "vision/ocr/utils/ocr_utils.h"


namespace modeldeploy::vision::ocr {
    StructureV2Table::StructureV2Table() {
    }

    StructureV2Table::StructureV2Table(const std::string& model_file,
                                       const std::string& table_char_dict_path,
                                       const RuntimeOption& custom_option)
        : postprocessor_(table_char_dict_path) {
        runtime_option = custom_option;
        runtime_option.model_file = model_file;
        initialized_ = initialize();
    }

    // Init
    bool StructureV2Table::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        return true;
    }


    bool StructureV2Table::predict(const cv::Mat& img,
                                   std::vector<std::array<int, 8>>* boxes_result,
                                   std::vector<std::string>* structure_result) {
        std::vector<std::vector<std::array<int, 8>>> det_results;
        std::vector<std::vector<std::string>> structure_results;
        if (!batch_predict({img}, &det_results, &structure_results)) {
            return false;
        }
        *boxes_result = std::move(det_results[0]);
        *structure_result = std::move(structure_results[0]);
        return true;
    }

    bool StructureV2Table::predict(const cv::Mat& img,
                                   vision::OCRResult* ocr_result) {
        if (!predict(img, &ocr_result->table_boxes, &ocr_result->table_structure)) {
            return false;
        }
        return true;
    }

    bool StructureV2Table::batch_predict(
        const std::vector<cv::Mat>& images,
        std::vector<vision::OCRResult>* ocr_results) {
        std::vector<std::vector<std::array<int, 8>>> det_results;
        std::vector<std::vector<std::string>> structure_results;
        if (!batch_predict(images, &det_results, &structure_results)) {
            return false;
        }
        ocr_results->resize(det_results.size());
        for (int i = 0; i < det_results.size(); i++) {
            (*ocr_results)[i].table_boxes = std::move(det_results[i]);
            (*ocr_results)[i].table_structure = std::move(structure_results[i]);
        }
        return true;
    }

    bool StructureV2Table::batch_predict(
        const std::vector<cv::Mat>& images,
        std::vector<std::vector<std::array<int, 8>>>* det_results,
        std::vector<std::vector<std::string>>* structure_results) {
        std::vector<cv::Mat> images_ = images;
        if (!preprocessor_.run(&images_, &reused_input_tensors_)) {
            MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
            return false;
        }
        const auto batch_det_img_info = preprocessor_.GetBatchImgInfo();

        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }

        if (!postprocessor_.run(reused_output_tensors_, det_results,
                                structure_results, *batch_det_img_info)) {
            MD_LOG_ERROR << "Failed to postprocess the inference cls_results by runtime." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::ocr
