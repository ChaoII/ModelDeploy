//
// Created by aichao on 2025/3/21.
//

#include "opencv2/opencv.hpp"
#include "csrc/vision/ocr/structurev2_table.h"
#include "csrc/vision/ocr/utils/ocr_utils.h"


namespace modeldeploy::vision::ocr {
    StructureV2Table::StructureV2Table() {
    }

    StructureV2Table::StructureV2Table(const std::string& model_file,
                                       const std::string& table_char_dict_path,
                                       const RuntimeOption& custom_option)
        : postprocessor_(table_char_dict_path) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = Initialize();
    }

    // Init
    bool StructureV2Table::Initialize() {
        if (!init_runtime()) {
            std::cerr << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        return true;
    }


    bool StructureV2Table::Predict(const cv::Mat& img,
                                   std::vector<std::array<int, 8>>* boxes_result,
                                   std::vector<std::string>* structure_result) {
        std::vector<std::vector<std::array<int, 8>>> det_results;
        std::vector<std::vector<std::string>> structure_results;
        if (!BatchPredict({img}, &det_results, &structure_results)) {
            return false;
        }
        *boxes_result = std::move(det_results[0]);
        *structure_result = std::move(structure_results[0]);
        return true;
    }

    bool StructureV2Table::Predict(const cv::Mat& img,
                                   vision::OCRResult* ocr_result) {
        if (!Predict(img, &(ocr_result->table_boxes),
                     &(ocr_result->table_structure))) {
            return false;
        }
        return true;
    }

    bool StructureV2Table::BatchPredict(
        const std::vector<cv::Mat>& images,
        std::vector<vision::OCRResult>* ocr_results) {
        std::vector<std::vector<std::array<int, 8>>> det_results;
        std::vector<std::vector<std::string>> structure_results;
        if (!BatchPredict(images, &det_results, &structure_results)) {
            return false;
        }
        ocr_results->resize(det_results.size());
        for (int i = 0; i < det_results.size(); i++) {
            (*ocr_results)[i].table_boxes = std::move(det_results[i]);
            (*ocr_results)[i].table_structure = std::move(structure_results[i]);
        }
        return true;
    }

    bool StructureV2Table::BatchPredict(
        const std::vector<cv::Mat>& images,
        std::vector<std::vector<std::array<int, 8>>>* det_results,
        std::vector<std::vector<std::string>>* structure_results) {
        std::vector<cv::Mat> images_ = images;
        if (!preprocessor_.Apply(&images_, &reused_input_tensors_)) {
            std::cerr << "Failed to preprocess input image." << std::endl;
            return false;
        }
        auto batch_det_img_info = preprocessor_.GetBatchImgInfo();

        reused_input_tensors_[0].name = get_input_info(0).name;
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            std::cerr << "Failed to inference by runtime." << std::endl;
            return false;
        }

        if (!postprocessor_.Run(reused_output_tensors_, det_results,
                                structure_results, *batch_det_img_info)) {
            std::cerr << "Failed to postprocess the inference cls_results by runtime." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::ocr
