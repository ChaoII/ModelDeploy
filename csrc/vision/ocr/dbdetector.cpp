//
// Created by aichao on 2025/2/21.
//


#include "dbdetector.h"
#include "./utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    DBDetector::DBDetector() {
    }

    DBDetector::DBDetector(const std::string& model_file,
                           const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = initialize();
    }

    // Init
    bool DBDetector::initialize() {
        if (!init_runtime()) {
            std::cerr << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        return true;
    }


    bool DBDetector::predict(const cv::Mat& img,
                             std::vector<std::array<int, 8>>* boxes_result) {
        std::vector<std::vector<std::array<int, 8>>> det_results;
        if (!batch_predict({img}, &det_results)) {
            return false;
        }
        *boxes_result = std::move(det_results[0]);
        return true;
    }

    bool DBDetector::predict(const cv::Mat& img, OCRResult* ocr_result) {
        if (!predict(img, &(ocr_result->boxes))) {
            return false;
        }
        return true;
    }

    bool DBDetector::batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<OCRResult>* ocr_results) {
        std::vector<std::vector<std::array<int, 8>>> det_results;
        if (!batch_predict(images, &det_results)) {
            return false;
        }
        ocr_results->resize(det_results.size());
        for (int i = 0; i < det_results.size(); i++) {
            (*ocr_results)[i].boxes = std::move(det_results[i]);
        }
        return true;
    }

    bool DBDetector::batch_predict(
        const std::vector<cv::Mat>& images,
        std::vector<std::vector<std::array<int, 8>>>* det_results) {
        std::vector<cv::Mat> imgs = images;
        if (!preprocessor_.apply(&imgs, &reused_input_tensors_)) {
            std::cerr << "Failed to preprocess input image." << std::endl;
            return false;
        }
        const auto batch_det_img_info = preprocessor_.get_batch_img_info();
        reused_input_tensors_[0].name = get_input_info(0).name;

        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            std::cerr << "Failed to inference by runtime." << std::endl;
            return false;
        }
        if (!postprocessor_.apply(reused_output_tensors_,
                                  det_results, *batch_det_img_info)) {
            std::cerr << "Failed to postprocess the inference cls_results by runtime." << std::endl;
            return false;
        }
        return true;
    }
}
