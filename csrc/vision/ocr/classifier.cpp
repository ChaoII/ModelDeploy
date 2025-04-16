//
// Created by aichao on 2025/2/21.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/ocr/classifier.h"
#include "csrc/vision/ocr/utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    Classifier::Classifier() = default;

    Classifier::Classifier(const std::string& model_file,
                           const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = initialize();
    }

    bool Classifier::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }

        return true;
    }


    bool Classifier::predict(const cv::Mat& img, int32_t* cls_label,
                             float* cls_score) {
        std::vector<int32_t> cls_labels(1);
        std::vector<float> cls_scores(1);
        if (const bool success = batch_predict({img}, &cls_labels, &cls_scores); !success) {
            return success;
        }
        *cls_label = cls_labels[0];
        *cls_score = cls_scores[0];
        return true;
    }

    bool Classifier::predict(const cv::Mat& img, OCRResult* ocr_result) {
        ocr_result->cls_labels.resize(1);
        ocr_result->cls_scores.resize(1);
        if (!predict(img, &ocr_result->cls_labels[0],
                     &ocr_result->cls_scores[0])) {
            return false;
        }
        return true;
    }

    bool Classifier::batch_predict(const std::vector<cv::Mat>& images,
                                   OCRResult* ocr_result) {
        return batch_predict(images, &ocr_result->cls_labels,
                             &ocr_result->cls_scores);
    }

    bool Classifier::batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<int32_t>* cls_labels,
                                   std::vector<float>* cls_scores) {
        return batch_predict(images, cls_labels, cls_scores, 0, images.size());
    }

    bool Classifier::batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<int32_t>* cls_labels,
                                   std::vector<float>* cls_scores,
                                   const size_t start_index, const size_t end_index) {
        const size_t total_size = images.size();
        const std::vector<cv::Mat>& images_ = images;
        if (!preprocessor_.run(&images_, &reused_input_tensors_, start_index, end_index)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }
        if (!postprocessor_.run(reused_output_tensors_, cls_labels, cls_scores,
                                start_index, total_size)) {
            MD_LOG_ERROR << "Failed to postprocess the inference cls_results by runtime." << std::endl;
            return false;
        }
        return true;
    }
} // namespace ocr
