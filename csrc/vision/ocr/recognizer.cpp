//
// Created by aichao on 2025/2/21.
//

#include "core/md_log.h"
#include "vision/ocr/recognizer.h"
#include "vision/ocr/utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    Recognizer::Recognizer(const std::string& model_file,
                           const std::string& label_path,
                           const RuntimeOption& custom_option)
        : postprocessor_(label_path) {
        runtime_option = custom_option;
        runtime_option.model_file = model_file;
        initialized_ = initialize();
    }

    // Init
    bool Recognizer::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy backend." << std::endl;
            return false;
        }
        return true;
    }


    bool Recognizer::predict(const cv::Mat& img, std::string* text,
                             float* rec_score) {
        std::vector<std::string> texts(1);
        std::vector<float> rec_scores(1);
        if (const bool success = batch_predict({img}, &texts, &rec_scores); !success) {
            return success;
        }
        *text = std::move(texts[0]);
        *rec_score = rec_scores[0];
        return true;
    }

    bool Recognizer::predict(const cv::Mat& img, OCRResult* ocr_result) {
        ocr_result->text.resize(1);
        ocr_result->rec_scores.resize(1);
        if (!predict(img, &ocr_result->text[0], &ocr_result->rec_scores[0])) {
            return false;
        }
        return true;
    }

    bool Recognizer::batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<std::string>* texts,
                                   std::vector<float>* rec_scores) {
        return batch_predict(images, texts, rec_scores, 0, images.size(), {});
    }

    bool Recognizer::batch_predict(const std::vector<cv::Mat>& images,
                                   OCRResult* ocr_result) {
        return batch_predict(images, &ocr_result->text, &ocr_result->rec_scores);
    }

    bool Recognizer::batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<std::string>* texts,
                                   std::vector<float>* rec_scores,
                                   const size_t start_index, const size_t end_index,
                                   const std::vector<int>& indices) {
        const size_t total_size = images.size();
        if (!indices.empty() && indices.size() != total_size) {
            MD_LOG_ERROR << "indices.size() should be 0 or " << images.size() << "." << std::endl;
            return false;
        }
        const std::vector<cv::Mat>& images_ = images;
        if (!preprocessor_.run(&images_, &reused_input_tensors_, start_index,
                               end_index, indices)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }
        if (!postprocessor_.run(reused_output_tensors_, texts, rec_scores,
                                start_index, total_size, indices)) {
            MD_LOG_ERROR << "Failed to postprocess the inference cls_results by runtime." << std::endl;
            return false;
        }
        return true;
    }
}
