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
        runtime_option.set_model_path(model_file);
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


    bool Recognizer::predict(const ImageData& image,
                             std::string* text,
                             float* rec_score,
                             TimerArray* timers) {
        std::vector<std::string> texts(1);
        std::vector<float> rec_scores(1);
        if (const bool success = batch_predict({image}, &texts, &rec_scores, timers); !success) {
            return success;
        }
        *text = std::move(texts[0]);
        *rec_score = rec_scores[0];
        return true;
    }

    bool Recognizer::predict(const ImageData& image,
                             OCRResult* ocr_result,
                             TimerArray* timers) {
        ocr_result->text.resize(1);
        ocr_result->rec_scores.resize(1);
        if (!predict(image, &ocr_result->text[0], &ocr_result->rec_scores[0], timers)) {
            return false;
        }
        return true;
    }

    bool Recognizer::batch_predict(const std::vector<ImageData>& images,
                                   std::vector<std::string>* texts,
                                   std::vector<float>* rec_scores,
                                   TimerArray* timers) {
        return batch_predict(images, texts, rec_scores, 0, images.size(), {}, timers);
    }

    bool Recognizer::batch_predict(const std::vector<ImageData>& images,
                                   OCRResult* ocr_result,
                                   TimerArray* timers) {
        return batch_predict(images, &ocr_result->text, &ocr_result->rec_scores, timers);
    }

    bool Recognizer::batch_predict(const std::vector<ImageData>& images,
                                   std::vector<std::string>* texts,
                                   std::vector<float>* rec_scores,
                                   const size_t start_index,
                                   const size_t end_index,
                                   const std::vector<int>& indices,
                                   TimerArray* timers) {
        const size_t total_size = images.size();
        if (!indices.empty() && indices.size() != total_size) {
            MD_LOG_ERROR << "indices.size() should be 0 or " << images.size() << "." << std::endl;
            return false;
        }
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(images, &reused_input_tensors_, start_index,
                               end_index, indices)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }
        if (timers) timers->pre_timer.stop();
        if (timers) timers->infer_timer.start();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        if (!postprocessor_.run(reused_output_tensors_, texts, rec_scores,
                                start_index, total_size, indices)) {
            MD_LOG_ERROR << "Failed to postprocess the inference cls_results by runtime." << std::endl;
            return false;
        }
        if (timers) timers->post_timer.stop();
        return true;
    }
}
