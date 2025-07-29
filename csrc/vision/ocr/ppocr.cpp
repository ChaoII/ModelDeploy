//
// Created by aichao on 2025/2/21.
//

#include "core/md_log.h"
#include "vision/ocr/ppocr.h"
#include "vision/ocr/utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    PaddleOCR::PaddleOCR(const std::string& det_model_path,
                         const std::string& cls_model_path,
                         const std::string& rec_model_path,
                         const std::string& dict_path,
                         const RuntimeOption& option) {
        detector_ = std::make_shared<DBDetector>(det_model_path, option);
        if (!cls_model_path.empty()) {
            classifier_ = std::make_shared<Classifier>(cls_model_path, option);
        }
        recognizer_ = std::make_shared<Recognizer>(rec_model_path, dict_path, option);
    }

    PaddleOCR::~PaddleOCR() = default;

    bool PaddleOCR::is_initialized() const {
        if (detector_ != nullptr && !detector_->is_initialized()) {
            return false;
        }
        if (classifier_ != nullptr && !classifier_->is_initialized()) {
            return false;
        }
        if (recognizer_ != nullptr && !recognizer_->is_initialized()) {
            return false;
        }
        return true;
    }

    bool PaddleOCR::set_cls_batch_size(const int cls_batch_size) {
        if (cls_batch_size < -1 || cls_batch_size == 0) {
            MD_LOG_ERROR << "batch_size > 0 or batch_size == -1." << std::endl;
            return false;
        }
        cls_batch_size_ = cls_batch_size;
        return true;
    }

    int PaddleOCR::get_cls_batch_size() const { return cls_batch_size_; }

    bool PaddleOCR::set_rec_batch_size(const int rec_batch_size) {
        if (rec_batch_size < -1 || rec_batch_size == 0) {
            MD_LOG_ERROR << "batch_size > 0 or batch_size == -1." << std::endl;
            return false;
        }
        rec_batch_size_ = rec_batch_size;
        return true;
    }

    int PaddleOCR::get_rec_batch_size() const { return rec_batch_size_; }


    std::shared_ptr<DBDetector> PaddleOCR::get_detector() {
        return detector_;
    }

    std::shared_ptr<Classifier> PaddleOCR::get_classifier() {
        return classifier_;
    }

    std::shared_ptr<Recognizer> PaddleOCR::get_recognizer() {
        return recognizer_;
    }


    bool PaddleOCR::predict(const ImageData& image, OCRResult* result, TimerArray* timers) {
        std::vector<OCRResult> batch_result(1);
        if (const bool success = batch_predict({image}, &batch_result, timers); !success) {
            return success;
        }
        *result = std::move(batch_result[0]);
        return true;
    }

    bool PaddleOCR::batch_predict(
        const std::vector<ImageData>& images,
        std::vector<OCRResult>* batch_result, TimerArray* timers) {
        batch_result->clear();
        batch_result->resize(images.size());
        std::vector<std::vector<std::array<int, 8>>> batch_boxes(images.size());
        if (timers) {
            timers->pre_timer.push_back(0);
            timers->post_timer.push_back(0);
            timers->infer_timer.start();
        }
        if (!detector_->batch_predict(images, &batch_boxes)) {
            MD_LOG_ERROR << "There's error while detecting image in PaddleOCR." << std::endl;
            return false;
        }
        for (int i_batch = 0; i_batch < batch_boxes.size(); ++i_batch) {
            sort_boxes(&batch_boxes[i_batch]);
            (*batch_result)[i_batch].boxes = batch_boxes[i_batch];
        }
        for (int i_batch = 0; i_batch < images.size(); ++i_batch) {
            OCRResult& ocr_result = (*batch_result)[i_batch];
            // Get cropped images by detection result
            const std::vector<std::array<int, 8>>& boxes = ocr_result.boxes;
            const ImageData& img = images[i_batch];
            std::vector<ImageData> image_list;
            if (boxes.empty()) {
                image_list.emplace_back(img);
            }
            else {
                image_list.resize(boxes.size());
                for (size_t i_box = 0; i_box < boxes.size(); ++i_box) {
                    cv::Mat _cropped_img;
                    img.to_mat(&_cropped_img);
                    auto _cv_image = get_rotate_crop_image(_cropped_img, boxes[i_box]);
                    image_list[i_box] = ImageData::from_mat(&_cv_image);
                }
            }
            std::vector<int32_t>* cls_labels_ptr = &ocr_result.cls_labels;
            std::vector<float>* cls_scores_ptr = &ocr_result.cls_scores;
            std::vector<std::string>* text_ptr = &ocr_result.text;
            std::vector<float>* rec_scores_ptr = &ocr_result.rec_scores;
            if (nullptr != classifier_) {
                for (size_t start_index = 0; start_index < image_list.size();
                     start_index += cls_batch_size_) {
                    const size_t end_index = std::min(start_index + cls_batch_size_, image_list.size());
                    if (!classifier_->batch_predict(image_list, cls_labels_ptr,
                                                    cls_scores_ptr, start_index,
                                                    end_index)) {
                        MD_LOG_ERROR << "There's error while recognizing image in OCR." << std::endl;
                        return false;
                    }
                    for (size_t i_img = start_index; i_img < end_index; ++i_img) {
                        if (cls_labels_ptr->at(i_img) % 2 == 1 &&
                            cls_scores_ptr->at(i_img) > classifier_->get_postprocessor().get_cls_thresh()) {
                            cv::Mat _cv_image;
                            image_list[i_img].to_mat(&_cv_image);
                            cv::rotate(_cv_image, _cv_image, 1);
                        }
                    }
                }
            }

            std::vector<float> width_list;
            width_list.reserve(image_list.size());
            for (const auto& image : image_list) {
                width_list.push_back(static_cast<float>(image.width()) / static_cast<float>(image.height()));
            }
            std::vector<int> indices = arg_sort(width_list);
            for (size_t start_index = 0; start_index < image_list.size();
                 start_index += rec_batch_size_) {
                const size_t end_index =
                    std::min(start_index + rec_batch_size_, image_list.size());
                if (!recognizer_->batch_predict(image_list, text_ptr, rec_scores_ptr,
                                                start_index, end_index, indices)) {
                    MD_LOG_ERROR << "There's error while recognizing image in PPOCR." << std::endl;
                    return false;
                }
            }
        }
        if (timers) timers->infer_timer.stop();
        return true;
    }
}
