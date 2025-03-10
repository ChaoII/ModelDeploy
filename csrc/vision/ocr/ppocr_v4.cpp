//
// Created by aichao on 2025/2/21.
//

#include "ppocr_v4.h"
#include "./utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    PPOCRv4::PPOCRv4(const std::string& det_model_path,
                     const std::string& cls_model_path,
                     const std::string& rec_model_path,
                     const std::string& dict_path,
                     const int thread_num,
                     const int max_side_len,
                     const double det_db_thresh,
                     const double det_db_box_thresh,
                     const double det_db_unclip_ratio,
                     const std::string& det_db_score_mode,
                     const bool use_dilation,
                     const int batch_size) {
        RuntimeOption option;
        option.set_cpu_thread_num(thread_num);
        detector_ = std::make_unique<DBDetector>(det_model_path, option);
        detector_->get_preprocessor().set_max_side_len(max_side_len);
        detector_->get_postprocessor().set_det_db_thresh(det_db_thresh);
        detector_->get_postprocessor().set_det_db_box_thresh(det_db_box_thresh);
        detector_->get_postprocessor().set_det_db_score_mode(det_db_score_mode);
        detector_->get_postprocessor().set_det_db_score_mode(det_db_score_mode);
        detector_->get_postprocessor().set_use_dilation(use_dilation);
        detector_->get_postprocessor().set_det_db_unclip_ratio(det_db_unclip_ratio);
        if (!cls_model_path.empty()) {
            classifier_ = std::make_unique<Classifier>(cls_model_path, option);
        }
        recognizer_ = std::make_unique<Recognizer>(rec_model_path, dict_path, option);
        set_cls_batch_size(batch_size);
        set_rec_batch_size(batch_size);
    }


    PPOCRv4::~PPOCRv4() {
    }

    bool PPOCRv4::set_cls_batch_size(int cls_batch_size) {
        if (cls_batch_size < -1 || cls_batch_size == 0) {
            std::cerr << "batch_size > 0 or batch_size == -1." << std::endl;
            return false;
        }
        cls_batch_size_ = cls_batch_size;
        return true;
    }

    int PPOCRv4::get_cls_batch_size() { return cls_batch_size_; }

    bool PPOCRv4::set_rec_batch_size(int rec_batch_size) {
        if (rec_batch_size < -1 || rec_batch_size == 0) {
            std::cerr << "batch_size > 0 or batch_size == -1." << std::endl;
            return false;
        }
        rec_batch_size_ = rec_batch_size;
        return true;
    }

    int PPOCRv4::get_rec_batch_size() { return rec_batch_size_; }

    bool PPOCRv4::initialized() const {
        if (detector_ != nullptr && !detector_->initialized()) {
            return false;
        }
        if (classifier_ != nullptr && !classifier_->initialized()) {
            return false;
        }
        if (recognizer_ != nullptr && !recognizer_->initialized()) {
            return false;
        }
        return true;
    }


    bool PPOCRv4::predict(cv::Mat* image, OCRResult* result) {
        return predict(*image, result);
    }

    bool PPOCRv4::predict(const cv::Mat& image, OCRResult* result) {
        std::vector<OCRResult> batch_result(1);
        const bool success = batch_predict({image}, &batch_result);
        if (!success) {
            return success;
        }
        *result = std::move(batch_result[0]);
        return true;
    };

    bool PPOCRv4::batch_predict(
        const std::vector<cv::Mat>& images,
        std::vector<OCRResult>* batch_result) {
        batch_result->clear();
        batch_result->resize(images.size());
        std::vector<std::vector<std::array<int, 8>>> batch_boxes(images.size());
        if (!detector_->batch_predict(images, &batch_boxes)) {
            std::cerr << "There's error while detecting image in PPOCR." << std::endl;
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
            const cv::Mat& img = images[i_batch];
            std::vector<cv::Mat> image_list;
            if (boxes.empty()) {
                image_list.emplace_back(img);
            }
            else {
                image_list.resize(boxes.size());
                for (size_t i_box = 0; i_box < boxes.size(); ++i_box) {
                    image_list[i_box] = get_rotate_crop_image(img, boxes[i_box]);
                }
            }
            std::vector<int32_t>* cls_labels_ptr = &ocr_result.cls_labels;
            std::vector<float>* cls_scores_ptr = &ocr_result.cls_scores;
            std::vector<std::string>* text_ptr = &ocr_result.text;
            std::vector<float>* rec_scores_ptr = &ocr_result.rec_scores;
            if (nullptr != classifier_) {
                for (size_t start_index = 0; start_index < image_list.size();
                     start_index += cls_batch_size_) {
                    const size_t end_index =
                        std::min(start_index + cls_batch_size_, image_list.size());
                    if (!classifier_->batch_predict(image_list, cls_labels_ptr,
                                                    cls_scores_ptr, start_index,
                                                    end_index)) {
                        std::cerr << "There's error while recognizing image in PPOCR." << std::endl;
                        return false;
                    }
                    for (size_t i_img = start_index; i_img < end_index; ++i_img) {
                        if (cls_labels_ptr->at(i_img) % 2 == 1 &&
                            cls_scores_ptr->at(i_img) >
                            classifier_->get_postprocessor().get_cls_thresh()) {
                            cv::rotate(image_list[i_img], image_list[i_img], 1);
                        }
                    }
                }
            }

            std::vector<float> width_list;
            width_list.reserve(image_list.size());
            for (const auto& image : image_list) {
                width_list.push_back(static_cast<float>(image.cols) / image.rows);
            }
            std::vector<int> indices = arg_sort(width_list);
            for (size_t start_index = 0; start_index < image_list.size();
                 start_index += rec_batch_size_) {
                const size_t end_index =
                    std::min(start_index + rec_batch_size_, image_list.size());
                if (!recognizer_->batch_predict(image_list, text_ptr, rec_scores_ptr,
                                                start_index, end_index, indices)) {
                    std::cerr << "There's error while recognizing image in PPOCR." << std::endl;
                    return false;
                }
            }
        }
        return true;
    }
}
