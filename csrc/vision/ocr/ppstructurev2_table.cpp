//
// Created by aichao on 2025/3/21.
//


#include "csrc/core/md_log.h"
#include "csrc/vision/ocr/utils/ocr_utils.h"
#include "csrc/vision/ocr/ppstructurev2_table.h"

namespace modeldeploy::vision::ocr {
    PPStructureV2Table::PPStructureV2Table(const std::string& det_model_file,
                                           const std::string& rec_model_file,
                                           const std::string& table_model_file,
                                           const std::string& rec_label_file,
                                           const std::string& table_char_dict_path,
                                           const int max_side_len,
                                           const double det_db_thresh,
                                           const double det_db_box_thresh,
                                           const double det_db_unclip_ratio,
                                           const std::string& det_db_score_mode,
                                           const bool use_dilation,
                                           const int rec_batch_size,
                                           const RuntimeOption& option) {
        // detection
        detector_ = std::make_unique<DBDetector>(det_model_file, option);
        detector_->get_preprocessor().set_max_side_len(max_side_len);
        detector_->get_postprocessor().set_det_db_thresh(det_db_thresh);
        detector_->get_postprocessor().set_det_db_box_thresh(det_db_box_thresh);
        detector_->get_postprocessor().set_det_db_score_mode(det_db_score_mode);
        detector_->get_postprocessor().set_det_db_score_mode(det_db_score_mode);
        detector_->get_postprocessor().set_use_dilation(use_dilation);
        detector_->get_postprocessor().set_det_db_unclip_ratio(det_db_unclip_ratio);
        // recognizer
        recognizer_ = std::make_unique<Recognizer>(rec_model_file, rec_label_file, option);
        set_rec_batch_size(rec_batch_size);
        // table recognizer
        table_ = std::make_unique<StructureV2Table>(table_model_file, table_char_dict_path, option);
    }

    bool PPStructureV2Table::is_initialized() const {
        if (detector_ != nullptr && !detector_->is_initialized()) {
            return false;
        }

        if (recognizer_ != nullptr && !recognizer_->is_initialized()) {
            return false;
        }

        if (table_ != nullptr && !table_->is_initialized()) {
            return false;
        }
        return true;
    }

    bool PPStructureV2Table::set_rec_batch_size(int rec_batch_size) {
        if (rec_batch_size < -1 || rec_batch_size == 0) {
            MD_LOG_ERROR << "batch_size > 0 or batch_size == -1." << std::endl;
            return false;
        }
        rec_batch_size_ = rec_batch_size;
        return true;
    }

    [[maybe_unused]] int PPStructureV2Table::get_rec_batch_size() const { return rec_batch_size_; }


    bool PPStructureV2Table::predict(cv::Mat* image,
                                     modeldeploy::vision::OCRResult* result) {
        return predict(*image, result);
    }

    bool PPStructureV2Table::predict(const cv::Mat& image,
                                     modeldeploy::vision::OCRResult* result) {
        std::vector<modeldeploy::vision::OCRResult> batch_result(1);
        const bool success = batch_predict({image}, &batch_result);
        if (!success) {
            return success;
        }
        *result = std::move(batch_result[0]);
        return true;
    }

    bool PPStructureV2Table::batch_predict(
        const std::vector<cv::Mat>& images,
        std::vector<modeldeploy::vision::OCRResult>* batch_result) {
        batch_result->clear();
        batch_result->resize(images.size());
        std::vector<std::vector<std::array<int, 8>>> batch_boxes(images.size());

        if (!detector_->batch_predict(images, &batch_boxes)) {
            MD_LOG_ERROR << "There's error while detecting image in PPOCR." << std::endl;
            return false;
        }

        for (int i_batch = 0; i_batch < batch_boxes.size(); ++i_batch) {
            vision::ocr::sort_boxes(&batch_boxes[i_batch]);
            (*batch_result)[i_batch].boxes = batch_boxes[i_batch];
        }

        for (int i_batch = 0; i_batch < images.size(); ++i_batch) {
            modeldeploy::vision::OCRResult& ocr_result = (*batch_result)[i_batch];
            // Get croped images by detection result
            const std::vector<std::array<int, 8>>& boxes = ocr_result.boxes;
            const cv::Mat& img = images[i_batch];
            std::vector<cv::Mat> image_list;
            if (boxes.empty()) {
                image_list.emplace_back(img);
            }
            else {
                image_list.resize(boxes.size());
                for (size_t i_box = 0; i_box < boxes.size(); ++i_box) {
                    image_list[i_box] = vision::ocr::get_rotate_crop_image(img, boxes[i_box]);
                }
            }

            std::vector<std::string>* text_ptr = &ocr_result.text;
            std::vector<float>* rec_scores_ptr = &ocr_result.rec_scores;

            std::vector<float> width_list;
            width_list.reserve(image_list.size());
            for (auto& image : image_list) {
                width_list.push_back(static_cast<float>(image.cols) / image.rows);
            }
            std::vector<int> indices = vision::ocr::arg_sort(width_list);

            for (size_t start_index = 0; start_index < image_list.size();
                 start_index += rec_batch_size_) {
                size_t end_index =
                    std::min(start_index + rec_batch_size_, image_list.size());
                if (!recognizer_->batch_predict(image_list, text_ptr, rec_scores_ptr,
                                                start_index, end_index, indices)) {
                    MD_LOG_ERROR << "There's error while recognizing image in PPOCR." << std::endl;
                    return false;
                }
            }
        }

        if (!table_->batch_predict(images, batch_result)) {
            MD_LOG_ERROR << "There's error while recognizing tables in images." << std::endl;
            return false;
        }

        for (int i_batch = 0; i_batch < batch_boxes.size(); ++i_batch) {
            modeldeploy::vision::OCRResult& ocr_result = (*batch_result)[i_batch];
            std::vector matched(ocr_result.table_boxes.size(),
                                std::vector<std::string>());

            std::vector<int> ocr_box;
            std::vector<int> structure_box;
            for (int i = 0; i < ocr_result.boxes.size(); i++) {
                ocr_box = vision::ocr::xyxyxyxy2xyxy(ocr_result.boxes[i]);
                ocr_box[0] -= 1;
                ocr_box[1] -= 1;
                ocr_box[2] += 1;
                ocr_box[3] += 1;

                std::vector dis_list(ocr_result.table_boxes.size(),
                                     std::vector<float>(3, 100000.0));

                for (int j = 0; j < ocr_result.table_boxes.size(); j++) {
                    structure_box = vision::ocr::xyxyxyxy2xyxy(ocr_result.table_boxes[j]);
                    dis_list[j][0] = vision::ocr::dis(ocr_box, structure_box);
                    dis_list[j][1] = 1 - vision::ocr::iou(ocr_box, structure_box);
                    dis_list[j][2] = static_cast<float>(j);
                }
                // find min dis idx
                std::ranges::sort(dis_list, vision::ocr::comparison_dis);
                matched[dis_list[0][2]].push_back(ocr_result.text[i]);
            }

            // get pred html
            std::string html_str;
            int td_tag_idx = 0;
            auto structure_html_tags = ocr_result.table_structure;
            for (const auto& structure_html_tag : structure_html_tags) {
                if (structure_html_tag.find("</td>") != std::string::npos) {
                    if (structure_html_tag.find("<td></td>") != std::string::npos) {
                        html_str += "<td>";
                    }
                    if (!matched[td_tag_idx].empty()) {
                        bool b_with = false;
                        if (matched[td_tag_idx][0].find("<b>") != std::string::npos &&
                            matched[td_tag_idx].size() > 1) {
                            b_with = true;
                            html_str += "<b>";
                        }
                        for (int j = 0; j < matched[td_tag_idx].size(); j++) {
                            std::string content = matched[td_tag_idx][j];
                            if (matched[td_tag_idx].size() > 1) {
                                // remove blank, <b> and </b>
                                if (!content.empty() && content.at(0) == ' ') {
                                    content = content.substr(0);
                                }
                                if (content.length() > 2 && content.substr(0, 3) == "<b>") {
                                    content = content.substr(3);
                                }
                                if (content.length() > 4 &&
                                    content.substr(content.length() - 4) == "</b>") {
                                    content = content.substr(0, content.length() - 4);
                                }
                                if (content.empty()) {
                                    continue;
                                }
                                // add blank
                                if (j != matched[td_tag_idx].size() - 1 &&
                                    content.at(content.length() - 1) != ' ') {
                                    content += ' ';
                                }
                            }
                            html_str += content;
                        }
                        if (b_with) {
                            html_str += "</b>";
                        }
                    }
                    if (structure_html_tag.find("<td></td>") != std::string::npos) {
                        html_str += "</td>";
                    }
                    else {
                        html_str += structure_html_tag;
                    }
                    td_tag_idx += 1;
                }
                else {
                    html_str += structure_html_tag;
                }
            }
            (*batch_result)[i_batch].table_html = html_str;
        }
        return true;
    }
}
