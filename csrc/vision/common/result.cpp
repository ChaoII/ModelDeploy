//
// Created by aichao on 2025/2/20.
//

#include <numeric>
#include <iostream>
#include <algorithm>
#include "csrc/utils/utils.h"
#include "tabulate/tabulate.hpp"
#include "csrc/vision/common/result.h"


namespace modeldeploy::vision {
    void ClassifyResult::free() {
        label_ids.shrink_to_fit();
        scores.shrink_to_fit();
        feature.shrink_to_fit();
    }

    void ClassifyResult::clear() {
        label_ids.clear();
        scores.clear();
        feature.clear();
    }

    void ClassifyResult::resize(const int size) {
        label_ids.resize(size);
        scores.resize(size);
        // TODO: feature not perform resize now.
        // may need the code below for future.
        // feature.resize(size);
    }

    void ClassifyResult::display() const {
        tabulate::Table output_table;
        output_table.format().font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);


        std::string label_ids_str, scores_str;
        for (size_t i = 0; i < label_ids.size(); ++i) {
            label_ids_str += std::to_string(label_ids[i]);
            if (i < label_ids.size() - 1) {
                label_ids_str += ", ";
            }
        }

        for (size_t i = 0; i < scores.size(); ++i) {
            scores_str += std::to_string(scores[i]);
            if (i < scores.size() - 1) {
                scores_str += ", ";
            }
        }

        if (feature.empty()) {
            output_table.add_row({"label_ids", "scores"});
            output_table.add_row({label_ids_str, scores_str});
        }
        else {
            output_table.add_row({"label_ids", "scores", "dim", "min", "max", "mean"});
            std::string feature_size_str = std::to_string(feature.size());
            const auto max_it = std::max_element(feature.begin(), feature.end());
            const auto min_it = std::max_element(feature.begin(), feature.end());
            const auto total_val = std::accumulate(feature.begin(), feature.end(), 0.0f);
            const float mean_val = total_val / static_cast<float>(feature.size());
            output_table.add_row({
                label_ids_str,
                scores_str,
                feature_size_str,
                std::to_string(*min_it),
                std::to_string(*max_it),
                std::to_string(mean_val)
            });
        }
        std::cout << termcolor::cyan << "ClassifyResult:" << termcolor::reset << std::endl;
        std::cout << output_table << std::endl;
    }


    ClassifyResult& ClassifyResult::operator=(ClassifyResult&& other) noexcept {
        if (&other != this) {
            label_ids = std::move(other.label_ids);
            scores = std::move(other.scores);
            feature = std::move(other.feature);
        }
        return *this;
    }

    void Mask::reserve(const int size) { buffer.reserve(size); }

    void Mask::resize(const int size) { buffer.resize(size); }

    void Mask::free() {
        buffer.shrink_to_fit();
        shape.shrink_to_fit();
    }

    void Mask::clear() {
        buffer.clear();
        shape.clear();
    }

    std::string Mask::str() const {
        std::string out = "Mask(";
        const size_t ndim = shape.size();
        for (size_t i = 0; i < ndim; ++i) {
            if (i < ndim - 1) {
                out += std::to_string(shape[i]) + ",";
            }
            else {
                out += std::to_string(shape[i]);
            }
        }
        out += ")\n";
        return out;
    }

    DetectionResult::DetectionResult(const DetectionResult& res) {
        boxes.assign(res.boxes.begin(), res.boxes.end());
        rotated_boxes.assign(res.rotated_boxes.begin(), res.rotated_boxes.end());
        scores.assign(res.scores.begin(), res.scores.end());
        label_ids.assign(res.label_ids.begin(), res.label_ids.end());
        contain_masks = res.contain_masks;
        if (contain_masks) {
            masks.clear();
            const size_t mask_size = res.masks.size();
            for (size_t i = 0; i < mask_size; ++i) {
                masks.emplace_back(res.masks[i]);
            }
        }
    }

    DetectionResult& DetectionResult::operator=(DetectionResult&& other) noexcept {
        if (&other != this) {
            boxes = std::move(other.boxes);
            rotated_boxes = std::move(other.rotated_boxes);
            scores = std::move(other.scores);
            label_ids = std::move(other.label_ids);
            contain_masks = other.contain_masks;
            if (contain_masks) {
                masks.clear();
                masks = std::move(other.masks);
            }
        }
        return *this;
    }

    void DetectionResult::free() {
        boxes.shrink_to_fit();
        rotated_boxes.shrink_to_fit();
        scores.shrink_to_fit();
        label_ids.shrink_to_fit();
        masks.shrink_to_fit();
        contain_masks = false;
    }

    void DetectionResult::clear() {
        boxes.clear();
        rotated_boxes.clear();
        scores.clear();
        label_ids.clear();
        masks.clear();
        contain_masks = false;
    }

    void DetectionResult::reserve(const int size) {
        boxes.reserve(size);
        rotated_boxes.reserve(size);
        scores.reserve(size);
        label_ids.reserve(size);
        if (contain_masks) {
            masks.reserve(size);
        }
    }

    void DetectionResult::resize(const int size) {
        boxes.resize(size);
        rotated_boxes.resize(size);
        scores.resize(size);
        label_ids.resize(size);
        if (contain_masks) {
            masks.resize(size);
        }
    }

    void DetectionResult::display() const {
        tabulate::Table output_table;
        output_table.format().font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);

        if (rotated_boxes.empty()) {
            output_table.add_row({"order", " box([x1, y1, x2, y2])", "label_id", " score", "mask_size"});
            for (size_t i = 0; i < boxes.size(); ++i) {
                std::string box_str = "["
                    + std::to_string(static_cast<int>(boxes[i][0])) + ", "
                    + std::to_string(static_cast<int>(boxes[i][1])) + ", "
                    + std::to_string(static_cast<int>(boxes[i][2])) + ", "
                    + std::to_string(static_cast<int>(boxes[i][3])) + "]";
                std::string label_id_str = std::to_string(label_ids[i]);
                std::string score_str = std::to_string(scores[i]);
                std::string mask_size_str = std::to_string(masks.size());
                output_table.add_row({std::to_string(i), box_str, label_id_str, score_str, mask_size_str});
            }
        }
        else {
            output_table.add_row({"order", " box([x1, y1, x2, y2, x3, y3, x4, y4])", "label_id", " score"});
            for (size_t i = 0; i < rotated_boxes.size(); ++i) {
                std::string box_str = "["
                    + std::to_string(static_cast<int>(rotated_boxes[i][0])) + ", "
                    + std::to_string(static_cast<int>(rotated_boxes[i][1])) + ", "
                    + std::to_string(static_cast<int>(rotated_boxes[i][2])) + ", "
                    + std::to_string(static_cast<int>(rotated_boxes[i][3])) + ", "
                    + std::to_string(static_cast<int>(rotated_boxes[i][4])) + ", "
                    + std::to_string(static_cast<int>(rotated_boxes[i][5])) + ", "
                    + std::to_string(static_cast<int>(rotated_boxes[i][6])) + ", "
                    + std::to_string(static_cast<int>(rotated_boxes[i][7])) + "]";
                std::string label_id_str = std::to_string(label_ids[i]);
                std::string score_str = std::to_string(scores[i]);
                output_table.add_row({std::to_string(i), box_str, label_id_str, score_str});
            }
        }
        std::cout << termcolor::cyan << "DetectionResult:" << termcolor::reset << std::endl;
        std::cout << output_table << std::endl;
    }


    void OCRResult::clear() {
        boxes.clear();
        text.clear();
        rec_scores.clear();
        cls_scores.clear();
        cls_labels.clear();
    }

#include <sstream>

    std::string OCRResult::str() const {
        std::ostringstream out;

        if (boxes.empty() && rec_scores.empty() && cls_scores.empty() && table_boxes.empty() && table_structure.
            empty()) {
            return "No Results!";
        }

        if (!boxes.empty()) {
            for (int n = 0; n < boxes.size(); n++) {
                out << "det boxes: [";
                for (int i = 0; i < 4; i++) {
                    out << "[" << boxes[n][i * 2] << "," << boxes[n][i * 2 + 1] << "]";
                    if (i != 3) {
                        out << ",";
                    }
                }
                out << "]";

                if (!rec_scores.empty() && n < rec_scores.size()) {
                    out << " rec text: " << text[n] << " rec score: " << rec_scores[n] << " ";
                }
                if (!cls_labels.empty() && n < cls_labels.size()) {
                    out << "cls label: " << cls_labels[n] << " cls score: " << cls_scores[n];
                }
                out << "\n";
            }

            if (!table_boxes.empty() && !table_structure.empty()) {
                for (auto& table_boxe : table_boxes) {
                    out << "table boxes: [";
                    for (int i = 0; i < 4; i++) {
                        out << "[" << table_boxe[i * 2] << "," << table_boxe[i * 2 + 1] << "]";
                        if (i != 3) {
                            out << ",";
                        }
                    }
                    out << "]\n";
                }

                out << "\ntable structure: \n";
                for (const auto& structure : table_structure) {
                    out << structure;
                }

                if (!table_html.empty()) {
                    out << "\n" << "table html: \n" << table_html;
                }
            }
        }
        else {
            if (!rec_scores.empty() && !cls_scores.empty()) {
                for (int i = 0; i < rec_scores.size(); i++) {
                    out << "rec text: " << text[i] << " rec score: " << rec_scores[i] << " ";
                    out << "cls label: " << cls_labels[i] << " cls score: " << cls_scores[i] << "\n";
                }
            }
            else if (!rec_scores.empty()) {
                for (int i = 0; i < rec_scores.size(); i++) {
                    out << "rec text: " << text[i] << " rec score: " << rec_scores[i] << "\n";
                }
            }
            else if (!cls_scores.empty()) {
                for (int i = 0; i < cls_scores.size(); i++) {
                    out << "cls label: " << cls_labels[i] << " cls score: " << cls_scores[i] << "\n";
                }
            }
            else if (!table_boxes.empty() && !table_structure.empty()) {
                for (auto& table_boxe : table_boxes) {
                    out << "table boxes: [";
                    for (int i = 0; i < 4; i++) {
                        out << "[" << table_boxe[i * 2] << "," << table_boxe[i * 2 + 1] << "]";
                        if (i != 3) {
                            out << ",";
                        }
                    }
                    out << "]\n";
                }

                out << "\ntable structure: \n";
                for (const auto& structure : table_structure) {
                    out << structure;
                }

                if (!table_html.empty()) {
                    out << "\n" << "table html: \n" << table_html;
                }
            }
        }

        return out.str();
    }


    DetectionLandmarkResult::DetectionLandmarkResult(const DetectionLandmarkResult& res) {
        boxes.assign(res.boxes.begin(), res.boxes.end());
        landmarks.assign(res.landmarks.begin(), res.landmarks.end());
        scores.assign(res.scores.begin(), res.scores.end());
        landmarks_per_instance = res.landmarks_per_instance;
    }

    void DetectionLandmarkResult::free() {
        boxes.shrink_to_fit();
        scores.shrink_to_fit();
        landmarks.shrink_to_fit();
        landmarks_per_instance = 0;
    }

    void DetectionLandmarkResult::clear() {
        boxes.clear();
        scores.clear();
        landmarks.clear();
        landmarks_per_instance = 0;
    }

    void DetectionLandmarkResult::reserve(const int size) {
        boxes.reserve(size);
        scores.reserve(size);
        if (landmarks_per_instance > 0) {
            landmarks.reserve(size * landmarks_per_instance);
        }
    }

    void DetectionLandmarkResult::resize(const int size) {
        boxes.resize(size);
        scores.resize(size);
        if (landmarks_per_instance > 0) {
            landmarks.resize(size * landmarks_per_instance);
        }
    }

    void DetectionLandmarkResult::display() const {
        tabulate::Table output_table;
        output_table.format().font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);

        if (landmarks.size() != boxes.size() * landmarks_per_instance) {
            std::cerr << "The size of landmarks != boxes.size * landmarks_per_face." << std::endl;
        }
        output_table.add_row({
            "order",
            "box([x1, y1, x2, y2])",
            "score",
            "landmarks(" + std::to_string(landmarks_per_instance) + " * point)"
        });
        for (size_t i = 0; i < boxes.size(); ++i) {
            std::string row_str_box = "["
                + std::to_string(static_cast<int>(boxes[i][0])) + ", "
                + std::to_string(static_cast<int>(boxes[i][1])) + ", "
                + std::to_string(static_cast<int>(boxes[i][2])) + ", "
                + std::to_string(static_cast<int>(boxes[i][3])) + "]";

            std::string row_str_score = std::to_string(scores[i]);
            std::string row_str_landmarks;
            for (size_t j = 0; j < landmarks_per_instance; ++j) {
                row_str_landmarks += "[" +
                    std::to_string(static_cast<int>(landmarks[i * landmarks_per_instance + j][0])) + "," +
                    std::to_string(static_cast<int>(landmarks[i * landmarks_per_instance + j][1]));
                row_str_landmarks += j < landmarks_per_instance - 1 ? "], " : "]";
            }
            output_table.add_row({std::to_string(i), row_str_box, row_str_score, row_str_landmarks});
        }
        std::cout << termcolor::cyan << "DetectionLandmarkResult:" << termcolor::reset << std::endl;
        std::cout << output_table << std::endl;
    }


    FaceRecognitionResult::FaceRecognitionResult(const FaceRecognitionResult& res) {
        embedding.assign(res.embedding.begin(), res.embedding.end());
    }

    void FaceRecognitionResult::free() { std::vector<float>().swap(embedding); }

    void FaceRecognitionResult::clear() { embedding.clear(); }

    void FaceRecognitionResult::reserve(const int size) { embedding.reserve(size); }

    void FaceRecognitionResult::resize(const int size) { embedding.resize(size); }

    void FaceRecognitionResult::display() {
        tabulate::Table output_table;
        output_table.format().font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);
        output_table.add_row({"Dim", "Min", "Max", "Mean"});
        const auto max_it = std::max_element(embedding.begin(), embedding.end());
        const auto min_it = std::max_element(embedding.begin(), embedding.end());
        const auto total_val = std::accumulate(embedding.begin(), embedding.end(), 0.0f);
        const float mean_val = total_val / static_cast<float>(embedding.size());
        output_table.add_row({
            std::to_string(embedding.size()),
            std::to_string(*min_it),
            std::to_string(*max_it),
            std::to_string(mean_val)
        });
        std::cout << termcolor::cyan << "FaceRecognitionResult:" << termcolor::reset << std::endl;
        std::cout << output_table << std::endl;
    }
}
