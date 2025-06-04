//
// Created by aichao on 2025/2/20.
//

#include <numeric>
#include <iostream>
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

    void ClassifyResult::reserve(const int size) {
        scores.reserve(size);
        label_ids.reserve(size);
    }


    void ClassifyResult::resize(const int size) {
        label_ids.resize(size);
        scores.resize(size);
        // TODO: feature not perform resize now.
        // may need the code below for future.
        // feature.resize(size);
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


    void OCRResult::clear() {
        boxes.clear();
        text.clear();
        rec_scores.clear();
        cls_scores.clear();
        cls_labels.clear();
    }

    void FaceRecognitionResult::display() {
        tabulate::Table output_table;
        output_table.format().font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);
        output_table.add_row({"Dim", "Min", "Max", "Mean"});

        const auto max_it = std::ranges::max_element(embedding);
        const auto min_it = std::ranges::min_element(embedding);
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
