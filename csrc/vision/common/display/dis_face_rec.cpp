//
// Created by aichao on 2025/06/04.
//

#include <numeric>
#include "csrc/core/md_log.h"
#include "csrc/vision/common/display/display.h"

namespace modeldeploy::vision {
    void dis_face_rec(std::vector<FaceRecognitionResult> results) {
        tabulate::Table output_table;
        output_table.format().font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);
        output_table.add_row({"order", "Dim", "Min", "Max", "Mean"});


        for (int i = 0; i < results.size(); i++) {
            auto embedding = results[i].embedding;
            const auto max_it = std::max_element(embedding.begin(), embedding.end());
            const auto min_it = std::min_element(embedding.begin(), embedding.end());
            const auto total_val = std::accumulate(embedding.begin(), embedding.end(), 0.0f);
            const float mean_val = total_val / static_cast<float>(embedding.size());
            output_table.add_row({
                std::to_string(i),
                std::to_string(embedding.size()),
                std::to_string(*min_it),
                std::to_string(*max_it),
                std::to_string(mean_val)
            });
        }

        std::cout << termcolor::cyan << "FaceRecognitionResult:" << termcolor::reset << std::endl;
        std::cout << output_table << std::endl;
    }
}
