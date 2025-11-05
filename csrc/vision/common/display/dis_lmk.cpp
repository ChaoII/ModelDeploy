//
// Created by aichao on 2025/06/04.
//

#include <tabulate/tabulate.hpp>
#include "vision/common/display/display.h"

namespace modeldeploy::vision {
    void dis_lmk(const std::vector<KeyPointsResult>& result) {
        tabulate::Table output_table;
        output_table.format().font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);
        output_table.add_row({
            "order",
            "box([x, y, width, height])",
            "label_id",
            "score",
            "landmarks(" + std::to_string(result.size() > 0 ? result[0].keypoints.size() : 0) +" * point)"
        });
        for (size_t i = 0; i < result.size(); ++i) {
            std::string row_str_box = "["
                + std::to_string(static_cast<int>(std::round(result[i].box.x))) + ", "
                + std::to_string(static_cast<int>(std::round(result[i].box.y))) + ", "
                + std::to_string(static_cast<int>(std::round(result[i].box.width))) + ", "
                + std::to_string(static_cast<int>(std::round(result[i].box.height))) + "]";
            std::string row_str_score = std::to_string(result[i].score);
            std::string row_str_label_id = std::to_string(result[i].label_id);
            std::string row_str_landmarks;
            for (size_t j = 0; j < result[i].keypoints.size(); ++j) {
                row_str_landmarks += "[" +
                    std::to_string(static_cast<int>(result[i].keypoints[j].x)) + "," +
                    std::to_string(static_cast<int>(result[i].keypoints[j].y));
                row_str_landmarks += j < result[i].keypoints.size() - 1 ? "], " : "]";
            }
            output_table.add_row({std::to_string(i), row_str_box, row_str_label_id, row_str_score, row_str_landmarks});
        }
        std::cout << termcolor::cyan << "DetectionLandmarkResult:" << termcolor::reset << std::endl;
        std::cout << output_table << std::endl;
    }
}
