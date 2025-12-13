//
// Created by aichao on 2025/06/04.
//

#include "vision/common/display/display.h"
#include <tabulate/tabulate.hpp>

namespace modeldeploy::vision {
    void dis_attr(const std::vector<AttributeResult>& result) {
        tabulate::Table output_table;
        output_table.format().font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);
        output_table.add_row({"order", " box([x, y, width, height])", " scores"});
        for (size_t i = 0; i < result.size(); ++i) {
            std::string box_str = "["
                + std::to_string(static_cast<int>(std::round(result[i].box.x))) + ", "
                + std::to_string(static_cast<int>(std::round(result[i].box.y))) + ", "
                + std::to_string(static_cast<int>(std::round(result[i].box.width))) + ", "
                + std::to_string(static_cast<int>(std::round(result[i].box.height))) + "]";
            std::string scores_str = "[";
            for (int j = 0; j < result[i].attr_scores.size(); j++) {
                scores_str += std::to_string(result[i].attr_scores[j]).substr(0, 4);
                if (j != result[i].attr_scores.size() - 1) {
                    scores_str += ", ";
                }
            }
            scores_str += "]";
            output_table.add_row({std::to_string(i), box_str, scores_str});
        }
        std::cout << termcolor::cyan << "AttributeResult:" << termcolor::reset << std::endl;
        std::cout << output_table << std::endl;
    }
} // namespace modeldeploy::vision
