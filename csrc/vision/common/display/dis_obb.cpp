//
// Created by aichao on 2025/06/04.
//

#include <tabulate/tabulate.hpp>
#include "vision/common/display/display.h"


namespace modeldeploy::vision {
    void dis_obb(const std::vector<ObbResult>& result) {
        tabulate::Table output_table;
        output_table.format().font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);
        output_table.add_row({"order", " roted_box([xc, yc, width, height, angle(deg)])", "label_id", " score"});
        for (size_t i = 0; i < result.size(); ++i) {
            std::string box_str = "["
                + std::to_string(static_cast<int>(result[i].rotated_box.xc)) + ", "
                + std::to_string(static_cast<int>(result[i].rotated_box.yc)) + ", "
                + std::to_string(static_cast<int>(result[i].rotated_box.width)) + ", "
                + std::to_string(static_cast<int>(result[i].rotated_box.height)) + ", "
                + std::to_string(static_cast<int>(result[i].rotated_box.angle)) + "]";
            std::string label_id_str = std::to_string(result[i].label_id);
            std::string score_str = std::to_string(result[i].score);
            output_table.add_row({std::to_string(i), box_str, label_id_str, score_str});
        }
        std::cout << termcolor::cyan << "ObbResult:" << termcolor::reset << std::endl;
        std::cout << output_table << std::endl;
    }
} // namespace modeldeploy::vision
