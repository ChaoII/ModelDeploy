//
// Created by aichao on 2025/06/04.
//

#include <tabulate/tabulate.hpp>
#include "vision/common/display/display.h"

namespace modeldeploy::vision {
    void dis_lpr(const std::vector<LprResult>& result) {
        tabulate::Table output_table;
        output_table.format()
                    .locale(std::locale::classic().name())
                    .multi_byte_characters(true)
                    .font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);

        output_table.add_row({
            "order",
            "box([x, y, width, height])",
            "label_id",
            "score",
            "color",
            "plate_str"
        });
        for (size_t i = 0; i < result.size(); ++i) {
            std::string row_str_box = "["
                + std::to_string(static_cast<int>(std::round(result[i].box.x))) + ", "
                + std::to_string(static_cast<int>(std::round(result[i].box.y))) + ", "
                + std::to_string(static_cast<int>(std::round(result[i].box.width))) + ", "
                + std::to_string(static_cast<int>(std::round(result[i].box.height))) + "]";
            std::string row_str_score = std::to_string(result[i].score);
            std::string row_str_label_id = std::to_string(result[i].label_id);
            std::string row_str_color = result[i].car_plate_color;
            std::string row_str_plate = result[i].car_plate_str;

            output_table.add_row({
                std::to_string(i),
                row_str_box,
                row_str_label_id,
                row_str_score,
                row_str_color,
                row_str_plate
            });
        }
        std::cout << termcolor::cyan << "LprResult:" << termcolor::reset << std::endl;
        std::cout << output_table << std::endl;
    }
}
