//
// Created by aichao on 2025/06/04.
//

#include <tabulate/tabulate.hpp>
#include "vision/common/display/display.h"


namespace modeldeploy::vision {
    void dis_iseg(const std::vector<InstanceSegResult>& result) {
        tabulate::Table output_table;
        output_table.format().font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);
        output_table.add_row({"order", " box([x, y, width, height])", "label_id", " score", "mask_size"});
        for (size_t i = 0; i < result.size(); ++i) {
            std::string box_str = "["
                + std::to_string(static_cast<int>(std::round(result[i].box.x))) + ", "
                + std::to_string(static_cast<int>(std::round(result[i].box.y))) + ", "
                + std::to_string(static_cast<int>(std::round(result[i].box.width))) + ", "
                + std::to_string(static_cast<int>(std::round(result[i].box.height))) + "]";
            std::string label_id_str = std::to_string(result[i].label_id);
            std::string score_str = std::to_string(result[i].score);
            std::string mask_size_str = std::to_string(result[i].mask.shape[0] * result[i].mask.shape[1]);
            output_table.add_row({std::to_string(i), box_str, label_id_str, score_str, mask_size_str});
        }
    }
} // namespace modeldeploy::vision
