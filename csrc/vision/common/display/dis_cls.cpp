//
// Created by aichao on 2025/06/04.
//

#include <core/md_log.h>
#include <tabulate/tabulate.hpp>
#include "vision/common/display/display.h"

namespace modeldeploy::vision {
    void dis_cls(const ClassifyResult& result) {
        tabulate::Table output_table;
        output_table.format().font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);
        output_table.add_row({"label_ids", "scores"});
        if (result.label_ids.size() != result.scores.size()) {
            MD_LOG_ERROR<<"label_ids and scores size not equal";
            return ;
        }
        for (size_t i = 0; i < result.label_ids.size(); ++i) {
            std::string label_ids_str = std::to_string(result.label_ids[i]);
            std::string scores_str = std::to_string(result.scores[i]);
            output_table.add_row({label_ids_str, scores_str});
        }
        std::cout << termcolor::cyan << "ClassificationResult:" << termcolor::reset << std::endl;
        std::cout << output_table << std::endl;
    }
}
