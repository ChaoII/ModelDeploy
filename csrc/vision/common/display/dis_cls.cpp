//
// Created by aichao on 2025/06/04.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/common/display/display.h"

namespace modeldeploy::vision {
    void dis_cls(const ClassifyResult& result) {
        tabulate::Table output_table;
        output_table.format().font_color(tabulate::Color::green)
                    .border_color(tabulate::Color::magenta)
                    .corner_color(tabulate::Color::magenta);

        std::string label_ids_str, scores_str;
        for (size_t i = 0; i < result.label_ids.size(); ++i) {
            label_ids_str += std::to_string(result.label_ids[i]);
            if (i < result.label_ids.size() - 1) {
                label_ids_str += ", ";
            }
        }
        for (size_t i = 0; i < result.scores.size(); ++i) {
            scores_str += std::to_string(result.scores[i]);
            if (i < result.scores.size() - 1) {
                scores_str += ", ";
            }
        }
        output_table.add_row({"label_ids", "scores"});
        output_table.add_row({label_ids_str, scores_str});
    }
}
