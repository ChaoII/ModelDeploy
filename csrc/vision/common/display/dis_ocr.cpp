//
// Created by aichao on 2025/06/04.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/common/display/display.h"

namespace modeldeploy::vision {
    void dis_ocr(const OCRResult& results) {
        std::ostringstream out;
        if (results.boxes.empty() && results.rec_scores.empty() && results.cls_scores.empty() && results.table_boxes.
            empty() && results.table_structure.empty()) {
            return;
        }

        if (!results.boxes.empty()) {
            for (int n = 0; n < results.boxes.size(); n++) {
                out << "det boxes: [";
                for (int i = 0; i < 4; i++) {
                    out << "[" << results.boxes[n][i * 2] << "," << results.boxes[n][i * 2 + 1] << "]";
                    if (i != 3) {
                        out << ",";
                    }
                }
                out << "]";
                if (!results.rec_scores.empty() && n < results.rec_scores.size()) {
                    out << " rec text: " << results.text[n] << " rec score: " << results.rec_scores[n] << " ";
                }
                if (!results.cls_labels.empty() && n < results.cls_labels.size()) {
                    out << "cls label: " << results.cls_labels[n] << " cls score: " << results.cls_scores[n];
                }
                out << "\n";
            }

            if (!results.table_boxes.empty() && !results.table_structure.empty()) {
                for (auto& table_boxe : results.table_boxes) {
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
                for (const auto& structure : results.table_structure) {
                    out << structure;
                }

                if (!results.table_html.empty()) {
                    out << "\n" << "table html: \n" << results.table_html;
                }
            }
        }
        else {
            if (!results.rec_scores.empty() && !results.cls_scores.empty()) {
                for (int i = 0; i < results.rec_scores.size(); i++) {
                    out << "rec text: " << results.text[i] << " rec score: " << results.rec_scores[i] << " ";
                    out << "cls label: " << results.cls_labels[i] << " cls score: " << results.cls_scores[i] << "\n";
                }
            }
            else if (!results.rec_scores.empty()) {
                for (int i = 0; i < results.rec_scores.size(); i++) {
                    out << "rec text: " << results.text[i] << " rec score: " << results.rec_scores[i] << "\n";
                }
            }
            else if (!results.cls_scores.empty()) {
                for (int i = 0; i < results.cls_scores.size(); i++) {
                    out << "cls label: " << results.cls_labels[i] << " cls score: " << results.cls_scores[i] << "\n";
                }
            }
            else if (!results.table_boxes.empty() && !results.table_structure.empty()) {
                for (auto& table_boxe : results.table_boxes) {
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
                for (const auto& structure : results.table_structure) {
                    out << structure;
                }

                if (!results.table_html.empty()) {
                    out << "\n" << "table html: \n" << results.table_html;
                }
            }
        }
        std::cout << termcolor::cyan << "OCRResult:" << termcolor::reset << std::endl;
        std::cout << termcolor::magenta << out.str() << termcolor::reset << std::endl;
    }
}
