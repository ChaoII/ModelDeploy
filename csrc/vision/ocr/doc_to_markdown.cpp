//
// Created by aichao on 2026/8/23.
//

#include "vision/ocr/doc_to_markdown.h"
#include <algorithm>

namespace modeldeploy::vision::ocr {
    bool DocToMarkdown::ready() const {
        if (!layout_) return false;
        return (bool)table_ || (bool)formula_ || (bool)ocr_;
    }

    DocToMarkdown::RegionType DocToMarkdown::label_to_region(int32_t label_id) {
        // CDLA (picodet_lcnet_x1_0_fgd_layout_cdla, num_class=5): mapping by convention.
        // Tune when real layout model available.
        switch (label_id) {
            case 0: return RegionType::TEXT;     // or TITLE depending on dict
            case 1: return RegionType::TABLE;
            case 2: return RegionType::FORMULA;
            default: return RegionType::OTHER;
        }
    }

    void DocToMarkdown::append_markdown(std::string* md, const std::string& text, RegionType type) {
        switch (type) {
            case RegionType::TABLE:
                (*md) += "\n" + text + "\n";       // table_html embedded as-is
                break;
            case RegionType::FORMULA:
                (*md) += "$" + text + "$\n\n";
                break;
            case RegionType::TITLE:
                (*md) += "## " + text + "\n\n";
                break;
            case RegionType::TEXT:
            default:
                (*md) += text + "\n\n";
                break;
        }
    }

    bool DocToMarkdown::predict(const ImageData& image, std::string* markdown) {
        if (!ready() || !markdown) return false;
        std::vector<DetectionResult> regions;
        if (!layout_->predict(image, &regions)) return false;
        // YAGNI: order by top-to-bottom (y of box); simple stable sort
        std::sort(regions.begin(), regions.end(),
                  [](const DetectionResult& a, const DetectionResult& b) {
                      return a.box.y < b.box.y;
                  });
        markdown->clear();
        for (auto& r : regions) {
            RegionType type = label_to_region(r.label_id);
            ImageData crop = image.crop(r.box);
            if (crop.empty()) continue;   // crop region from page (by-value; drop on failure)
            switch (type) {
                case RegionType::TABLE:
                    if (table_) {
                        OCRResult tbl;
                        if (table_->predict(crop, &tbl) && !tbl.table_html.empty())
                            append_markdown(markdown, tbl.table_html, RegionType::TABLE);
                    }
                    break;
                case RegionType::FORMULA:
                    if (formula_) {
                        std::string latex;
                        if (formula_->predict(crop, &latex) && !latex.empty())
                            append_markdown(markdown, latex, RegionType::FORMULA);
                    }
                    break;
                case RegionType::TEXT:
                case RegionType::TITLE:
                case RegionType::FIGURE:
                case RegionType::OTHER:
                default:
                    if (ocr_) {
                        OCRResult ocr;
                        if (ocr_->predict(crop, &ocr)) {
                            std::string joined;
                            for (auto& t : ocr.text) joined += t + " ";
                            if (!joined.empty())
                                append_markdown(markdown, joined, type);
                        }
                    }
                    break;
            }
        }
        return true;
    }
} // namespace modeldeploy::vision::ocr
