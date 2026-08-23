//
// Created by aichao on 2026/8/23.
//

#pragma once

#include "base_model.h"
#include "vision/common/result.h"
#include "vision/common/image_data.h"
#include "vision/ocr/ppocr.h"
#include "vision/ocr/ppstructurev2_table.h"
#include "vision/ocr/structurev2_layout.h"
#include "vision/ocr/formula_recognition.h"
#include <memory>
#include <string>

namespace modeldeploy::vision::ocr {
    /*! @brief Orchestrates layout -> (ocr/table/formula) -> Markdown.
     *  YAGNI: single-column top-down ordering only (no multi-column reflow).
     */
    class MODELDEPLOY_CXX_EXPORT DocToMarkdown {
    public:
        DocToMarkdown() = default;

        void set_layout(std::unique_ptr<StructureV2Layout> l) { layout_ = std::move(l); }
        void set_table(std::unique_ptr<PPStructureV2Table> t) { table_ = std::move(t); }
        void set_formula(std::unique_ptr<FormulaRecognizer> f) { formula_ = std::move(f); }
        void set_ocr(std::unique_ptr<PaddleOCR> o) { ocr_ = std::move(o); }

        // CDLA label_id -> region type name (self-maintained; postprocessor has no names)
        enum class RegionType { TEXT, TITLE, TABLE, FORMULA, FIGURE, OTHER };

        bool predict(const ImageData& image, std::string* markdown);
        [[nodiscard]] bool ready() const;   // at least layout + one content recognizer

    private:
        static RegionType label_to_region(int32_t label_id);
        static void append_markdown(std::string* md, const std::string& text, RegionType type);

        std::unique_ptr<StructureV2Layout> layout_;
        std::unique_ptr<PPStructureV2Table> table_;
        std::unique_ptr<FormulaRecognizer> formula_;
        std::unique_ptr<PaddleOCR> ocr_;
    };
} // namespace modeldeploy::vision::ocr
