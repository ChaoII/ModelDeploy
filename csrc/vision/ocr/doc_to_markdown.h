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

        // Non-owning borrow: sub-models stay alive on the caller's side (e.g. the
        // Python bound instances). The orchestration only reads from them.
        void set_layout(StructureV2Layout* l) { layout_ = l; }
        void set_table(PPStructureV2Table* t) { table_ = t; }
        void set_formula(FormulaRecognizer* f) { formula_ = f; }
        void set_ocr(PaddleOCR* o) { ocr_ = o; }

        // CDLA label_id -> region type name (self-maintained; postprocessor has no names)
        enum class RegionType { TEXT, TITLE, TABLE, FORMULA, FIGURE, OTHER };

        bool predict(const ImageData& image, std::string* markdown);
        [[nodiscard]] bool ready() const;   // at least layout + one content recognizer

    private:
        static RegionType label_to_region(int32_t label_id);
        static void append_markdown(std::string* md, const std::string& text, RegionType type);

        StructureV2Layout* layout_ = nullptr;
        PPStructureV2Table* table_ = nullptr;
        FormulaRecognizer* formula_ = nullptr;
        PaddleOCR* ocr_ = nullptr;
    };
} // namespace modeldeploy::vision::ocr
