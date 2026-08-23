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
        ~DocToMarkdown() = default;

        // Owning overload: DocToMarkdown takes ownership of the sub-model and
        // releases it when the DocToMarkdown instance is destroyed.
        void set_layout(std::unique_ptr<StructureV2Layout> l) {
            layout_own_ = std::move(l);
            layout_ = layout_own_.get();
        }
        void set_table(std::unique_ptr<PPStructureV2Table> t) {
            table_own_ = std::move(t);
            table_ = table_own_.get();
        }
        void set_formula(std::unique_ptr<FormulaRecognizer> f) {
            formula_own_ = std::move(f);
            formula_ = formula_own_.get();
        }
        void set_ocr(std::unique_ptr<PaddleOCR> o) {
            ocr_own_ = std::move(o);
            ocr_ = ocr_own_.get();
        }

        // Borrowing overload: DocToMarkdown keeps a NON-OWNING reference and
        // never takes ownership. The caller must keep the sub-model alive for as
        // long as DocToMarkdown (or its predict/ready) is used. This is the path
        // used by the Python bindings, where keep_alive guarantees the bound
        // sub-model outlives this instance.
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

        std::unique_ptr<StructureV2Layout> layout_own_;   // ownership (owning overload)
        std::unique_ptr<PPStructureV2Table> table_own_;
        std::unique_ptr<FormulaRecognizer> formula_own_;
        std::unique_ptr<PaddleOCR> ocr_own_;
        StructureV2Layout* layout_ = nullptr;             // effective pointer (borrow/own)
        PPStructureV2Table* table_ = nullptr;
        FormulaRecognizer* formula_ = nullptr;
        PaddleOCR* ocr_ = nullptr;
    };
} // namespace modeldeploy::vision::ocr
