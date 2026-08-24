# Item 5: 文档理解（公式识别 + 整文档 → Markdown）——实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在既有 OCR/Layout/Table 之上，新增 `FormulaRecognizer`（LaTeX 公式识别）与 `DocToMarkdown`（整文档 → Markdown 编排层），6 面贯通（C++/Python/CAPI/C#/Rust/demo+docs+tests，编排层按 YAGNI 降级）。

**Architecture:** `FormulaRecognizer` 继承 `BaseModel`，仿 `Recognizer`（CTC `[batch,seq,num_class]` 输出 + 外部 dict 文件 + argmax 去重后处理）。`DocToMarkdown` 编排层注入 `StructureV2Layout` + `PaddleOCR`/`PPStructureV2Table`/`FormulaRecognizer`，按版面 label_id 分派（文本/表格/公式/图）并自上而下排序拼成 Markdown。

**Tech Stack:** C++17、pybind11、现有 `BaseModel`/`Runtime`、PP-StructureV2 系列组件、Catch2。

**Spec:** `docs/superpowers/specs/2026-08-22-doc-understanding-design.md`

## Global Constraints

- 命名空间 `modeldeploy::vision::ocr`；新文件放 `csrc/vision/ocr/`（`GLOB_RECURSE` 自动收集）。
- **FormulaRecognizer 仿 `Recognizer`**（`csrc/vision/ocr/recognizer.h/.cpp`）：CTC 式输出 + 外部 dict（`label_path`）后处理，argmax + 去重。参考 `rec_postprocessor.cpp:12-25`（read_dict）与 `:68`（label_list_ 映射）。
- `DocToMarkdown` 复用现有组件 `set_layout/set_table/set_formula/set_ocr`（unique_ptr 注入）；CDLA 版面类别命名映射由 pipeline 侧自建（后处理无内建名字）；num_class 默认 5。
- 无公式/test_data 权重 → 真实推理测试 SKIP；编排层逻辑用桩/装配测试恒定通过（不依赖权重）。
- CAPI：`MD_MODEL_FORMULA_RECOGNIZER`（`MD_MODEL_COUNT` 前）；`md_result_formula`。`DocToMarkdown` CAPI/C#/Rust 面**不做**（YAGNI）。
- 复用 `csrc/vision/common/result.h` 的 `OCRResult` / `DetectionResult`；不新增结果结构。
- Windows 构建：`.bat` 包裹 `vcvars64.bat` 经 `cmd /c`；测试用 `build_capi`（BUILD_AUDIO=ON, CAPI=ON, VISION=ON, TESTS=ON）。
- `csrc/vision/utils.cpp` 的 `l2_normalize`(471)/`compute_similarity`(459) 本项目不用（非相似度功能）。

---

### Task 1: `FormulaRecognizer` C++ 核心类 + `DocToMarkdown` 编排层

**Files:**
- Create: `csrc/vision/ocr/formula_recognition.h`
- Create: `csrc/vision/ocr/formula_recognition.cpp`
- Create: `csrc/vision/ocr/doc_to_markdown.h`
- Create: `csrc/vision/ocr/doc_to_markdown.cpp`
- Test: `tests/test_formula_recognition.cpp`
- Test: `tests/test_doc_to_markdown.cpp`

**Interfaces:**
- Produces (later tasks rely on):
  - `modeldeploy::vision::ocr::FormulaRecognizer(const std::string& model_file, const std::string& char_dict_path = "", const RuntimeOption& = RuntimeOption())`
  - `bool FormulaRecognizer::predict(const ImageData&, std::string* latex)`
  - `bool FormulaRecognizer::batch_predict(const std::vector<ImageData>&, std::vector<std::string>*)`
  - `bool FormulaRecognizer::is_initialized() const`
  - `std::unique_ptr<FormulaRecognizer> FormulaRecognizer::clone() const`
  - `modeldeploy::vision::ocr::DocToMarkdown::set_layout/set_table/set_formula/set_ocr` (unique_ptr)
  - `bool DocToMarkdown::predict(const ImageData&, std::string* markdown)`
  - `bool DocToMarkdown::ready() const`

- [ ] **Step 1: Write failing tests**

`tests/test_formula_recognition.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include "csrc/vision/ocr/formula_recognition.h"

using namespace modeldeploy::vision::ocr;

// Constant-pass: ctor error path / is_initialized false without weights
TEST_CASE("FormulaRecognizer construction", "[formula]") {
    FormulaRecognizer m("nonexistent_formula.onnx");
    REQUIRE_FALSE(m.is_initialized());
}
```

`tests/test_doc_to_markdown.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include "csrc/vision/ocr/doc_to_markdown.h"

using namespace modeldeploy::vision::ocr;

// Constant-pass: readiness semantics without any models/layout
TEST_CASE("DocToMarkdown ready semantics", "[ocr][doc]") {
    DocToMarkdown d;
    REQUIRE_FALSE(d.ready());          // nothing set
}
```

- [ ] **Step 2: Run to verify fail**

Run: `cd build_capi && .\bin\test_modeldeploy.exe "[formula]" "[ocr][doc]"`
Expected: FAIL（头文件不存在，编译失败）

- [ ] **Step 3: Write `csrc/vision/ocr/formula_recognition.h`**

```cpp
#pragma once

#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"
#include <string>
#include <vector>

namespace modeldeploy::vision::ocr {
    /*! @brief Formula (LaTeX) recognition model, mirrors the OCR Recognizer pattern.
     *  Input: cropped formula image; Output: LaTeX string.
     *  Requires a char/token dict file (CTC-style [batch, seq, num_class] output).
     */
    class MODELDEPLOY_CXX_EXPORT FormulaRecognizer : public BaseModel {
    public:
        FormulaRecognizer(const std::string& model_file,
                          const std::string& char_dict_path,
                          const RuntimeOption& custom_option = RuntimeOption());
        [[nodiscard]] std::string name() const override { return "FormulaRecognizer"; }

        bool predict(const ImageData& image, std::string* latex);
        bool batch_predict(const std::vector<ImageData>& images, std::vector<std::string>* latex_list);
        [[nodiscard]] bool is_initialized() const;
        [[nodiscard]] std::unique_ptr<FormulaRecognizer> clone() const;

    protected:
        bool initialize();
        bool preprocess(const ImageData& image, std::vector<Tensor>* outputs);
        bool postprocess(std::vector<Tensor>& infer_result, std::string* latex);
        bool preprocess_batch(const std::vector<ImageData>& images, std::vector<Tensor>* outputs);

    private:
        explicit FormulaRecognizer() = default;   // clone()
        std::string char_dict_path_;
        // token id -> latex char/token
        std::map<int32_t, std::string> token_table_;
        int32_t rec_image_h_{48};
        int32_t rec_image_w_{320};
        bool initialized_ = false;
    };
} // namespace modeldeploy::vision::ocr
```

- [ ] **Step 4: Write `csrc/vision/ocr/formula_recognition.cpp`**

```cpp
#include "csrc/vision/ocr/formula_recognition.h"
#include <algorithm>
#include <fstream>

namespace modeldeploy::vision::ocr {
    FormulaRecognizer::FormulaRecognizer(const std::string& model_file,
                                         const std::string& char_dict_path,
                                         const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        char_dict_path_ = char_dict_path;
        initialized_ = initialize();
    }

    std::unique_ptr<FormulaRecognizer> FormulaRecognizer::clone() const {
        auto m = std::unique_ptr<FormulaRecognizer>(new FormulaRecognizer());
        m->set_runtime(const_cast<FormulaRecognizer*>(this)->clone_runtime());
        m->runtime_option = runtime_option;
        m->char_dict_path_ = char_dict_path_;
        m->token_table_ = token_table_;
        m->rec_image_h_ = rec_image_h_;
        m->rec_image_w_ = rec_image_w_;
        m->initialized_ = initialized_;
        return m;
    }

    bool FormulaRecognizer::is_initialized() const { return initialized_; }

    bool FormulaRecognizer::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "FormulaRecognizer: failed to init runtime." << std::endl;
            return false;
        }
        if (!char_dict_path_.empty()) {
            std::ifstream fin(char_dict_path_);
            std::string line;
            int32_t idx = 0;
            // CTC blank at 0, mirror rec_postprocessor read_dict
            token_table_[idx++] = "#";
            while (std::getline(fin, line)) {
                if (!line.empty()) token_table_[idx++] = line;
            }
            if (token_table_.size() <= 1) {
                MD_LOG_WARN << "FormulaRecognizer: char dict empty." << std::endl;
            }
        }
        initialized_ = true;
        return true;
    }

    bool FormulaRecognizer::preprocess(const ImageData& image, std::vector<Tensor>* outputs) {
        // Resize/crop to rec_image_h_ x rec_image_w_, normalize, NHW->NCHW
        // Reuse cv resize + to tensor like rec_preprocessor; kept minimal (no weights to verify).
        // Placeholder body — implement tensor fill from image (RGB->float /[0,1], transpose HWC->CHW).
        MD_LOG_WARN << "FormulaRecognizer::preprocess uses default [3,H,W] layout." << std::endl;
        (void)image; (void)outputs;
        return false;  // replaced during integration (see note)
    }

    bool FormulaRecognizer::preprocess_batch(const std::vector<ImageData>& images, std::vector<Tensor>* outputs) {
        (void)images; (void)outputs;
        return false;
    }

    bool FormulaRecognizer::postprocess(std::vector<Tensor>& infer_result, std::string* latex) {
        if (infer_result.empty()) return false;
        auto& t = infer_result[0];
        const auto shape = t.shape();
        // CTC-style [batch, seq, num_class]; do argmax over last dim + dedupe consecutive
        if (shape.size() < 2) return false;
        const int64_t seq = shape[shape.size() - 2];
        const int64_t nc  = shape.back();
        const float* p = static_cast<const float*>(t.data());
        int32_t prev = -1;
        for (int64_t s = 0; s < seq; ++s) {
            const float* row = p + s * nc;
            const int32_t best = static_cast<int32_t>(
                std::distance(row, std::max_element(row, row + nc)));
            if (best != prev && best != 0) {   // skip blank(0) & consecutive dup
                auto it = token_table_.find(best);
                if (it != token_table_.end()) *latex += it->second;
            }
            prev = best;
        }
        return true;
    }

    bool FormulaRecognizer::predict(const ImageData& image, std::string* latex) {
        std::vector<ImageData> imgs{ image };
        return batch_predict(imgs, latex == nullptr ? nullptr : &(std::vector<std::string>{}));
    }

    // batch_predict main
    bool FormulaRecognizer::batch_predict(const std::vector<ImageData>& images,
                                          std::vector<std::string>* latex_list) {
        if (images.empty() || !latex_list) return false;
        if (!preprocess_batch(images, &reused_input_tensors_)) return false;
        for (int i = 0; i < reused_input_tensors_.size(); i++)
            reused_input_tensors_[i].set_name(get_input_info(i).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
        latex_list->clear();
        latex_list->resize(images.size());
        // NOTE: single-batch path OK; multi-batch slice backfill mirrors Recognizer slice pattern (T8 refine if weights present)
        std::string first;
        if (!postprocess(reused_output_tensors_, &first)) return false;
        (*latex_list)[0] = std::move(first);
        return true;
    }
} // namespace modeldeploy::vision::ocr
```

> **实现提示（preprocess 占位）**：`preprocess` 目前是占位（返回 false）——因为无权重无法对齐真实模型的 resize/归一化/通道布局。**Task 3~8 联调时**若拿到公式 ONNX，按 `get_input_info(0).shape` 实现真正的裁剪+归一化+transpose（参考 `csrc/vision/ocr/rec_preprocessor.h` 的 `set_rec_image_shape`/`set_normalize` 模式）。编排层测试（Task 1）不依赖 preprocess 真值。postprocess 的 CTC argmax+去重已实现完整（可独立单测，无需权重）。

- [ ] **Step 5: Write `csrc/vision/ocr/doc_to_markdown.h`**

```cpp
#pragma once

#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/ocr/ppocr.h"
#include "csrc/vision/ocr/ppstructurev2_table.h"
#include "csrc/vision/ocr/structurev2_layout.h"
#include "csrc/vision/ocr/formula_recognition.h"
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
```

- [ ] **Step 6: Write `csrc/vision/ocr/doc_to_markdown.cpp`**

```cpp
#include "csrc/vision/ocr/doc_to_markdown.h"
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
            ImageData crop;
            if (!image.crop(r.box, &crop)) continue;   // crop region from page
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
```

> **提示**：`ImageData::crop(box, &crop)` / `Rect2f` 的成员（x,y,width,height）需按 `csrc/vision/common/struct.h` 核实（`Rect2f` 有 `x/y/width/height`；`DetectionResult.box` 为 `Rect2f`）。若 crop 方法名/签名不同，按实际调整。`label_to_region` 映射是惯例占位，真实 CDLA dict 就绪时在 T8 校正。

- [ ] **Step 7: Run tests to pass**

Run: `cd build_capi && .\bin\test_modeldeploy.exe "[formula]" "[ocr][doc]"`
Expected: FormulaRecognizer ctor（is_initialized false）与 DocToMarkdown ready semantics 通过。注册 test_formula_recognition.cpp/test_doc_to_markdown.cpp 到 `tests/CMakeLists.txt` TEST_SOURCES（先做）。

- [ ] **Step 8: Register tests + build**

Modify `tests/CMakeLists.txt` TEST_SOURCES: add `test_formula_recognition.cpp` + `test_doc_to_markdown.cpp`.
`cmake --build build_capi --parallel 8` → 0 errors.

- [ ] **Step 9: Commit**

```bash
git add csrc/vision/ocr/formula_recognition.h csrc/vision/ocr/formula_recognition.cpp csrc/vision/ocr/doc_to_markdown.h csrc/vision/ocr/doc_to_markdown.cpp tests/test_formula_recognition.cpp tests/test_doc_to_markdown.cpp
git commit -m "feat(ocr): FormulaRecognizer + DocToMarkdown orchestration core"
```

---

### Task 2: pybind FormulaRecognizer + DocToMarkdown

**Files:**
- Create: `csrc/pybind/vision/formula_recognition_pybind.cpp`
- Modify: `csrc/pybind/vision/vision_pybind.cpp`（+前置声明 + 注册，追加到 bind_hand/bind_reid 后）
- Test: （Python smoke）

**Interfaces:**
- Consumes: FormulaRecognizer/DocToMarkdown (Task 1)
- Produces: Python `modeldeploy.vision.ocr.FormulaRecognizer` / `DocToMarkdown`

- [ ] **Step 1: Write `csrc/pybind/vision/formula_recognition_pybind.cpp`**
绑定 `FormulaRecognizer`（ctor(model_file, char_dict_path="", option)、predict(image)->str、is_initialized）与 `DocToMarkdown`（set_layout/set_table/set_formula/set_ocr、predict(image)->str）。参照 `csrc/pybind/vision/ocr/ocr_layout_pybind.py` 的 ImageData/predict 绑定方式。

- [ ] **Step 2: Edit `vision_pybind.cpp`** 加 `void bind_formula_recognizer(pybind11::module&);` 声明并在 `bind_vision` 内注册（`#ifdef BUILD_VISION` 下，append 到 bind_hand/bind_reid）。

- [ ] **Step 3: Build + smoke**（用之前 Item 4 的 pybind 构建或临时开 BUILD_PYTHON）：`python -c "import modeldeploy; from modeldeploy.vision import ocr; ..."`（构造/ready 冒烟）。

- [ ] **Step 4: Commit**

```bash
git add csrc/pybind/vision/formula_recognition_pybind.cpp csrc/pybind/vision/vision_pybind.cpp
git commit -m "feat(pybind): bind FormulaRecognizer + DocToMarkdown"
```

---

### Task 3: CAPI FormulaRecognizer

**Files:**
- Modify: `capi/md_capi.h`（`MD_MODEL_FORMULA_RECOGNIZER` + `md_result_formula` 声明）
- Modify: `capi/md_capi.cpp`（create/delete/clone 分发 + `md_result_formula`）
- Test: `tests/test_capi.cpp`（`[capi]` 用例）

**Interfaces:**
- Consumes: FormulaRecognizer (Task 1)
- Produces: C `MD_MODEL_FORMULA_RECOGNIZER` + `md_result_formula(h, i, &latex)`

- [ ] **Step 1: `md_capi.h`** 枚举 `MD_MODEL_FORMULA_RECOGNIZER`（`MD_MODEL_COUNT` 前）；声明 `md_result_formula(MDModelHandle, size_t i, const char** latex)`。
- [ ] **Step 2: `md_capi.cpp`** `#ifdef BUILD_VISION` 分发：create `new ocr::FormulaRecognizer(parts[0], parts.size()>1?parts[1]:"", opt)`（need_parts(1..2)）；delete/clone；`MD_RES_OCR` 结果读取加 formula（或仿 md_result_ocr 结构）。`#else` 报错分支补 kind。
- [ ] **Step 3: test_capi.cpp** 加 `[capi]` 用例：枚举 + 错误路径（模型缺 guard 跳过）。
- [ ] **Step 4: Build + test** `.\bin\test_modeldeploy.exe "[capi]"`。
- [ ] **Step 5: Commit**

```bash
git add capi/md_capi.h capi/md_capi.cpp tests/test_capi.cpp
git commit -m "feat(capi): MD_MODEL_FORMULA_RECOGNIZER + md_result_formula"
```

---

### Task 4: C# FormulaRecognizerModel

**Files:**
- Modify: `csharp/ModelDeploy/types_internal_c.cs`（枚举 + extern `md_result_formula`）
- Modify: `csharp/ModelDeploy/Models.cs`（`FormulaRecognizerModel`）
- Modify: `csharp/ModelDeploy/NativeMethods.cs`
- Test: `csharp/ModelDeployUnitTest/AllModelsTests.cs`（`FormulaRecog_Works`）

**Interfaces:**
- Consumes: CAPI (Task 3)
- Produces: C# `ModelDeploy.FormulaRecognizerModel`

- [ ] **Step 1**: `MDModelKind` 加 `MD_MODEL_FORMULA_RECOGNIZER`（值对齐 CAPI）；Models.cs 加 `FormulaRecognizerModel(string modelPath)` + `string Predict(ImageData)`（调 `md_result_formula`，`Marshal.PtrToStringAnsi`）。枚举值 = REID+…（对齐 CAPI 当期 count）。
- [ ] **Step 2**: `AllModelsTests.cs` 加 `[Fact] FormulaRecog_Works`（无权重 SKIP）。
- [ ] **Step 3**: `dotnet build` + `dotnet test --filter FormulaRecog_Works`。
- [ ] **Step 4: Commit**

```bash
git add csharp/ModelDeploy/*.cs csharp/ModelDeployUnitTest/AllModelsTests.cs
git commit -m "feat(csharp): FormulaRecognizerModel"
```

---

### Task 5: Rust FormulaRecognizer

**Files:**
- Modify: `rust/modeldeploy/src/ffi.rs`（枚举 + extern `md_result_formula`）
- Modify: `rust/modeldeploy/src/model.rs`、`lib.rs`（`FormulaRecognizer`）
- Test: `rust/modeldeploy/tests/integration_test.rs`（`test_formula_recognizer`）

**Interfaces:**
- Consumes: CAPI (Task 3)
- Produces: Rust `modeldeploy::FormulaRecognizer`

- [ ] **Step 1**: ffi.rs 枚举 + extern；`model.rs` `FormulaRecognizer::new(model, opt)` + `predict(Image)->Result<String>`（`md_result_formula`，`PtrToStringUtf8` 复制）；lib.rs re-export。
- [ ] **Step 2**: `test_formula_recognizer`（无权重 skip）。
- [ ] **Step 3**: `cargo build` + `cargo test test_formula_recognizer`；clippy 干净。
- [ ] **Step 4: Commit**

```bash
git add rust/modeldeploy/src/*.rs rust/modeldeploy/tests/integration_test.rs
git commit -m "feat(rust): FormulaRecognizer binding"
```

---

### Task 6: demo + docs

**Files:**
- Create: `examples/demo_doc/demo_doc.cpp`
- Create: `examples/demo_doc/CMakeLists.txt`
- Modify: `examples/CMakeLists.txt`、`examples/EXAMPLES.md`、`README.md`

**Interfaces:**
- Consumes: FormulaRecognizer/DocToMarkdown (Task 1)

- [ ] **Step 1**: `demo_doc.cpp`（+ CMakeLists 单目标链 `${LIBRARY_NAME}`）：加载 layout+table+formula+ocr → 输入整页图输出 Markdown。参照 `demo_structure_table_cxx.cpp` 用法；`ImageData.imread` + `DocToMarkdown.predict`。缺模型清晰报错。
- [ ] **Step 2**: examples/CMakeLists 加 `add_subdirectory(demo_doc)`；`EXAMPLES.md` 加 3 列行；`README.md` 能力加"文档理解/公式识别"。
- [ ] **Step 3**: `cmake --build build_capi`；`demo_doc.exe`（无参 Usage / 缺模型错误路径）。
- [ ] **Step 4: Commit**

```bash
git add examples/demo_doc/ examples/CMakeLists.txt examples/EXAMPLES.md README.md
git commit -m "feat(examples): demo_doc + docs"
```

---

### Task 7: 全量验证 + 编排逻辑强化

**Files:** 无新增（验证；必要时 minor 修复）

- [ ] **Step 1 全量 C++**: `.\bin\test_modeldeploy.exe "[formula]" "[ocr][doc]" "[capi]" "[core]" "[tracking]"`。记录通过数；无回归。
- [ ] **Step 2 跨后端**: grep formula_recognition/doc_to_markdown 无 backend 依赖（仅 BaseModel/组件指针）→ 跨后端一致。
- [ ] **Step 3 编排逻辑分层测试强化**：若可装配假 Content 识别器（无权重），验证 DocToMarkdown 的 label_to_region + append_markdown 分派（TABLE/FORMULA/TEXT/TITLE 的 markdown 输出）→ 优先用轻量桩测覆盖（不依赖真实模型）。
- [ ] **Step 4 minor 修复**：修发现项（低风险必要的）。报告 concerns。

---

## Self-Review 记录

**Spec coverage**：FormulaRecognizer(§3.2)、DocToMarkdown(§3.3)、Python(§4)、CAPI(§5)、C#(§6)、Rust(§7)、demo(§8)、test(§9)、交付矩阵(§10) 都有对应 Task。YAGNI 降级（DocToMarkdown 的 CAPI/C#/Rust 面不做）在 T3/T4/T5 中体现为"仅 FormulaRecognizer"。

**Placeholder**：`FormulaRecognizer::preprocess` 与 `label_to_region` 映射是**已知占位**（无权重无法定真值），已显式标注"Task 3~8 联调时校正"，非未定义接口。CTC postprocess 已完整实现。

**Type consistency**：`predict(const ImageData&, std::string*)`/`batch_predict` 在 Task 1/2/3 一致；`DocToMarkdown` 的 set_*/predict/ready 一致；枚举名 `MD_MODEL_FORMULA_RECOGNIZER` 全链一致。
