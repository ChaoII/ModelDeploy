# 文档理解（公式识别 + 整文档 → Markdown）——设计规范

- 日期：2026-08-22
- 状态：已批准（brainstorming 一次规划）
- 路线：Item 5（顺序 4→5→7→6→8）
- 核心目标：在既有成熟 OCR / 版面(Layout) / 表格(Table) / 语义实体(SER) 之上，补齐**公式识别**，并提供**整文档 → Markdown** 的编排层，形成完整的文档理解闭环。

---

## 1. 背景与动机

探索已确认 `csrc/vision/ocr/` 现有能力成熟：
- `ppocr.h` `PaddleOCR`：det(DB)+cls+rec 顺序 pipeline，成熟。
- `structurev2_layout.h` `StructureV2Layout`：PP-StructureV2 版面检测，成熟。
- `structurev2_table.h` / `ppstructurev2_table.h`：SLANet 表格结构识别 / 完整表格 pipeline，成熟。
- `structurev2_ser_vi_layoutxlm.h` `StructureV2SERViLayoutXLMModel`：SER 语义实体（键值抽取），部分。
- `OCRResult`（`csrc/vision/common/result.h:88-97`）已含 `table_boxes / table_structure / table_html`。

缺失项：**公式识别（formula/LaTeX）**、**整文档 → Markdown** 的编排。本 Item 补这两块。

## 2. 范围（In-Scope / Out-of-Scope，YAGNI 收紧）

### In-Scope
- **公式识别模型类** `FormulaRecognizer`（LaTeX 输出）。
- **整文档 → Markdown 编排**：版面 → (表格 pipeline | 公式识别 | OCR 文本) → Markdown 结构化输出。
- 6 面贯通（C++ / Python / CAPI / C# / Rust / demo+docs+tests；降级处理见 §10）。
- 复用现有 OCR/Layout/Table/SER 组件，不重写。

### Out-of-Scope（明确不做）
- 不做 docling 那种全自动格式推断 / 复杂版面重排。
- 不做公式**检测**（det）+ 识别两段（仅识别给定裁剪块的公式；检测由布局版面盒子承担）——YAGNI，若需求明确需检测再扩展。
- 不做整文档渲染比对 / 像素级对齐。
- 不做表格的复杂合并/多维表头还原（沿用 SLANet 现有 HTML 能力）。

## 3. 架构与组件

### 3.1 目录
```
csrc/vision/ocr/
    formula_recognition.h / .cpp      # FormulaRecognizer : BaseModel
    doc_to_markdown.h / .cpp          # DocToMarkdown（编排层）
```
原有文件不动。命名空间沿用 `modeldeploy::vision::ocr`。

### 3.2 `FormulaRecognizer`（继承 `BaseModel`）
```cpp
namespace modeldeploy::vision::ocr {
class MODELDEPLOY_CXX_EXPORT FormulaRecognizer : public BaseModel {
public:
    explicit FormulaRecognizer(const std::string& model_file,
                               const RuntimeOption& custom_option = RuntimeOption());
    [[nodiscard]] std::string name() const override { return "FormulaRecognizer"; }
    bool predict(const ImageData& image, std::string* latex);          // 单块
    bool batch_predict(const std::vector<ImageData>& images, std::vector<std::string>* latex_list);
    [[nodiscard]] std::unique_ptr<FormulaRecognizer> clone() const;
    [[nodiscard]] bool is_initialized() const;
protected:
    bool initialize();
    bool preprocess(const ImageData& image, std::vector<Tensor>* outputs);
    bool postprocess(std::vector<Tensor>& infer_result, std::string* latex);
private:
    explicit FormulaRecognizer() = default;
};
} // namespace modeldeploy::vision::ocr
```
- 典型公式识别模型（如 PP-FormulaNet / pix2tex / 其他 LaTeX 识别 ONNX）：输入公式裁剪图，输出 token 序列（LaTeX）。后处理将 token ids → LaTeX 文本（依赖模型配套 token 表）。

### 3.3 `DocToMarkdown`（编排层）
```cpp
namespace modeldeploy::vision::ocr {
class MODELDEPLOY_CXX_EXPORT DocToMarkdown {
public:
    /// 以组件指针注入，允许调用方复用已有模型实例；缺省构造后由 set_* 装配。
    DocToMarkdown() = default;

    void set_layout(std::unique_ptr<StructureV2Layout> l);
    void set_table(std::unique_ptr<PPStructureV2Table> t);       // 可选
    void set_formula(std::unique_ptr<FormulaRecognizer> f);       // 可选
    void set_ocr(std::unique_ptr<PaddleOCR> o);                   // 可选

    /// 输入整页图，输出 Markdown 文本。
    bool predict(const ImageData& image, std::string* markdown);

    [[nodiscard]] bool ready() const;   // 至少 layout + 一个内容识别器已装配
};
} // namespace modeldeploy::vision::ocr
```
- **数据流**：整页图 → `StructureV2Layout` 得版面盒 → 对每个盒按类型分派：
  - 文本盒 → `PaddleOCR` 识别 → markdown 文本段。
  - 表格盒 → `PPStructureV2Table` → HTML 表格（嵌入 markdown）。
  - 公式盒 → `FormulaRecognizer` → `$...$` LaTeX。
  - 其他 → 跳过或按 OCR 文本处理。
- 输出按版面几何（自上而下/分栏）排序组合成 Markdown。**YAGNI：先做单栏 / 简单分栏（自上而下排序），不做复杂多栏重排。**

## 4. Python（pybind）
- 新增 `csrc/pybind/vision/formula_recognition_pybind.cpp`（`bind_formula_recognizer`）与 `doc_to_markdown_pybind.cpp`（`bind_doc_to_markdown`），在 `csrc/pybind/vision/vision_pybind.cpp` 注册（`bind_vision` 内 + 前置 `void bind_...` 声明，与 bind_hand/bind_reid 并列）。
- 注册顺序注意：`vision_pybind.cpp` 已因 Item 2/3 加入 `bind_hand` / `bind_reid`；新增 `bind_formula_recognizer` / `bind_doc_to_markdown` 追加其后。
- `bind_vision` 在 `main.cpp` 的 `#ifdef BUILD_VISION` 下调用（沿用现有）。

## 5. CAPI
- `MD_MODEL_KIND` 新增 `MD_MODEL_FORMULA_RECOGNIZER`（在 `MD_MODEL_COUNT` 前）。
- `md_model_create` switch 增加分发（`new ocr::FormulaRecognizer(parts[0], opt)`）。
- 新入口 `md_result_formula(h, i, const char** latex)`：单块 LaTeX 文本。
- `DocToMarkdown` 编排层：**CAPI 面不做**（多组件装配不适合 CAPI 分发的单一 model 模型；Python/C++ 层足够）——YAGNI。

## 6. C#
- 新增 `FormulaRecognizerModel`（仿 `RecognizerModel` / `ReIdModel` 模式）：构造 + `string Predict(ImageData)`。
- `MDModelKind` 枚举新增 `MD_MODEL_FORMULA_RECOGNIZER`（值对齐 CAPI）。
- `DocToMarkdown` C# 面：不做（同 CAPI 理由，YAGNI）。

## 7. Rust
- `FormulaRecognizer`（`model_wrapper!`）：`ffi.rs` `MDModelKind` 新增、`extern` 声明 `md_result_formula`；`model.rs` `FormulaRecognizer::new` + `predict(Image) -> String`。
- `DocToMarkdown` Rust 面：不做（YAGNI）。

## 8. demo + docs
- `examples/demo_doc/`：`demo_doc.cpp + CMakeLists.txt`（加载 layout + table + formula + ocr → 对输入整页图输出 Markdown 到 stdout/文件）。OpenCV 依赖同现有 vision demo。
- `examples/CMakeLists.txt` `add_subdirectory(demo_doc)`；`EXAMPLES.md` 加行；README 能力加"文档理解/公式识别"。

## 9. 测试
- `tests/test_formula_recognition.cpp`（`[formula]`）：无权重 SKIP；有权重时对合成公式图断言输出非空 LaTeX。
- `tests/test_doc_to_markdown.cpp`（`[ocr][doc]`）：装配逻辑分层测试——用桩/空模型验证编排分派与排序逻辑（若可装配假识别器）；真实推理缺权重 SKIP。
- CAPI 用例：`[capi]` 标签，公式模型缺失 guard 跳过。
- Rust `test_formula_recognizer`、C# `FormulaRecog_Works`（缺失权重 SKIP）。

## 10. 交付矩阵（6 面，YAGNI 降级）

| 面 | FormulaRecognizer | DocToMarkdown 编排 |
|----|-------------------|--------------------|
| C++ | ✅ | ✅ |
| Python | ✅ | ✅ |
| CAPI | ✅（`MD_MODEL_FORMULA_RECOGNIZER` + `md_result_formula`） | ❌（YAGNI） |
| C# | ✅ | ❌（YAGNI） |
| Rust | ✅ | ❌（YAGNI） |
| demo+docs+tests | ✅ | ✅（demo 展示闭环） |

## 11. 已知限制 / 假设
- 公式模型权重外链（modelscope / 外部）；测试对缺失 SKIP。
- `DocToMarkdown` 先支持单栏/简单分栏，不做多栏重排（YAGNI）。
- 公式"块"来自布局盒子；若布局不输出公式类盒，需调用方自行裁剪公式区域传入 `FormulaRecognizer`（编排层按布局类型分派，缺布局类型时退化为 OCR 文本）。

## 12. 风险
- 公式识别模型 ONNX 输出形态多样（logits → token 需 char 表 / tokenizer）。缓解：以模型配套的后处理脚本为准，封装进 `postprocess`。
- 整文档 markdown 质量取决于下游模型；编排层只保证结构正确（顺序 + 类型分派），不保证语义完美。
