# ModelDeploy 文档系统重构 — 设计

日期：2026-08-24
状态：已批准（阶段性 1：核心高价值）

## 1. 背景与问题

ModelDeploy 功能与文档均已相当成熟（docs 中心 17 篇、examples 100+ demo、C++/Python/C/C#/Rust 五套绑定），但存在以下结构性问题：

1. **README.md 内容过重**：既是概述，又内嵌了大量细节（加密、混合精度、量化、TRT engine、Sophgo bmodel 转换、使用方法、模型配置占位），导致"README 不像概述、细节又不够系统"。
2. **README 结构性缺陷**：章节序号错乱（"2.模型加密 / 2.使用方法"重复）；第 8 节"模型配置"（8.1 通用配置 / 8.2 模型输入配置 / 8.3 模型输出配置）仅有空标题，无内容。
3. **文档覆盖缺失**：`docs/models.md` 只到第 14 章，已实现但未入文档的能力包括——多目标跟踪、视频动作识别（TSN/ST-GCN）、文档理解（→Markdown）、行人 Re-ID（OSNet）、声纹验证（ECAPA）、说话人分段（diarization）、流式 STT、TTS 批处理、NLP（分词/分类）、条码/二维码、手部关键点、关键点扩展、CV 解决方案（ObjectCounter/Heatmap/SpeedEstimator/ParkingManager）。
4. **多语言 API 单文件过重**：`docs/apis.md` 一文件覆盖五语言，粒度粗，Rust 覆盖率说明不足。
5. **内部开发笔记混入公开文档**：`capi_risk_register.md`、`image_data_issue_register.md`、`performance_analysis_2026.md`、`optimization_prd.md` 等属开发内部笔记，与公开使用文档混排。

目标：将文档系统重构为 **README = 概述门户、docs/ = 细节库、docs/README.md = 总导航** 的三层结构，补全缺失能力文档并理顺导航。

## 2. 范围

### 本阶段（核心高价值）包含

1. **README 瘦身为概述门户**，细节下沉 docs/。
2. **新增 `docs/conversion.md`**：模型转换/量化/混合精度/TRT engine/bmodel 一站式文档。
3. **补 `docs/backends.md`**：吸收 TRT engine 生成细节。
4. **补 `docs/runtime_option.md`**：补齐模型配置（8.1 通用 / 8.2 输入 / 8.3 输出）。
5. **补 `docs/models.md`**：新增已实现能力的章节并整理小节。
6. **新增 `docs/api/` 目录**：由 `docs/apis.md` 拆分为 `README.md` + `cpp.md`、`python.md`、`capi.md`、`csharp.md`、`rust.md`，并补齐 Rust 覆盖率。
7. **新增 `docs/internal/` 目录**：收纳开发内部笔记，与公开文档主线分离。
8. **新增一键拉取模型/数据脚本** `tools/fetch_test_data.ps1`（Windows）与 `tools/fetch_test_data.sh`（Linux/macOS），README/quickstart 引用。
9. **更新 `docs/README.md` 文档中心导航**，索引全部文档。

### 明确不在本阶段范围

- Qwen3 等功能**实现**（未实现，不写不存在的 API 文档；如后续实现再补）。
- 逐 demo 加注释/输出规范改造、性能基准跑分脚本。
- Triton serving 文档、加密格式二进制细节的深挖（保留现状）。

## 3. 目标文档结构

```
README.md                         概述门户（精简）
docs/
├── README.md                     文档中心 · 总导航
├── quickstart.md                 快速开始（吸收 README 编译/使用方法细节 + 一键拉模型）
├── architecture.md               架构（保持）
├── backends.md                   后端详解（吸收 TRT engine 生成细节）
├── conversion.md                 ★新增·模型转换/量化/混合精度/trtexec/bmodel
├── runtime_option.md             配置详解（补齐 8.1/8.2/8.3 模型配置）
├── models.md                     模型详解（补全新能力章节）
├── preprocess.md                 预处理（保持）
├── performance.md                性能（保持）
├── encryption.md                 模型加密（吸收 README 加密章）
├── multi_thread.md               多线程（保持）
├── sophgo_cross_build_and_test.md Sophgo SOP（保持）
├── api/
│   ├── README.md                 多语言 API 概览 + 导航
│   ├── cpp.md                     C++ 绑定
│   ├── python.md                  Python 绑定
│   ├── capi.md                    C 绑定
│   ├── csharp.md                  C# 绑定
│   └── rust.md                    Rust 绑定（补覆盖率）
└── internal/                     开发内部笔记（迁移自公开目录）
    ├── capi_risk_register.md
    ├── image_data_issue_register.md
    ├── performance_analysis_2026.md
    ├── optimization_prd.md
    └── capi_cpp_language_binding_best_practices.md（如属内部则一并迁移）
```

> 说明：`docs/capi_bindings_analysis.md`、`docs/capi_cpp_language_binding_best_practices.md`、`docs/optimization_prd.md`、`docs/performance_analysis_2026.md` 等是否属内部笔记，实施时逐篇判断；公开使用价值者并入对应公开文档，否则迁入 `internal/`。

## 4. 关键设计决策

### 4.1 README 精简（概述门户）
- **保留**：项目简介与功能亮点、五后端/五绑定/核心能力摘要、一条命令构建、一条命令拉数据、最小可用示例（C++/Python 各一个）、支持矩阵摘要、路线图（勾选状态）、指向 docs/README.md 的链接区。
- **下沉到 docs/**：详细编译流程与 FAQ → `quickstart.md`；加密格式与用法 → `encryption.md`；混合精度/量化/trtexec/bmodel → `conversion.md` + `backends.md`；完整使用方法 → `quickstart.md`；模型配置 → `runtime_option.md`。
- **清理**：删除空占位"第 8 节 模型配置"（内容在 runtime_option.md 落地）。

### 4.2 新增 `docs/conversion.md`
合并 README 中的：
- OnnxRuntime 混合精度（fp32→fp16，keep_io_types）。
- uint8 动态量化减小模型体积。
- TRT engine 生成（`trtexec` 关键参数：min/opt/max shapes、fp16）。
- Sophgo bmodel 转换（`convert.sh`、F16/BF16/INT8/校准/混合精度 qtable、NMS 裁剪注意）。

并互相链接 `backends.md`（后端差异）与 `sophgo_cross_build_and_test.md`（实战 SOP）。

### 4.3 `docs/models.md` 补全新能力
在现有 14 章之后新增章节，覆盖：
15. 多目标跟踪（ByteTracker / BoT-SORT）
16. 视频动作识别（TSN / ST-GCN 骨架）
17. 文档理解（layout + 公式/OCR/表格 → Markdown）
18. 行人 Re-ID（OSNet）
19. 声纹验证（ECAPA-TDNN）
20. 说话人分段（diarization）/ 流式 STT / TTS 批处理
21. NLP（jieba 分词/分句/关键词/统计 + BERT 分类）
22. 条码 / 二维码
23. 手部关键点 / 关键点扩展（车辆关键点、面部 Landmark）
24. CV 解决方案（ObjectCounter / Heatmap / SpeedEstimator / ParkingManager）

每章给出 `class` 名、最小代码片段、指向对应 `examples/` demo 的链接。若某功能章节过长，实施时可拆为 `docs/models/<功能>.md` 子文档，但需在 `models.md` 保留小节与链接索引。

### 4.4 新增 `docs/api/`（按语言拆分）
由 `docs/apis.md` 重构：
- `api/README.md`：五语言统一概览 + 每语言文档链接 + 绑定一致性与差异说明。
- `cpp.md` / `python.md` / `capi.md` / `csharp.md` / `rust.md`：各语言安装、最小示例、关键类型映射、调用示例、链接到对应 `examples/`。
- `rust.md` 明确列出 11 个 Rust 示例（classification / depth / detection / face_age / face_detection / face_gender / face_rec / obb / pose / seg / sem）与 FFI 封装结构。

### 4.5 `docs/internal/`
为开发内部笔记建立独立目录，公开文档导航中不将其作为用户文档展示。迁移判断依据：内容是否面向 SDK 最终使用者（用户文档）还是面向 SDK 开发者（内部笔记）。

### 4.6 一键拉取脚本 `tools/fetch_test_data.{ps1,sh}`
- 行为：从 modelscope 拉取 `test_data.zip`，解压到仓库根 `test_data/`。
- 特性：幂等（已存在/已解压则跳过，可用 `--force` 覆盖）；`--url <url>` 覆盖默认源；下载失败返回非零并给出可读错误；解压失败清理半成品。
- 位置：`tools/` 下，与既有 `tools/docker` 平级。
- 引用：README 快速开始、`docs/quickstart.md`。

## 5. 一致性要求

- 每个文档链接需为相对路径且目标存在；实施时用脚本校验 `docs/` 内所有 `.md` 相对链接均可解析。
- `docs/README.md`（文档中心）必须索引全部公开文档，且与最终目录一致，无死链、无孤儿文档。
- README 精简后不得残留指向不存在文件的链接，也不得把仍在 README 中的内容与 docs/ 重复维护（内容只在一处，另一处链接）。

## 6. 验收标准

- [ ] README.md 为概述门户，无大段细节，目录结构清晰，链接全部有效。
- [ ] docs/conversion.md、docs/api/、docs/internal/ 存在且内容完整。
- [ ] docs/models.md 覆盖本阶段列出的全部新能力章节。
- [ ] docs/README.md 索引与最终目录一致，链接全部可解析。
- [ ] tools/fetch_test_data.ps1 与 tools/fetch_test_data.sh 可运行（至少一平台在干净环境验证拉取+解压）。
- [ ] 所有文档相对链接经脚本校验无死链。

## 7. 拆分建议（后续迭代，不在本阶段实现）

- 逐 demo 加注释 + 文档章节链接。
- 性能基准跑分脚本统一化。
- Qwen3 等功能实现后补文档。
