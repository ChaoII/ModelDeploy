# 设计：通过 capi + C#/Rust 暴露模型前/后处理参数

日期：2026-08-17 · 分支：capi-v2

## 背景与目标

ModelDeploy SDK 的 C++ 层各模型通过 `get_preprocessor()/get_postprocessor()` 暴露一组前/后处理参数 setter（如检测的 `conf_threshold`/`nms_threshold`、OCR det 的 `det_db_thresh` 等）。但 capi 层目前**没有任何**暴露这些参数的 API，只有 `md_model_set_input_size` 等尺寸类入口。因此 C#/Rust 绑定也无法设置这些参数，只能使用模型默认值。

本设计为 capi 增加统一的"按参数名设置前/后处理参数"能力，覆盖**所有 `MDModelKind`（含全部 pipeline 模型）**，并下沉到 C#/Rust。

## 已确认决策

| 面向 | 决策 |
|------|------|
| 范围 | 覆盖所有 `MDModelKind`，含 pipeline 模型（OCR 整链路 / LPR pipeline / FaceRecPipeline / InsightFaceAnalysis / PedestrianAttribute 等） |
| C API 形态 | `md_model_set_param_i/d/b/s` 按类型拆分（int64/bool, double, bool, string）+ **扁平参数名** + kind 分发表 |
| 参数命名 | 英文小写_下划线、**扁平不嵌套**（pipeline 子模型参数也直接平铺，如 `det_db_thresh` / `cls_thresh`，不引入 `det.*` / `cls.*` 前缀） |
| 作用域 | 模型级持久：设置存于模型句柄，对后续所有 predict 生效，直到再次设置或重建模型；未设置参数用模型默认值 |
| 自省 | `md_model_param_names(kind)` 返回参数名列表 + `md_model_param_type(kind, name)` 返回类型提示，供 C#/Rust 生成强类型包装 |
| C#/Rust 包装 | kv 透传 + 由自省驱动的强类型包装（为简化 binding 实现） |
| 缺失 setter | 在 C++ 子模型 pre/postprocessor 上补齐缺失的 public setter |

## 架构

```
C#/Rust 绑定
    │  kv 透传 + 自省驱动强类型包装
    ▼
capi2/md_capi.*        ← md_model_set_param_* + md_model_param_names/type
    │  kind × 参数名 分发表（数据驱动，switch→static_cast→setter）
    ▼
C++ 模型类 (det/pose/obb/iseg/cls/face/ocr/lpr/ped/insightface…)
    │  get_preprocessor()/get_postprocessor().set_*(…)
    ▼
pre/postprocessor 对象（参数实际存储处 = 模型持久状态）
```

模型对象存放在 `md_model_handle::model`（`void*`），capi 分发表负责 kind→具体类的 setter 映射，沿用现有 `md_model_set_input_size` 的 `switch(kind)` + `static_cast` 模式，不做额外抽象。

## C API 设计

### 新增头文件声明（`capi2/md_capi.h`）

```c
/* ==================== 模型前/后处理参数 ==================== */

/* 按参数名设置模型前/后处理参数（模型级持久；未设置用模型默认值）。
 * 参数名英文小写_下划线、扁平（pipeline 子模型参数直接平铺）。
 * 不支持的 kind 返回 MD_ERR_UNSUPPORTED_TYPE；
 * 不认识的参数名返回 MD_ERR_INVALID_ARGUMENT；类型不匹配返回 MD_ERR_INVALID_TYPE。 */
MDStatus md_model_set_param_i(MDModelHandle, const char* name, int64_t value);   /* 整型参数 */
MDStatus md_model_set_param_d(MDModelHandle, const char* name, double value);    /* 浮点参数 */
MDStatus md_model_set_param_b(MDModelHandle, const char* name, int enable);      /* 布尔参数 (0/1) */
MDStatus md_model_set_param_s(MDModelHandle, const char* name, const char* value);/* 字符串/枚举参数 */

/* 自省：
 * - md_model_param_names：返回该 kind 支持的参数名，以 '|' 分隔的单个字符串（调用方持字符串，无需释放，生命周期绑定库）。kind 无参数时返回空串。
 * - md_model_param_type：返回参数类型提示（'I' 整数 / 'D' 浮点 / 'B' 布尔 / 'S' 字符串）。未知参数名返回 MD_ERR_INVALID_ARGUMENT。 */
MDStatus md_model_param_names(MDModelKind kind, const char** names);
MDStatus md_model_param_type(MDModelKind kind, const char* name, char* type_out);
```

### 分发表（`md_capi.cpp` 内部数据驱动表）

每个条目：`{ 参数名, 类型, 所属 kind 集合, setter 访问器 }`。setter 访问器为"kind→模型→pre/postprocessor 对象→setter"的收口。`set_param_*` 先查表定位参数，再按 kind 分派到具体模型调用 setter。

## 参数分发表（覆盖全部 kind）

### 单模型 kinds

| kind | 参数名 | 类型 | setter（现有或需补齐） |
|------|--------|------|------|
| DETECTION | `conf_threshold` | D | `post.set_conf_threshold` |
| DETECTION | `nms_threshold` | D | `post.set_nms_threshold` |
| POSE | `conf_threshold` / `nms_threshold` / `keypoints_num` | D / D / I | `post.set_*` |
| OBB | `conf_threshold` / `nms_threshold` | D / D | `post.set_*` |
| INSTANCE_SEG | `conf_threshold` / `nms_threshold` / `mask_threshold` | D / D / D | `post.set_*` |
| CLASSIFICATION | `top_k` / `multi_label` | I / B | `post.set_top_k` / `post.set_multi_label` |
| FACE_DET | `conf_threshold` / `nms_threshold` / `landmarks_per_face` | D / D / I | `post.set_*` |
| OCR_DET | `det_db_thresh` / `det_db_box_thresh` / `det_db_unclip_ratio` / `det_db_score_mode` / `use_dilation` | D / D / D / S / B | `post.set_*` |
| OCR_CLS | `cls_thresh` | D | `post.set_cls_thresh` |
| OCR_REC | （rec 无前/后处理阈值类参数；如需仅批大小，属尺寸类，不入本表） | — | — |

### pipeline / 复合 kinds（扁平参数名，路由到子模型）

| kind | 参数名 | 类型 | 路由（现有或需补齐 setter） |
|------|--------|------|------|
| OCR（整链路 PaddleOCR） | `det_db_thresh` / `det_db_box_thresh` / `det_db_unclip_ratio` / `det_db_score_mode` / `use_dilation` / `cls_thresh` | D/D/D/S/B/D | `get_detector()->get_postprocessor().set_*`、`get_classifier()->get_postprocessor().set_cls_thresh` |
| PED_ATTR | `det_threshold` | D | 主类 `set_det_threshold`（已存在） |
| INSIGHTFACE | `det_thresh` | D | 主类 `set_det_thresh`（已存在） |
| FACE_REC_PIPELINE | 视内部子模型暴露而定；需补齐 | — | 路由到 det 子模型 postprocessor |
| LPR_PIPELINE | 视内部暴露而定；需补齐 | — | 路由到 det/rec 子模型 postprocessor |
| SEM_SEG / DEPTH | 当前无前/后处理阈值类参数 | — | 返回空参数列表 |
| CLASSIFICATION（单） | 同单模型分类，无 pipeline 差异 | — | — |
| FACE_REC / FACE_AGE / FACE_GENDER（单） | 无阈值类前/后处理参数 | — | 返回空参数列表 |
| ASR / TTS（音频） | 无前/后处理参数 | — | 返回空参数列表 |

> **说明**：单模型 kinds 中 `classification` 已在单表列出；sem/depth/face_rec 等"无参数"kind 归入空列表，仍可被 `md_model_param_names` 访问（返回空串），保证自省一致性。

### 需在 C++ 层补齐的 setter（覆盖全量所需）

盘点发现需补齐的 public setter（均为在对应子模型 pre/postprocessor 上新增，遵循现有分层）：

- **pipeline 子模型路由**：OCR 整链路需经主类访问 `get_detector()/get_classifier()/get_recognizer()`（已有）路由到各自 postprocessor 的 `set_*`；若 DBDetector 的 det 参数 postprocessor 已有 setter 则直接用，否则补齐。
- **LPR pipeline**：确认 `LprPipeline` 是否暴露 det/rec 子模型 getter；缺则补主类 getter + 子模型 setter。
- **FACE_REC_PIPELINE**：确认 `FaceRecognizerPipeline` 是否暴露 det 子模型 getter 及 det postprocessor 的 `set_conf_threshold`/`set_nms_threshold`；缺则补齐。
- **INSIGHTFACE 子模型**：`set_det_thresh` 已在主类；确认 landmark/genderage/recognition 子模型无需额外阈值 setter。

> 具体补齐清单以实施时对上述类的核对为准；实施阶段（writing-plans）会逐 kind 列出精确的"补 setter"动作。

## 错误处理

- `md_model_set_param_*`：未知 kind / 未就绪 → `MD_ERR_UNSUPPORTED_TYPE` / `MD_ERR_MODEL_INIT`；空参数名 / 未知参数名 → `MD_ERR_INVALID_ARGUMENT`；类型不匹配 → **`MD_ERR_INVALID_TYPE`（新增枚举值，追加到 `MD_STATUS` 末尾，隐式递增为 14，不破坏既有 ABI）**；字符串参数但 value 为空指针 → `MD_ERR_NULL_POINTER`。
- `md_model_param_names`：kind 越界 → `MD_ERR_INVALID_ARGUMENT`。
- 所有错误经 `md_get_last_error` 返回可读信息（沿用现有 thread_local 错误模型）。

## C# / Rust 绑定

### C#（csharp/ModelDeploy/）
- `NativeMethods.cs`：新增 `md_model_set_param_i/d/b/s` 与 `md_model_param_names/type` 的 DllImport。
- `V2/BaseModel.cs` / `V2/Models.cs`：`Model` 基类提供
  `SetParamInt/Double/Bool/Str(string name, …)`、`ParamNames()`、`ParamType(name)`；强类型包装由自省驱动生成（遍历 `ParamNames` 按 `ParamType` 暴露类型化属性）。

### Rust（rust/modeldeploy/src/）
- `ffi.rs`：新增对应 FFI 声明。
- `model.rs` / `image.rs`：`Model::set_param_*`、`param_names()`、`param_type()`；强类型包装由自省驱动。

> **简化 binding 的实现**：因 capi 提供自省（参数名 + 类型），C#/Rust 的强类型包装**生成式**构建，而非为每个模型每参数手工硬编码，降低维护成本和跨语言漂移风险。

## 测试

- **capi 单测**（`tests/test_capi.cpp`，`[capi]` 标签）：对若干 kind（det/pose/ocr/cls/ped）设置参数、读取自省、验证错误码（未知 kind / 未知名 / 类型不匹配）。
- **C#**：`dotnet build` 编译通过 + 一个冒烟用例（set param + param_names）。
- **Rust**：`cargo build` 编译通过 + `cargo test` 冒烟。

## 范围外（YAGNI，不在本期）

- 不做参数"读取当前值"API（`md_model_param_get_*`）；如需后续加。
- 不做批量/结构体设置。
- 不改音频模型参数（无前/后处理）。
- 尺寸/批大小类参数（`set_size`/`set_*_batch_size`）不并入本参数表（保持现有专门 API）。
