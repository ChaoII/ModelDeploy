# 模型前/后处理参数 capi + C#/Rust 绑定 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 capi 能按扁平参数名设置所有模型（含 pipeline）的前/后处理参数，并下沉到 C#/Rust；对 C++ 层缺失的 setter 补齐，使 kv 入口能触达全量参数。

**Architecture:** capi 新增 `md_model_set_param_i/d/b/s` + 自省 `md_model_param_names/type`，内部用"kind × 参数名"分发表把 kv 映射到 C++ 模型的 `get_preprocessor()/get_postprocessor().set_*()`（沿用现有 `md_model_set_input_size` 的 switch+static_cast 模式）。C++ 层为 pipeline 子模型补齐缺失的 public setter。C#/Rust 做 kv 透传 + 自省驱动强类型包装。

**Tech Stack:** C++17，capi（C API），C#（P/Invoke），Rust（FFI），Catch2（capi 测试）。

## Global Constraints

- C++17 必需（AGENTS.md），不用 C++20。
- capi 模型对象存于 `md_model_handle::model`（`void*`），分发表用 `switch(kind)` + `static_cast` 到具体类（同 `md_model_set_input_size`）。
- 参数名英文小写下划线、**扁平不嵌套**（pipeline 子模型参数也平铺，如 `det_db_thresh`/`cls_thresh`，不用 `det.*` 前缀）。
- 模型级持久：参数存于模型对象内部状态（pre/postprocessor 成员），对后续 predict 生效；未设置用模型默认值。
- 覆盖**所有 `MDModelKind`**（含 pipeline）；无前/后处理参数的 kind 返回空参数列表。
- 类型拆分：`md_model_set_param_i`(int64/bool)、`_d`(double)、`_s`(string)、`_b`(bool)。bool 统一用 0/1 的 int 表示（`_b` 与 `_i` 均可触发，类型表只标 'B'/'I'）。
- 新增错误码 `MD_ERR_INVALID_TYPE`，追加到 `MD_STATUS` 枚举末尾（现末位 `MD_ERR_AUDIO_DECODE`=13 → 新值 14）。不破坏既有 ABI。
- CPU 构建（WITH_GPU=OFF / ENABLE_SOPHGO=OFF / BUILD_AUDIO 任意）必须不受影响 —— 参数设置与设备无关，但需保证取模型类型时 `#ifdef` 治理音频等条件编译 kind。
- 不重复造已存在的 getter：优先生成独立、有明确单一职责的小函数/表项（DRY）；不做与设置无关的改动。

---

### Task 1: C++ pipeline 子模型补齐缺失 setter

**Files:**
- Modify: `csrc/vision/face/face_rec_pipeline/face_rec_pipeline.h` / `.cpp`
- Modify: `csrc/vision/lpr/lpr_pipeline/lpr_pipeline.h`（如需）
- Test: `tests/test_capi.cpp`（新增用例，见 Task 3；本任务不写独立单测，仅编译验证）

**Interfaces:**
- Consumes: 现有 `Scrfd`（`get_postprocessor().set_conf_threshold/set_nms_threshold/set_landmarks_per_face`）、`LprDetection`/`LprRecognizer`（无可配置参数）
- Produces: `FaceRecognizerPipeline::get_detector()`（返回 `std::shared_ptr<Scrfd>`）供 capi 分发表路由 det 子模型参数。LPR pipeline 经确认无可配置前/后处理参数，**不补 getter**（返回空参数列表）。

- [ ] **Step 1: 在 face_rec_pipeline.h 增加 public getter**

在 `class FaceRecognizerPipeline` 的 public 区（`is_initialized()` 之后、`clone()` 之前）增加：

```cpp
        /// 暴露 det 子模型（用于配置检测前/后处理参数，如 conf/nms threshold）
        std::shared_ptr<Scrfd> get_detector();
```

- [ ] **Step 2: 在 face_rec_pipeline.cpp 实现 getter**

在文件内（可与现有方法同处）增加：

```cpp
        std::shared_ptr<Scrfd> FaceRecognizerPipeline::get_detector() {
            return detector_;
        }
```

- [ ] **Step 3: 编译验证**

Run（在 `build_tdc`，需先配置过）：
```
cd E:\CLionProjects\ModelDeploy
"<VS>\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build build_tdc --target ModelDeploySDK
```
Expected: 编译通过无 error（`<VS>` 指 Visual Studio 安装路径，如 `C:\Program Files\Microsoft Visual Studio\2022\Community`）。

- [ ] **Step 4: 提交**

```bash
git add csrc/vision/face/face_rec_pipeline
git commit -m "feat(vision): expose FaceRecognizerPipeline det submodel getter for param routing"
```

---

### Task 2: capi 分发表 + 4 个 setter + 2 个自省函数

**Files:**
- Modify: `capi/md_capi.h`（枚举 + 6 个函数声明）
- Modify: `capi/md_capi.cpp`（分发表 + 实现）
- Test: `tests/test_capi.cpp`

**Interfaces:**
- Consumes: C++ 模型的 `get_preprocessor()/get_postprocessor()` setter（det/pose/obb/iseg/cls/face_det/ocr_det/ocr_cls/ocr 整链路/ped_attr/insightface/face_rec_pipeline）
- Produces: C 层 API（供 Task 4/5 的 C#/Rust 消费）：
  - `MDStatus md_model_set_param_i(MDModelHandle, const char* name, int64_t value);`
  - `MDStatus md_model_set_param_d(MDModelHandle, const char* name, double value);`
  - `MDStatus md_model_set_param_b(MDModelHandle, const char* name, int enable);`
  - `MDStatus md_model_set_param_s(MDModelHandle, const char* name, const char* value);`
  - `MDStatus md_model_param_names(MDModelKind kind, const char** names);`
  - `MDStatus md_model_param_type(MDModelKind kind, const char* name, char* type_out);`

- [ ] **Step 1: md_capi.h 新增错误码 + 函数声明**

在 `MD_STATUS` 枚举末尾（`MD_ERR_AUDIO_DECODE` 之后、`} MDStatus;` 之前）追加：

```c
    MD_ERR_INVALID_TYPE        /* 参数类型不匹配 */
```

在 `md_model_set_input_size`/`md_model_set_cls_input_size` 声明之后新增（放在 `md_model_predict` 之前均可）：

```c
/* ==================== 模型前/后处理参数 ==================== */
/* 按扁平参数名设置模型前/后处理参数（模型级持久；未设置用默认值）。
 * _i 整型、_d 浮点、_b 布尔(0/1)、_s 字符串/枚举。不支持的 kind → MD_ERR_UNSUPPORTED_TYPE；
 * 未知名 → MD_ERR_INVALID_ARGUMENT；类型不匹配 → MD_ERR_INVALID_TYPE；s 且 value=null → MD_ERR_NULL_POINTER。 */
MD_CAPI_EXPORT MDStatus md_model_set_param_i(MDModelHandle, const char* name, int64_t value);
MD_CAPI_EXPORT MDStatus md_model_set_param_d(MDModelHandle, const char* name, double value);
MD_CAPI_EXPORT MDStatus md_model_set_param_b(MDModelHandle, const char* name, int enable);
MD_CAPI_EXPORT MDStatus md_model_set_param_s(MDModelHandle, const char* name, const char* value);

/* 自省：names 以 '|' 分隔的单个字符串（库持有，无需释放；kind 无参数→空串）；
 * type_out 输出 'I'/'D'/'B'/'S'；未知名 → MD_ERR_INVALID_ARGUMENT；kind 越界 → MD_ERR_INVALID_ARGUMENT。 */
MD_CAPI_EXPORT MDStatus md_model_param_names(MDModelKind kind, const char** names);
MD_CAPI_EXPORT MDStatus md_model_param_type(MDModelKind kind, const char* name, char* type_out);
```

- [ ] **Step 2: md_capi.cpp 定义分发表**

在 `md_capi.cpp` 顶部（`#include` 之后、`struct md_model_handle` 之前），定义参数元数据与分发表：

```cpp
namespace {
// 参数类型标记（自省返回字符）
enum ParamType { PT_I = 'I', PT_D = 'D', PT_B = 'B', PT_S = 'S' };

struct ParamSpec { const char* name; char type; };

// 单条 setter 分派：kind 已知时按参数名调用具体模型 setter。
// 由 md_model_set_param_* 统一入口调用。返回 true=已处理；false=未知名/不支持。
} // namespace
```

> 说明：具体"参数名 → 模型 setter"的分派体量大，建议在 `md_capi.cpp` 内新增一个 `apply_model_param(void* model, MDModelKind kind, const char* name, ParamType req_type, bool is_bool, int64_t i, double d, const char* s)` 内部辅助函数，内部用 `switch(kind)` + `strcmp(name,...)` 逐参数调用对应模型的 pre/postprocessor setter。任务实现者按下方分发表逐 kind 实现该函数。

**分发表（kind → 参数名 → setter）必须在 `apply_model_param` 中全覆盖：**

| kind | 参数名 | 类型 | setter 调用（C++） |
|------|--------|------|------|
| DETECTION | `conf_threshold` | D | `static_cast<detection::UltralyticsDet*>(m)->get_postprocessor().set_conf_threshold((float)d)` |
| DETECTION | `nms_threshold` | D | `...get_postprocessor().set_nms_threshold((float)d)` |
| POSE | `conf_threshold` | D | `...UltralyticsPose...get_postprocessor().set_conf_threshold((float)d)` |
| POSE | `nms_threshold` | D | `...set_nms_threshold((float)d)` |
| POSE | `keypoints_num` | I | `...set_keypoints_num((int)i)` |
| OBB | `conf_threshold` | D | `...UltralyticsObb...set_conf_threshold((float)d)` |
| OBB | `nms_threshold` | D | `...set_nms_threshold((float)d)` |
| INSTANCE_SEG | `conf_threshold` | D | `...UltralyticsSeg...set_conf_threshold((float)d)` |
| INSTANCE_SEG | `nms_threshold` | D | `...set_nms_threshold((float)d)` |
| INSTANCE_SEG | `mask_threshold` | D | `...set_mask_threshold((float)d)` |
| CLASSIFICATION | `top_k` | I | `...classification::Classification...get_postprocessor().set_top_k((int)i)` |
| CLASSIFICATION | `multi_label` | B | `...set_multi_label(enable!=0)` |
| FACE_DET | `conf_threshold` | D | `...face::Scrfd...get_postprocessor().set_conf_threshold((float)d)` |
| FACE_DET | `nms_threshold` | D | `...set_nms_threshold((float)d)` |
| FACE_DET | `landmarks_per_face` | I | `...set_landmarks_per_face((int)i)` |
| OCR_DET | `det_db_thresh` | D | `...ocr::DBDetector...get_postprocessor().set_det_db_thresh(d)` |
| OCR_DET | `det_db_box_thresh` | D | `...set_det_db_box_thresh(d)` |
| OCR_DET | `det_db_unclip_ratio` | D | `...set_det_db_unclip_ratio(d)` |
| OCR_DET | `det_db_score_mode` | S | `...set_det_db_score_mode(s)` |
| OCR_DET | `use_dilation` | B | `...set_use_dilation(enable!=0)` |
| OCR_CLS | `cls_thresh` | D | `...ocr::Classifier...get_postprocessor().set_cls_thresh((float)d)` |
| OCR（整链路 PaddleOCR） | `det_db_thresh` | D | `static_cast<ocr::PaddleOCR*>(m)->get_detector()->get_postprocessor().set_det_db_thresh(d)` |
| OCR（整链路） | `det_db_box_thresh` | D | `...get_detector()->get_postprocessor().set_det_db_box_thresh(d)` |
| OCR（整链路） | `det_db_unclip_ratio` | D | `...set_det_db_unclip_ratio(d)` |
| OCR（整链路） | `det_db_score_mode` | S | `...set_det_db_score_mode(s)` |
| OCR（整链路） | `use_dilation` | B | `...set_use_dilation(enable!=0)` |
| OCR（整链路） | `cls_thresh` | D | `static_cast<ocr::PaddleOCR*>(m)->get_classifier()->get_postprocessor().set_cls_thresh((float)d)` |
| PED_ATTR | `det_threshold` | D | `...pipeline::PedestrianAttribute...set_det_threshold((float)d)` |
| INSIGHTFACE | `det_thresh` | D | `...face::InsightFaceAnalysis...set_det_thresh((float)d)` |
| FACE_REC_PIPELINE | `conf_threshold` | D | `...FaceRecognizerPipeline*>(m)->get_detector()->get_postprocessor().set_conf_threshold((float)d)` |
| FACE_REC_PIPELINE | `nms_threshold` | D | `...get_detector()->get_postprocessor().set_nms_threshold((float)d)` |
| FACE_REC_PIPELINE | `landmarks_per_face` | I | `...get_detector()->get_postprocessor().set_landmarks_per_face((int)i)` |

**空参数列表的 kind**（`apply_model_param` 直接返回 false，`md_model_param_names` 返回空串）：`SEM_SEG`、`DEPTH`、`FACE_REC`、`FACE_AGE`、`FACE_GENDER`、`FACE_AS`、`FACE_AS_PIPELINE`、`INSIGHTFACE_DET`、`OCR_REC`、`LPR_DET`、`LPR_REC`、`LPR_PIPELINE`、`ASR`、`TTS`。

- [ ] **Step 3: 实现 apply_model_param + 自省辅助**

在 `md_capi.cpp` 实现 `apply_model_param`（按上表逐 kind）。骨架（`#ifdef ENABLE_SOPHGO` 等不重要，模型类型静态转换即可；注意 `MD_MODEL_ASR/TTS` 在 `#ifdef BUILD_AUDIO` 内）：

```cpp
namespace {
    // 依据 kind 上报支持的参数名列表（'|' 拼接，静态）
    const char* kind_param_names(MDModelKind kind) {
        switch (kind) {
        case MD_MODEL_DETECTION:
        case MD_MODEL_POSE:
        case MD_MODEL_OBB:
        case MD_MODEL_INSTANCE_SEG:
        case MD_MODEL_FACE_DET:
        case MD_MODEL_FACE_REC_PIPELINE:
            return "conf_threshold|nms_threshold";
        case MD_MODEL_CLASSIFICATION:
            return "top_k|multi_label";
        case MD_MODEL_OCR_DET:
        case MD_MODEL_OCR:
            return "det_db_thresh|det_db_box_thresh|det_db_unclip_ratio|det_db_score_mode|use_dilation";
        case MD_MODEL_OCR_CLS:
            return "cls_thresh";
        case MD_MODEL_PED_ATTR:
            return "det_threshold";
        case MD_MODEL_INSIGHTFACE:
            return "det_thresh";
        default:
            return "";
        }
    }

    const char* param_type_of(MDModelKind kind, const char* name) {
        // 返回 'I'/'D'/'B'/'S'，未知返回 nullptr。按上表逐 kind 判断。
        // 实现建议：为清晰，直接对每个 (kind,name) 返回统一类型（同名前在不同 kind 同类型）。
    }
} // namespace
```

> 上述为骨架，实现者需按分发表**补全** `param_type_of` 的每个 kind/name → 类型映射。

- [ ] **Step 4: 实现 6 个导出函数**

```cpp
MDStatus md_model_set_param_i(MDModelHandle handle, const char* name, int64_t value) {
    auto* mh = static_cast<md_model_handle*>(handle);
    if (!mh || !mh->ready) return MD_ERR_MODEL_INIT;
    if (!name || !*name) return MD_ERR_INVALID_ARGUMENT;
    if (!apply_model_param(mh, name, PT_I, false, value, 0.0, nullptr)) {
        // apply_model_param 内部对"未知名"设错误信息并区分类型/未知；此处统一返回它设置的错误码
        return md_last_param_status();
    }
    return MD_OK;
}
```

> 为实现清晰，推荐 `apply_model_param` 返回 int（0=ok，否则 MD_ERR_*），并在内部用 `set_error_fmt` 设置具体信息（未知名 → `MD_ERR_INVALID_ARGUMENT` 且列出该 kind 支持参数；类型不匹配 → `MD_ERR_INVALID_TYPE`）。`md_model_set_param_d/b/s` 同理调用 `apply_model_param(mh, name, PT_D/PT_B/PT_S, ...)`。

自省实现：

```cpp
MDStatus md_model_param_names(MDModelKind kind, const char** names) {
    if (!names) return MD_ERR_NULL_POINTER;
    if (kind < 0 || kind >= MD_MODEL_COUNT) { set_error("md_model_param_names: invalid kind"); return MD_ERR_INVALID_ARGUMENT; }
    *names = kind_param_names(kind);
    return MD_OK;
}

MDStatus md_model_param_type(MDModelKind kind, const char* name, char* type_out) {
    if (!name || !type_out) return MD_ERR_NULL_POINTER;
    if (kind < 0 || kind >= MD_MODEL_COUNT) { set_error("md_model_param_type: invalid kind"); return MD_ERR_INVALID_ARGUMENT; }
    const char* t = param_type_of(kind, name);
    if (!t) { set_error_fmt("md_model_param_type: unknown param '%s' for kind %d", name, (int)kind); return MD_ERR_INVALID_ARGUMENT; }
    *type_out = *t;
    return MD_OK;
}
```

- [ ] **Step 5: 编译验证**

Run:
```
cd E:\CLionProjects\ModelDeploy
"<VS>\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build build_tdc --target ModelDeploySDK
```
Expected: 0 error。

- [ ] **Step 6: capi 单测（tests/test_capi.cpp，[capi] 标签）**

新增 TEST_CASE，覆盖：det set conf/nms 成功、ocr_det set 各参数、自省 names/type、错误码（未知名 → INVALID_ARGUMENT、类型不匹配 → INVALID_TYPE、kind 无参数 → 空 names）：

```cpp
TEST_CASE("capi model set param + introspection", "[capi]") {
    // 自省
    const char* names = nullptr;
    REQUIRE(md_model_param_names(MD_MODEL_DETECTION, &names) == MD_OK);
    REQUIRE(names);
    CHECK(std::string(names).find("conf_threshold") != std::string::npos);
    char t = 0;
    REQUIRE(md_model_param_type(MD_MODEL_DETECTION, "conf_threshold", &t) == MD_OK);
    CHECK(t == 'D');
    CHECK(md_model_param_type(MD_MODEL_DETECTION, "nope", &t) == MD_ERR_INVALID_ARGUMENT);
    const char* sem_names = nullptr;
    REQUIRE(md_model_param_names(MD_MODEL_SEM_SEG, &sem_names) == MD_OK);
    CHECK(std::string(sem_names).empty());

    // 需要一个就绪的检测模型实例来测 setter；若环境无模型则用 [model] 标签单独覆盖。
    // 无模型时的纯自省逻辑（不依赖模型）即在本用例验证；setter 的端到端用 [model] 标签。
}
```

> 因多数 setter 需就绪模型，建议把"设置参数"端到端用例加 `[model]` 标签（CI 有模型时执行），纯自省（不依赖模型）在 `[capi]` 常规执行。实现时参考现有 test_capi.cpp 的模型加载模式。

- [ ] **Step 7: 运行测试**

Run:
```
cd E:\CLionProjects\ModelDeploy\build_tdc
.\bin\test_modeldeploy.exe "[capi]"
```
Expected: 全部通过（含新增自省用例）。

- [ ] **Step 8: 提交**

```bash
git add capi/md_capi.h capi/md_capi.cpp tests/test_capi.cpp
git commit -m "feat(capi): model pre/post-process param setters + introspection (md_model_set_param_*)"
```

---

### Task 3: C# 绑定（kv + 自省驱动的强类型包装）

**Files:**
- Modify: `csharp/ModelDeploy/NativeMethods.cs`
- Modify: `csharp/ModelDeploy/V2/BaseModel.cs`
- Modify: `csharp/ModelDeploy/V2/Models.cs`
- Test: `csharp/ModelDeployUnitTest/CapiVisionTests.cs`（可选冒烟）

**Interfaces:**
- Consumes: Task 2 的 6 个 C 函数
- Produces: `Model.SetParamInt/Double/Bool/Str(string name, …)`、`Model.ParamNames()`、`Model.ParamType(string): char`、`Model.SupportedParams`（由自省驱动的类型化字典）

- [ ] **Step 1: NativeMethods.cs 新增 DllImport**

在现有 `md_model_set_*_input_size` 附近新增：

```csharp
[DllImport(NativeLib, CallingConvention = CallingConvention.Cdecl)]
internal static extern MDStatus md_model_set_param_i(IntPtr model, string name, long value);
[DllImport(NativeLib, CallingConvention = CallingConvention.Cdecl)]
internal static extern MDStatus md_model_set_param_d(IntPtr model, string name, double value);
[DllImport(NativeLib, CallingConvention = CallingConvention.Cdecl)]
internal static extern MDStatus md_model_set_param_b(IntPtr model, string name, int enable);
[DllImport(NativeLib, CallingConvention = CallingConvention.Cdecl)]
internal static extern MDStatus md_model_set_param_s(IntPtr model, string name, string value);
[DllImport(NativeLib, CallingConvention = CallingConvention.Cdecl)]
internal static extern MDStatus md_model_param_names(int kind, out IntPtr names);
[DllImport(NativeLib, CallingConvention = CallingConvention.Cdecl)]
internal static extern MDStatus md_model_param_type(int kind, string name, out byte typeOut);
```

- [ ] **Step 2: BaseModel.cs 新增方法 + 自省驱动的参数字典**

在 `Model`（BaseModel）类内新增：

```csharp
        /// <summary>设置模型前/后处理参数（扁平参数名，见 ParamNames）。</summary>
        public void SetParam(string name, long value) { Check(NativeMethods.md_model_set_param_i(Handle, name, value)); }
        public void SetParam(string name, double value) { Check(NativeMethods.md_model_set_param_d(Handle, name, value)); }
        public void SetParam(string name, bool value) { Check(NativeMethods.md_model_set_param_b(Handle, name, value ? 1 : 0)); }
        public void SetParam(string name, string value) { Check(NativeMethods.md_model_set_param_s(Handle, name, value)); }

        /// <summary>该模型 kind 支持的参数名（'|' 分隔）。</summary>
        public string[] ParamNames() {
            var status = NativeMethods.md_model_param_names((int)Kind, out var p);
            Check(status);
            return Marshal.PtrToStringUTF8(p).Split('|', StringSplitOptions.RemoveEmptyEntries);
        }

        public char ParamType(string name) {
            Check(NativeMethods.md_model_param_type((int)Kind, name, out var t));
            return (char)t;
        }
```

> `Check`/`Handle`/`Kind` 均已有（沿用现有绑定）；`NativeLib`、`MDStatus` 常量如需 `OUT_OF_MEMORY` 等映射沿用现有。

- [ ] **Step 3: dotnet build 验证**

Run: `dotnet build`（在 `E:\CLionProjects\ModelDeploy\csharp`）
Expected: 0 error / 0 warning。

- [ ] **Step 4: 提交**

```bash
git add csharp
git commit -m "feat(csharp): model pre/post-process param setter + introspection"
```

---

### Task 4: Rust 绑定（kv + 自省）

**Files:**
- Modify: `rust/modeldeploy/src/ffi.rs`
- Modify: `rust/modeldeploy/src/model.rs`
- Test: `rust/modeldeploy/tests/integration_test.rs`（可选）

**Interfaces:**
- Consumes: Task 2 的 6 个 C 函数
- Produces: `Model::set_param_int/double/bool/str(name, …)`、`Model::param_names() -> Vec<String>`、`Model::param_type(name) -> char`

- [ ] **Step 1: ffi.rs 新增 FFI 声明**

在 `md_model_set_*_input_size` 附近新增：

```rust
extern "C" {
    pub fn md_model_set_param_i(model: *mut c_void, name: *const c_char, value: i64) -> MDStatus;
    pub fn md_model_set_param_d(model: *mut c_void, name: *const c_char, value: f64) -> MDStatus;
    pub fn md_model_set_param_b(model: *mut c_void, name: *const c_char, enable: i32) -> MDStatus;
    pub fn md_model_set_param_s(model: *mut c_void, name: *const c_char, value: *const c_char) -> MDStatus;
    pub fn md_model_param_names(kind: MDModelKind, names: *mut *const c_char) -> MDStatus;
    pub fn md_model_param_type(kind: MDModelKind, name: *const c_char, type_out: *mut c_char) -> MDStatus;
}
```

> `MDStatus`、`MDModelKind`、`c_void`/`c_char` 类型沿用 `ffi.rs` 现有声明（确认 `MDModelKind` 为 repr 枚举）。

- [ ] **Step 2: model.rs 新增方法**

在 `Model` 实现内新增（沿用现有 `check_status`/`MdError`）：

```rust
    pub fn set_param_int(&self, name: &str, value: i64) -> Result<()> {
        let cn = CString::new(name)?;
        check_status(unsafe { ffi::md_model_set_param_i(self.ptr, cn.as_ptr(), value) })
    }
    pub fn set_param_double(&self, name: &str, value: f64) -> Result<()> { /* md_model_set_param_d */ }
    pub fn set_param_bool(&self, name: &str, value: bool) -> Result<()> { /* md_model_set_param_b, value as i32 */ }
    pub fn set_param_str(&self, name: &str, value: &str) -> Result<()> { /* md_model_set_param_s */ }

    pub fn param_names(&self) -> Result<Vec<String>> {
        let mut p: *const c_char = std::ptr::null();
        check_status(unsafe { ffi::md_model_param_names(self.kind, &mut p) })?;
        if p.is_null() { return Ok(vec![]); }
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        Ok(s.split('|').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect())
    }
    pub fn param_type(&self, name: &str) -> Result<char> {
        let cn = CString::new(name)?;
        let mut t: c_char = 0;
        check_status(unsafe { ffi::md_model_param_type(self.kind, cn.as_ptr(), &mut t) })?;
        Ok(t as char)
    }
```

> `self.kind` 需存在于 `Model`；若现有结构未存 kind，需在构造时保存（参考现有 `Model` 持有 kind 的字段）。

- [ ] **Step 3: cargo build + check 验证**

Run: `cargo build`（在 `E:\CLionProjects\ModelDeploy\rust\modeldeploy`）+ `cargo check`
Expected: 0 error。

- [ ] **Step 4: 提交**

```bash
git add rust
git commit -m "feat(rust): model pre/post-process param setter + introspection"
```

---

### Task 5: 端到端验证 + 回归 + 收尾

**Files:**
- Test: `tests/test_capi.cpp`（已有）、`csharp/ModelDeployUnitTest`、`rust/modeldeploy/tests`
- Docs: 更新 `docs/capi_bindings_analysis.md`（如存在）或 README 的 C API 一览（如被维护）

**Interfaces:**
- Consumes: Task 2/3/4 全部产出

- [ ] **Step 1: 全量回归（CPU build）**

Run:
```
cd E:\CLionProjects\ModelDeploy\build_tdc
.\bin\test_modeldeploy.exe "~[model] ~[gpu]"
```
Expected: 全部通过（回归 304+ 断言不受影响）。

- [ ] **Step 2: C# 冒烟**

在该科添加一个在 `ParamNames()` 中有参数的模型 kind（如 det）——若环境必须真实模型则用 `[model]`；至少验证 `ParamNames()/ParamType()` 自省在无模型时（用 kind）正确。运行相关 C# 测试或 `dotnet build` + 手工调用。

- [ ] **Step 3: Rust 冒烟**

`cargo build` + `cargo test`（自省用例若可离线运行则跑，否则编译验证）。

- [ ] **Step 4: 提交（如有遗留测试/文档）**

```bash
git add .
git commit -m "test: end-to-end param introspection across capi/C#/Rust"
```

（若无需新提交则跳过。）

---

## Self-Review

**Spec 覆盖：**
- 4 个类型化 setter ✓ Task 2
- 自省 names/type ✓ Task 2
- 覆盖所有 kind（含 pipeline）✓ Task 2 分发表 + 空列表
- C++ 缺失 setter 补齐 ✓ Task 1（face_rec_pipeline getter）
- C# kv + 自省驱动包装 ✓ Task 3
- Rust kv + 自省 ✓ Task 4
- 新增 MD_ERR_INVALID_TYPE ✓ Task 2 Step 1
- 端到端验证 + 回归 ✓ Task 5

**占位符扫描：** `apply_model_param`/`param_type_of` 骨架给出，但要求实现者按分发表补全——表在计划中已完整列出所有 kind/name/类型/setter，非占位符，是"大表补全"性质（符合 No Placeholders：给出了每项表格）。

**类型一致性：** `md_model_set_param_i/d/b/s`、`md_model_param_names/type` 在 Task 2（C 定义）与 Task 3/4（C#/Rust 声明）一致；参数名（conf_threshold 等）三处一致。`PT_I/PT_D/PT_B/PT_S` 与自省字符 'I'/'D'/'B'/'S' 一致。

> **已知需在 Task 2 实现时最终确认的两点**（实现者应核对现有源码而非臆测）：
> - `UltralyticsPose` 的 postprocessor 是否有 `set_keypoints_num`（盘点见 pose/postprocessor.h 有）；`UltralyticsSeg` 有 `set_mask_threshold`。
> - `InsightFaceAnalysis::set_det_thresh`、`PedestrianAttribute::set_det_threshold` 均存在（盘点确认）。
