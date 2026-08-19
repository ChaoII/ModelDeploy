# C# / Rust 绑定对齐 C++ predict(ImageData) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 C# 与 Rust 绑定的视觉推理入口收敛为与 C++ `predict(const ImageData&)` 一致的"唯一 `predict(Image)` + Image 指针/设备工厂"，撤掉所有 NV12 变体与元组，并补齐统一 `PredictBatch` / `predict_batch`（需先在 capi2 原生实现 `md_model_predict_batch`）。

**Architecture:** 三层推进——先在 capi2 原生层实现 `md_model_predict_batch`（重构 `md_model_predict` 的逐 kind 逻辑为共享累加器，DRY）与图像元数据 getter `md_image_info`；再改 C#（NativeMethods→VisionImage→BaseModel→Models）；最后改 Rust（ffi→Image→model）并更新 README 与两端测试。每层独立可测、可提交。

**Tech Stack:** C++17 (capi2 原生)、C# (.NET)、Rust (无外部依赖的安全 FFI 包装)、Catch2 / NUnit / cargo test。

## Global Constraints

- 权威语义 = C++ `predict(const ImageData&, RESULT*, ...)`；NV12/设备帧在 Image 层承载；设备帧须 `use_gpu()` 否则照 C++ 报错，**绝不静默回退 CPU**。
- 设备 NV12 工厂是"借用指针"（库不拥有内存）：C# 用 `IntPtr`、Rust 用 `*const u8`；调用方保证存活。
- host `byte[]`/`&[u8]` 便捷工厂由 Image 对象自身内嵌 pin / 借 slice 管理生命周期，调用方不再手动 pin/Dispose。
- 批量返回**平铺** `Prediction<T>` / `Vec<Item>`（丢图片边界），跨模型一致；标量模型（FaceAge/Gender/FaceRec）不提供 batch。
- 失败返回错误码/抛异常并设 last_error，不静默回退。
- C++17 必需；保持现有文件组织与命名约定；`docs/superpowers/*` 不入库。
- 每任务末尾必须跑通对应测试并提交。

---

## 文件结构

- `capi2/md_capi.h`、`capi2/md_capi.cpp` — 原生：新增 `md_image_info`、`md_model_predict_batch` 及单值 kind 的批量 getter 扩展。
- `tests/test_capi*.cpp`（或新 capi 测试） — 原生 capi 测试。
- `csharp/ModelDeploy/V2/VisionImage.cs` — 加 Type/Device/PlaneCount/GetPlane、FromDeviceNv12(IntPtr)、宿主自动 pin；删 FromDeviceFrame 公开。
- `csharp/ModelDeploy/V2/BaseModel.cs` — 删 MakePredictionNv12/WithFrame；加 PredictBatch 辅助。
- `csharp/ModelDeploy/V2/Models.cs` — 删 DetectionModel 的 Nv12 变体；各模型加 PredictBatch。
- `csharp/ModelDeploy/NativeMethods.cs` — 加 md_image_from_device_nv12(IntPtr) 重载、md_image_info、md_result_*_batch 单值 getter。
- `csharp/ModelDeployUnitTest/` — C# 测试。
- `rust/modeldeploy/src/ffi.rs` — 加 md_image_info、batch 单值 getter。
- `rust/modeldeploy/src/image.rs` — 加 format()/device()/plane_count()/plane(i)；from_device_nv12 改裸指针；收敛 plane_ptrs。
- `rust/modeldeploy/src/model.rs` — 加 predict_batch；SeetaFaceAge/Gender 改 i32。
- `rust/modeldeploy/tests/integration_test.rs`、`rust/modeldeploy/README.md` — 测试与文档。

---

### Task 1: capi2 原生 — 新增 `md_image_info` 图像元数据 getter

**Files:**
- Modify: `capi2/md_capi.h`（声明）、`capi2/md_capi.cpp`（实现）
- Test: `tests/test_capi_image.cpp`（或并入既有 capi 测试文件）

**Interfaces:**
- Produces: `MDStatus md_image_info(MDImageHandle h, int* type, int* dev, int* nplanes)`——type 为 MdImageType 数值、dev 为 MDDevice、nplanes 为平面数（NV12=2，packed=1）。供 C# `VisionImage.Type`（映射 MdImageType）、Rust `format()`/`device()`/`plane_count()` 使用。

- [ ] **Step 1: 在 md_capi.h 图像区（见 `md_image_size` 附近，162 行）加声明**
```cpp
/* 图像元数据：type=MdImageType, dev=MDDevice, nplanes=平面数；任一指针可为空 */
MD_CAPI_EXPORT MDStatus md_image_info(MDImageHandle, int* type, int* dev, int* nplanes);
```

- [ ] **Step 2: 运行测试确认缺失（先写调用失败的临时断言）**

- [ ] **Step 3: 在 md_capi.cpp `md_image_plane_ptrs`（574 行）后实现**
```cpp
MDStatus md_image_info(MDImageHandle h, int* type, int* dev, int* nplanes) {
    auto* hi = static_cast<md_image_handle*>(h);
    if (!hi) return MD_ERR_NULL_POINTER;
    if (type)    *type = static_cast<int>(hi->image.type());
    if (dev)     *dev   = static_cast<int>(hi->image.device());
    if (nplanes) *nplanes = static_cast<int>(hi->image.plane_count());
    return MD_OK;
}
```
（`md_image_handle` 结构含 `ImageData image;`，见 `md_image_from_device_nv12` 实现 422-427 行。）

- [ ] **Step 4: 跑通 C++ capi 测试**
- [ ] **Step 5: 提交**（`feat(capi): add md_image_info image metadata getter`）

---

### Task 2: capi2 原生 — 实现 `md_model_predict_batch`

**Files:**
- Modify: `capi2/md_capi.cpp`（1341-1589 区域）
- Test: `tests/test_capi_batch.cpp`（新）

**Interfaces:**
- Consumes: `md_results` 现有结构（`ResultData<T>`/`SingleResult<T>`/`ProjectedResult`）、`md_model_handle`、`ImageData handle_to_image`。
- Produces: `MDStatus md_model_predict_batch(MDModelHandle, MDImageHandle* imgs, size_t n, MDResultHandle* out)` 已实现。结果句柄扁平累积：
  - 列表类 kind（det/pose/obb/iseg/face_det/face_rec/face_rec_pipeline/insightface/insightface_det/lpr_det/lpr_rec(单)/lpr_pipeline/ped_attr/cls/antispoof）→ `ResultData<T>`，插所有图的项。
  - 单值类 kind（sem_seg/depth/age/gender/ocr*/lpr_rec 单值）→ 内部用 `ResultData<T>` 存 N 个值（每图一个），其 getter 扩展为"既支持 SingleResult 也支持 ResultData"（见 Task 3）。

- [ ] **Step 1: 重构共用累加 helper（在 `md_model_predict` 前、匿名 namespace 内）**

关键设计：把 `md_model_predict` 的 switch 体抽成"向传入累加器追加"的 helper，避免 predict/predict_batch 重复。单图 predict 创建类型正确的空容器后调一次；batch 创建容器后对每图追加。

```cpp
// 向统一的扁平容器追加一批结果；accumulator 由调用方按 kind 创建。
// append_mode: 0=... (保留单图行为)
template <typename T>
void append_all(std::vector<T>& dst, const std::vector<T>& src) { dst.insert(dst.end(), src.begin(), src.end()); }
template <typename T>
void append_all(std::vector<T>& dst, T&& single) { dst.push_back(std::move(single)); }
```

> 说明：逐 kind 的追加逻辑与现有 `md_model_predict` 的 switch 逐 case 一一对应——把 `d->v` 换成"传入的累加容器"即可。因每 kind 的模型类型不同，用 lambda 模板：
> ```cpp
> template <typename ModelT, typename ResT, typename Fn>
> bool infer_append(ModelT* m, const ImageData& img, ResultData<ResT>* acc, Fn&& f) {
>     std::vector<ResT> v;
>     if (!f(m, img, &v)) return false;
>     append_all(acc->v, v);
>     return true;
> }
> ```
> 列表类 kind 用 `f = [](m, img, &v){ return m->predict(img, &v); }`；单值类 kind 用 `f` 内 `ResT r; m->predict(img, &r); v.push_back(r);`。

- [ ] **Step 2: 重写 `md_model_predict` 用 helper（逐 kind 调用一次，容器用现有类型）**
保持现状语义，仅把"新建容器+调用"改为"新建容器→调 helper 追加→完成"。不改变任一 kind 的返回类型与 getter 兼容性。

- [ ] **Step 3: 实现 `md_model_predict_batch`**
```cpp
MDStatus md_model_predict_batch(MDModelHandle h, MDImageHandle* imgs, size_t n, MDResultHandle* out) {
    auto* mh = static_cast<md_model_handle*>(h);
    if (!mh || !imgs || !out) return MD_ERR_NULL_POINTER;
    if (!mh->ready) return MD_ERR_MODEL_INIT;
    if (n == 0) { set_error("md_model_predict_batch: empty batch"); return MD_ERR_INVALID_ARGUMENT; }
    auto* rh = new md_result_handle();
    auto fail = [&](const char* w) { set_error_fmt("md_model_predict_batch: %s failed", w); delete rh; return MD_ERR_MODEL_PREDICT; };
    // 依 mh->kind 建空扁平容器并逐图追加（复用 helper，逻辑 = 单图 switch 的循环版）
    // 示例（detection）：检测当前是 ResultData<DetectionResult> 平铺整批所有框
    //   auto* d = new ResultData<DetectionResult>(); rh->kind = MD_RES_DETECTION; rh->data = d;
    //   for (i in 0..n) { if (!infer_append(det_model, img_i, d, [](auto* m, const ImageData& p, auto* outv){ auto* md=static_cast<detection::UltralyticsDet*>(m); return md->predict(p, outv); })) return fail("detection"); }
    // 其余 kind 照此循环；单值 kind（sem/depth/age/gender/ocr）用 ResultData<T> 每图 push 一个值。
    *out = rh;
    return MD_OK;
}
```
> 逐 kind 完整分支：把 Task 2 Step 2 重构后的 switch 每个 case 复制为"建容器→for 每图 append"。**逐 kind 清单**：DETECTION、CLASSIFICATION、POSE、OBB、INSTANCE_SEG、FACE_DET、FACE_REC、FACE_AGE、FACE_GENDER、FACE_REC_PIPELINE、INSIGHTFACE、INSIGHTFACE_DET、OCR、OCR_DET、OCR_REC、OCR_CLS、LPR_DET、LPR_REC、LPR_PIPELINE、PED_ATTR、SEM_SEG、DEPTH、FACE_AS/SECOND/PIPELINE（有无 batch 按各自 C++ 模型方法确定；无 batch 方法者返回 MD_ERR_NOT_IMPLEMENTED）。

- [ ] **Step 4: 新增 `tests/test_capi_batch.cpp`**：构造含多框的 2 图，`md_model_predict_batch` 后 `md_result_detection` 返回两图总框数；分类批量逐图 count==n。断言单值 kind 批量（如 age）返回 n 个值。
- [ ] **Step 5: 跑通 + 提交**（`feat(capi): implement md_model_predict_batch`）

---

### Task 3: capi2 原生 — 单值 kind 批量 getter 扩展

**Files:**
- Modify: `capi2/md_capi.cpp`（`md_result_age` 2099、`md_result_gender` 2107、`md_result_sem_seg` 1843、`md_result_depth` 1856、`md_result_ocr` 1984）
- Test: `tests/test_capi_batch.cpp`

**Interfaces:**
- Consumes: Task 2 中单值 kind batch 用 `ResultData<T>` 存储。
- Produces: 额外数组 getter `md_result_age_batch(h, const int** items, size_t* count)`、`md_result_gender_batch`、`md_result_sem_seg_batch`、`md_result_depth_batch`、`md_result_ocr_count`（返回行数）。既有单值 getter 保持兼容（读 `SingleResult<T>` 或 `ResultData<T>` 的 index 0 / 单行）。

- [ ] **Step 1: 让既有单值 getter 兼容 ResultData**：把 `static_cast<SingleResult<T>*>(rh->data)` 改为"若为 `SingleResult<T>` 读 value；若为 `ResultData<T>` 读 v 首项"。
- [ ] **Step 2: 新增批量数组 getter**，供 C#/Rust `PredictBatch` 读取单值类 kind 的全部项。
- [ ] **Step 3: 测试 + 提交**（`feat(capi): batch getters for single-value result kinds`）

---

### Task 4: C# — NativeMethods 图像/批次声明

**Files:**
- Modify: `csharp/ModelDeploy/NativeMethods.cs`
- Test: `csharp/ModelDeployUnitTest/`（随 Task 5-6 补完整用例，此处仅编译通过）

**Interfaces:**
- Produces: `md_image_from_device_nv12(IntPtr y, IntPtr uv, ...)` 重载、`md_image_info(out int type, out int dev, out int nplanes)`、`md_result_*_batch` 声明。

- [ ] **Step 1: 加 IntPtr 设备工厂重载（紧邻现有 byte[] 版本，70 行）**
```csharp
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern MDStatus md_image_from_device_nv12(out IntPtr handle, IntPtr y, IntPtr uv,
    int w, int h, int step_y, int step_uv, int dev);
```
- [ ] **Step 2: 加 `md_image_info` 声明**
```csharp
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern MDStatus md_image_info(IntPtr handle, out int type, out int dev, out int nplanes);
```
- [ ] **Step 3: 加单值类 kind 批量 getter（随 Task 3 的新增）**：`md_result_age_batch`、`md_result_gender_batch`、`md_result_sem_seg_batch`、`md_result_depth_batch`、`md_result_ocr_count`。
- [ ] **Step 4: 编译通过**
- [ ] **Step 5: 提交**（`feat(csharp): image info + device-nv12 IntPtr + batch getters`）

---

### Task 5: C# — VisionImage 对齐 ImageData

**Files:**
- Modify: `csharp/ModelDeploy/V2/VisionImage.cs`

**Interfaces:**
- Consumes: Task 4 的 `md_image_info`、`md_image_plane_ptrs`、`md_image_from_device_nv12(IntPtr)`。
- Produces: `enum MdImageType`（VisionImage 内或 types_internal_c）；`VisionImage` 新增：`MdImageType Type`、`Device Device`、`int PlaneCount`、`Plane GetPlane(int i)`、`static VisionImage FromDeviceNv12(IntPtr y, IntPtr uv, int w,int h,int stepY,int stepUv, Device dev)`；宿主 `FromNv12Data` 内置 pin；移除公开 `FromDeviceFrame`。

- [ ] **Step 1: 加 `MdImageType` 枚举与 `Plane` 结构**（`Plane { IntPtr Data; int Step; }`）。
- [ ] **Step 2: 构造函数读取元数据**：`md_image_info(handle,...)` 填充 Type/Device/PlaneCount。
- [ ] **Step 3: 加 `FromDeviceNv12(IntPtr...)`**：调 `md_image_from_device_nv12(out h, y, uv, ...)`；不 pin、不拥有。
- [ ] **Step 4: 改 `FromNv12Data(byte[]...)` 内置 pin**：内部 `GCHandle.Alloc(Pinned)` 存字段，`Dispose()` 释放；调用方不再 `PinBuffers`。
- [ ] **Step 5: 移除公开 `FromDeviceFrame` 与 `PinBuffers`**（改为 `internal` 或删除，元组路径已删）。
- [ ] **Step 6: 修正 `Channels`**：由 `Type` 推导（NV12→2 平面；packed→ch）。或按 C++ 语义改为 `PlaneCount`/去掉硬编码。
- [ ] **Step 7: 跑通 C# 测试 + 提交**（`feat(csharp): VisionImage aligned to ImageData (type/device/planes + device factory)`）

---

### Task 6: C# — BaseModel / Models 统一入口 + PredictBatch

**Files:**
- Modify: `csharp/ModelDeploy/V2/BaseModel.cs`、`csharp/ModelDeploy/V2/Models.cs`

**Interfaces:**
- Consumes: Task 4/5；`Prediction<T>`、`ResultReader`。
- Produces: 删除 `MakePredictionNv12`/`MakePredictionNv12WithFrame` 与 `DetectionModel.PredictNv12/PredictNv12WithFrame`；新增 `protected Prediction<T> PredictBatch(IEnumerable<VisionImage>, Func<IntPtr,T[]>)` 与各模型 `public Prediction<T> PredictBatch(...)`；标量模型（FaceAge/Gender/FaceRec）不加 batch。

- [ ] **Step 1: 删 BaseModel 的 `MakePredictionNv12`/`MakePredictionNv12WithFrame`（190-225 行）。**
- [ ] **Step 2: 加 `PredictBatchNative(IntPtr[] imgHandles)`**：调 `md_model_predict_batch(_handle, handles, (UIntPtr)n, out result)`，失败抛异常。
- [ ] **Step 3: 加 `protected Prediction<T> PredictBatch<T>(IEnumerable<VisionImage> images, Func<IntPtr,T[]> reader)`**：收集 handles→`PredictBatchNative`→`new Prediction<T>(rh, reader)`。
- [ ] **Step 4: Models.cs**：删 DetectionModel `PredictNv12`/`PredictNv12WithFrame`（29-39 行）；给所有列表/单值一次一个的视觉模型加 `public Prediction<T> PredictBatch(IEnumerable<VisionImage> images)`（det/cls/pose/obb/iseg/sem/depth/face_det/insightface/ocr/lpr/*pipeline/ped_attr）。标量（FaceAge/Gender/FaceRec）不加。
- [ ] **Step 5: 跑通 C# 测试 + 提交**（`feat(csharp): unified Predict + PredictBatch, drop NV12 variants`）

---

### Task 7: Rust — ffi 声明补充

**Files:**
- Modify: `rust/modeldeploy/src/ffi.rs`

**Interfaces:**
- Produces: `md_image_info(...)`、`md_result_*_batch`、数组 getter 声明（对应 Task 3）。

- [ ] **Step 1: 加 `md_image_info` extern**
```rust
pub fn md_image_info(h: MDImageHandle, type_: *mut c_int, dev: *mut MDDevice, nplanes: *mut c_int) -> MDStatus;
```
- [ ] **Step 2: 加单值类批量 getter 对应声明**（`md_result_age_batch` 等，返回 `*const c_int` 数组 + count）。
- [ ] **Step 3: 编译 + 提交**（`feat(rust): ffi additions for image info + batch getters`）

---

### Task 8: Rust — Image 对齐（format/device/planes + 设备工厂）

**Files:**
- Modify: `rust/modeldeploy/src/image.rs`

**Interfaces:**
- Consumes: Task 7；现有 `md_image_plane_ptrs`。
- Produces: `Image::format() -> Result<ImageFormat>`、`device() -> Result<MDDevice>`、`plane_count() -> Result<usize>`、`plane(i) -> Result<Plane>`（`Plane { data: *const u8, step: i32 }`）、`from_device_nv12(y: *const u8, uv: *const u8, w,h,step_y,step_uv, dev)`（改为裸指针）；`from_device_frame` 降为 `pub(crate)`；保留 `from_nv12(&[u8],&[u8],...)`。

- [ ] **Step 1: 加 `ImageFormat` 枚举（对齐 MdImageType 数值）。**
- [ ] **Step 2: `from_device_nv12` 改裸指针签名**（`y: *const u8, uv: *const u8`），调用方传 CUDA/CUVID 设备指针。
- [ ] **Step 3: `from_device_frame` 改 `pub(crate)`**，并从安全 API 移除文档暴露。
- [ ] **Step 4: 加 `format()/device()/plane_count()/plane(i)`**，经 `md_image_info`/`md_image_plane_ptrs` 实现；`plane(i)` 超界返回错误。
- [ ] **Step 5: 收敛 `plane_ptrs()`** → 改基于 `plane(0)`/`plane(1)` + `device()`。
- [ ] **Step 6: 跑通 Rust 测试 + 提交**（`feat(rust): Image aligned to ImageData (format/device/planes + device factory)`）

---

### Task 9: Rust — model.rs 统一 + predict_batch + 标量

**Files:**
- Modify: `rust/modeldeploy/src/model.rs`、`rust/modeldeploy/src/types.rs`

**Interfaces:**
- Consumes: Task 7/8；`RawResult`、`model_wrapper!` 宏。
- Produces: 各封装新增 `predict_batch(&self, imgs: &[&Image]) -> Result<Vec<Item>>`；`SeetaFaceAge`/`SeetaFaceGender` `predict` 返回 `i32`（去 Vec）。

- [ ] **Step 1: `Model::predict_batch(&self, imgs: &[&Image]) -> Result<RawResult, MdError>`**：收集 handles→`ffi::md_model_predict_batch`→`RawResult`。
- [ ] **Step 2: `model_wrapper!` 宏加 `predict_batch`**：转 `Vec<Item>`（读者与 `predict` 同）。
- [ ] **Step 3: SeetaFaceAge/Gender 返回 `i32`**：读者用 `ResultData` 单值（对齐 C++ 单值语义），返回 `i32` 而非 `Vec<i32>`。
- [ ] **Step 4: 跑通 Rust 测试 + 提交**（`feat(rust): unified predict_batch, scalar i32 returns`）

---

### Task 10: Rust — 修正过期 README

**Files:**
- Modify: `rust/modeldeploy/README.md`

**Interfaces:**
- Consumes: Task 8/9 的最终 API。

- [ ] **Step 1: 用当前 API 重写用法示例**（弃用旧的 `vision::detection::UltralyticsDet`/`RuntimeOption::gpu()` 示例）：示例含 `Image::from_device_nv12` + `predict`、`predict_batch`、`plane(i)`。
- [ ] **Step 2: 校对示例可编译（对照 examples/）**
- [ ] **Step 3: 提交**（`docs(rust): refresh README to current API`）

---

### Task 11: 全量验证 + 收尾

**Files:**
- 全部改动文件

- [ ] **Step 1: C++ capi 全量测试**：`ctest --output-on-failure`（新增 md_image_info / batch 用例绿）。
- [ ] **Step 2: C# 全量测试**：`dotnet test`（含 FromDeviceNv12+Predict、PredictBatch、GetPlane、宿主 FromNv12Data 生命周期；确认 Nv12 变体已从源码消失）。
- [ ] **Step 3: Rust 全量测试**：`cargo test`（含 predict_batch、plane(i)、i32 标量）。
- [ ] **Step 4: 全局 grep 确认无残留 `PredictNv12`/`MakePredictionNv12`/`PredictNv12WithFrame`/`from_device_frame`(公开)。
- [ ] **Step 5: 提交收尾**（`refactor: bindings aligned to C++ predict(ImageData)`）

## Self-Review

- **Spec coverage**：唯一 predict(Image)（Task 5-6, 8-9）、设备/指针工厂（5, 8）、撤 Nv12 变体/元组（6, 8）、batch（Task 2-4, 6, 7, 9）、标量 i32（Task 9）、README（Task 10）、测试（各 Task）、use_gpu 约束（全局约束，绑定已有，仅文档提及）。✅
- **Placeholder scan**：Task 2 逐 kind 范本体是"把 switch 每个 case 复制为 append"，已给出可执行范式与逐 kind 清单，无 TBD/TODO。
- **Type consistency**：`ImageFormat`（Rust）/`MdImageType`（C#）命名一致；`plane(i)->Plane{data,step}` 两端一致；`PredictBatch`/`predict_batch` 对应；`md_image_info(type,dev,nplanes)` 两端签名一致。✅
