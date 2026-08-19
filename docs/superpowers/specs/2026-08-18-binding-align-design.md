# C# / Rust 绑定对齐 C++ predict(ImageData) 设计

日期：2026-08-18
状态：待评审

## 背景

C++ SDK 的权威语义是**每个视觉模型一个统一入口 `predict(const ImageData&, RESULT*, ...)`**，
NV12/设备帧由 `ImageData` 类型承载（`from_planes/from_raw/plane(i)/device()`），
模型内部按 `device() != Device::CPU` 分派零拷贝路径；设备帧须模型 `use_gpu()`。

绑定侧不一致：
- **C#**：仅 `DetectionModel` 有 `PredictNv12(byte[] y, byte[] uv, ...)` 与
  `PredictNv12WithFrame(...) → (Prediction, VisionImage)` 元组变体 —— C++ 中不存在的增殖 API；
  由此引入 `GCHandle` 手动 pin、`FromDeviceFrame(IntPtr)` 句柄泄露、双 Dispose。
  `md_model_predict_batch` / `md_image_plane_ptrs` 已声明但安全层未暴露。
- **Rust**：虽已是统一 `predict(&Image)`，但 `Image` 缺 `format()/device()/plane_count()/plane(i)`；
  `from_nv12/from_device_nv12` 未对齐 `from_planes`；`plane_ptrs()` 返回裸指针元组；
  `from_device_frame(raw)` 泄露 FFI 层；批量 `predict_batch` 安全层未暴露；
  `SeetaFaceAge/Gender` 返回 `Vec<i32>`（应为单值 `i32`）；`README.md` 过期。

## 目标

对齐 C++ 语义，一次收敛 C# 与 Rust：

1. 每个视觉模型**唯一入口** `Predict(VisionImage)` / `predict(&Image)`。
2. NV12/设备帧通过 Image 类型的**指针/工厂**承载；设备指针零拷贝、免 pin。
3. 撤掉所有 Nv12 预测变体与 `(Prediction, Frame)` 元组。
4. 补统一批量入口 `PredictBatch` / `predict_batch`（平铺返回）。
5. 标量模型规范为单值返回。
6. 模型 `use_gpu` 才走设备零拷贝，否则照 C++ 报错不静默回退（保持）。

## 设计

### A. C# `VisionImage`（对齐 `ImageData`）

- 新增只读属性：`MdImageType Type`、`Device Device`、`int PlaneCount`。
- 新增 `Plane` 访问：`Plane GetPlane(int i)`（`IntPtr Data` + `int Step`），
  经 `md_image_plane_ptrs` 安全暴露，生命周期绑定本对象。
- 设备工厂（零拷贝、免 pin、借用外部指针）：
  `static VisionImage FromDeviceNv12(IntPtr y, IntPtr uv, int w, int h, int stepY, int stepUv, Device dev)`
  语义同 C++ `from_planes(..., Dev, 无 owner)`；调用方保证指针存活。
- host 便捷工厂（保留）：`FromNv12Data(byte[] y, byte[] uv, ...)`，
  内部自动 pin + `Dispose` 释放，把 pin 账本藏进 `VisionImage`。
- 移除公开 `FromDeviceFrame(IntPtr)`（原生句柄仅内部/FFI 使用）。
- 修正 `Channels`（不再硬编码 3，由 `Type` 推导）。

### B. C# `BaseModel` / Models

- 删除 `MakePredictionNv12` / `MakePredictionNv12WithFrame`（元组）。
- 删除 `DetectionModel.PredictNv12` / `PredictNv12WithFrame`。
- 全模型唯一入口 `Predict(VisionImage)`。
- 新增 `PredictBatch(IEnumerable<VisionImage>)`，接 `md_model_predict_batch`，返回平铺 `Prediction<T>`。
- 标量模型（FaceAge/Gender → int、FaceRec → 结构）维持各自 C++ 单值语义；**不给标量提供 batch**。

### C. Rust `Image`（对齐 `ImageData`）

- 新增 `format()`、`device()`、`plane_count()`、`plane(i) -> Result<Plane>`（`Plane{ data: *const u8, step: i32 }`，生命周期绑定 `&self`）。
- `from_device_nv12(y: *const u8, uv: *const u8, w, h, step_y, step_uv, dev: MDDevice)` —— 设备指针零拷贝。
- 保留 `from_nv12(&[u8], &[u8], w, h, step_y, step_uv)` host。
- `from_device_frame(raw)` 降为内部（crate 私有），不公开在安全层。
- `plane_ptrs()` 改用新 `plane(i)` 收敛裸指针元组。

### D. Rust `model.rs`

- 新增 `predict_batch(&self, imgs: &[&Image]) -> Result<Vec<Item>>`，接 ffI `md_model_predict_batch`。
- 标量规范：`SeetaFaceAge` / `SeetaFaceGender` 改返回 `i32`（去 `Vec` 包装）。
- 修正过期 `rust/README.md`。（README 属文档，不入安全层代码，一并更新）

### E. 批量平铺语义

C++ 各模型 `batch_predict` 结果形状不一（det 每图一组框、cls 每图一个）。
绑定统一为**平铺 `Vec<Item>` / `Prediction<T>`**（丢图片边界），保持跨模型一致。

### F. 测试

- C#：`FromDeviceNv12`+`Predict`、`PredictBatch`、`GetPlane`、host `FromNv12Data` 生命周期。
- Rust：`tests/integration_test.rs` 用例 + README 用法示例。

## 非目标

- 不改 C++/capi2 原生层语义（`md_image_from_device_nv12` 借指针、`from_planes` 保持）。
- 不新增人脸反欺骗（FACE_AS）C# 包装类（当前本就无，超出本次范围）。
- 不改 OCR 具体模型的 batch 分组细节（统一平铺）。

## 风险

- 指针工厂是"借用"，调用方须保证设备/托管内存存活；用内嵌 pin / owner 与文档双重约束。
- `predict_batch` 平铺会丢图片边界，属有意的跨模型一致性取舍，文档需说明。
- 改的是公共 API（删除方法/改签名），C#/Rust 属破坏性变更，需同步更新示例与测试。
