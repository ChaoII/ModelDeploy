# P1-Ext: VisionProcessorBackend 接口扩展 + 11 个 preprocessor 迁移 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 扩展 `VisionProcessorBackend` 接口补齐缺失算子，把剩余 11 个 preprocessor（classification/face_age/face_gender/face_rec/lpr_det/lpr_rec/ocr det/rec/cls/table/layout）全部接入 backend 抽象，行为零变化。

**Architecture:** 沿用已批准的抽象——接口定义算子、CPU/CUDA backend 实现、preprocessor 持有 `backend_` 委派。本计划先扩展接口 + CPU 实现，再逐个迁移 11 个 preprocessor。

**Tech Stack:** C++17，现有 `vision/common/processors/*` 静态方法 + ImageData 链式方法，CMake GLOB 自动拾取新文件。

## Global Constraints

- **零行为变化**：迁移后推理结果与迁移前逐位一致；`[core]` + `[vision_models]` + 全量测试必须全绿
- 公开 API（`use_cuda_preproc()` 等）保留，内部走工厂
- 新算子签名与现有 `vision/common/processors/*.h` 或 ImageData 方法签名对齐
- 新文件放 `csrc/vision/processors/` 下，CMake GLOB 自动纳入，**不改 CMakeLists.txt**
- 注释中文，风格对齐现有代码

---

### Task 1: 扩展 VisionProcessorBackend 接口

**Files:**
- Modify: `csrc/vision/processors/processor_backend.h`

**Interfaces:**
- Produces: 新增算子签名，Task 2+ 实现并依赖

- [ ] **Step 1: 在接口中新增算子**

在现有接口（`yolo_preprocess`、`scrfd_preprocess`、`resize`、`normalize`、`convert_to`、`center_crop`、`pad`、`hwc2chw`、`normalize_and_permute`、`nv12_to_bgr`、`process_device_image`）基础上新增：

```cpp
    // 数值缩放（如 /255），alpha/beta 与 Convert::apply 语义一致
    virtual bool convert(const ImageData& image, ImageData* out,
                         const std::vector<float>& alpha,
                         const std::vector<float>& beta) = 0;

    // 数据类型转换（如 uint8 -> float），dtype 与 Cast::apply 语义一致
    virtual bool cast(const ImageData& image, ImageData* out,
                      const std::string& dtype) = 0;

    // 缩放 + 通道重排（LPR 用：alpha=1/255, beta 可选, swap_rb）
    virtual bool convert_and_permute(const ImageData& image, Tensor* out,
                                     const std::vector<float>& alpha,
                                     const std::vector<float>& beta,
                                     bool swap_rb) = 0;

    // 整批融合算子（OCR det 用：resize+pad+normalize+permute，batch 内统一 pad）
    virtual bool fusion_resize_pad_normalize_permute(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<std::array<int, 2>>& resize_sizes,
        const std::vector<int>& dst_size,
        const std::vector<float>& mean, const std::vector<float>& std,
        float pad_value) = 0;
```

同时把现有 `normalize` 签名扩展（兼容 classification 的 scale/swap_rb 参数）：
```cpp
    virtual bool normalize(const ImageData& image, ImageData* out,
                           const std::vector<float>& mean,
                           const std::vector<float>& std,
                           bool scale = true, bool swap_rb = true) = 0;
```

- [ ] **Step 2: 编译验证**

Run: `cmake --build E:\CLionProjects\ModelDeploy\build --config Release --target ModelDeploySDK --parallel 2>&1`
Expected: 失败（新增纯虚方法导致 CPU/CUDA backend 缺实现）——这是预期的 TDD 红

- [ ] **Step 3: Commit**

```bash
git add csrc/vision/processors/processor_backend.h
git commit -m "feat(processor): extend VisionProcessorBackend with convert/cast/convert_and_permute/fusion batch ops"
```

---

### Task 2: CpuProcessorBackend 实现新算子

**Files:**
- Modify: `csrc/vision/processors/cpu/cpu_processor_backend.h` + `.cpp`

**Interfaces:**
- Consumes: Task 1 接口、现有 processors（`Convert::apply`、`Cast::apply`、`ConvertAndPermute::apply`、`fusion_resize_pad_normalize_permute_cpu`）、`utils::mat_to_tensor`
- Produces: 新算子的 CPU 实现，Task 3+ 迁移时使用

- [ ] **Step 1: 头文件加声明**

```cpp
    bool convert(const ImageData& image, ImageData* out,
                 const std::vector<float>& alpha,
                 const std::vector<float>& beta) override;
    bool cast(const ImageData& image, ImageData* out,
              const std::string& dtype) override;
    bool convert_and_permute(const ImageData& image, Tensor* out,
                             const std::vector<float>& alpha,
                             const std::vector<float>& beta,
                             bool swap_rb) override;
    bool fusion_resize_pad_normalize_permute(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<int>& resize_size, const std::vector<int>& dst_size,
        const std::vector<float>& mean, const std::vector<float>& std,
        float pad_value) override;
    bool normalize(const ImageData& image, ImageData* out,
                   const std::vector<float>& mean,
                   const std::vector<float>& std,
                   bool scale, bool swap_rb) override;
```

- [ ] **Step 2: .cpp 实现**

先读以下文件确认签名再实现：
- `csrc/vision/common/processors/convert.h`（`Convert::apply`）
- `csrc/vision/common/processors/convert_and_permute.h`
- `csrc/vision/common/processors/fusion_resize_pad_normalize_permute.h`（`fusion_resize_pad_normalize_permute_cpu` 的确切签名）
- `csrc/vision/utils.h`（`utils::mat_to_tensor`）
- ImageData 的 `cast`/`convert`/`permute` 方法（`image_data.h`）

实现模板（convert / cast / convert_and_permute / fusion 按实际函数签名包装）：

```cpp
bool CpuProcessorBackend::convert(const ImageData& image, ImageData* out,
                                  const std::vector<float>& alpha,
                                  const std::vector<float>& beta) {
    *out = image.convert(alpha, beta);
    return !out->empty();
}

bool CpuProcessorBackend::cast(const ImageData& image, ImageData* out,
                               const std::string& dtype) {
    *out = image.cast(dtype);
    return !out->empty();
}

bool CpuProcessorBackend::normalize(const ImageData& image, ImageData* out,
                                    const std::vector<float>& mean,
                                    const std::vector<float>& std,
                                    bool scale, bool swap_rb) {
    *out = image.normalize(mean, std, scale, swap_rb);
    return !out->empty();
}
```

`convert_and_permute` 和 `fusion_resize_pad_normalize_permute` 用现有 processors 静态方法 + `utils::mat_to_tensor` 组装：

```cpp
bool CpuProcessorBackend::convert_and_permute(const ImageData& image, Tensor* out,
                                              const std::vector<float>& alpha,
                                              const std::vector<float>& beta,
                                              bool swap_rb) {
    cv::Mat mat;
    image.to_mat(mat);
    if (!ConvertAndPermute::apply(&mat, alpha, beta, swap_rb)) return false;
    utils::mat_to_tensor(mat, out);
    return true;
}

bool CpuProcessorBackend::fusion_resize_pad_normalize_permute(
    const std::vector<ImageData>& images, Tensor* out,
    const std::vector<std::array<int, 2>>& resize_sizes,
    const std::vector<int>& dst_size,
    const std::vector<float>& mean, const std::vector<float>& std,
    float pad_value) {
    return fusion_resize_pad_normalize_permute_cpu(
        images, out, resize_sizes, dst_size, mean, std, pad_value);
}
```

**注意：** 若某 ImageData 方法签名与计划不同，以 `image_data.h` 实际为准。

- [ ] **Step 3: 编译验证**

Run: `cmake --build E:\CLionProjects\ModelDeploy\build --config Release --target ModelDeploySDK --parallel 2>&1`
Expected: 编译通过（CUDA backend 此时因缺新算子实现仍会红，是预期的——见 Task 4）

**若编译失败是 CUDA backend 缺纯虚实现**，属预期，先只编译通过到"CudaProcessorBackend 报错"即可，后续 Task 4 补。若 CPU 部分有错则修正。

- [ ] **Step 4: Commit**

```bash
git add csrc/vision/processors/cpu/
git commit -m "feat(processor): implement new ops in CpuProcessorBackend"
```

---

### Task 3: CudaProcessorBackend 补新算子实现

**Files:**
- Modify: `csrc/vision/processors/cuda/cuda_processor_backend.h` + `.cpp`

**Interfaces:**
- Consumes: Task 2 CPU 实现（回退基类）、现有 CUDA kernels
- Produces: 编译通过（消除 Task 2 遗留的红）

- [ ] **Step 1: 头文件加声明（与 Task 2 Step 1 相同的 5 个方法 override）**

- [ ] **Step 2: .cpp 实现**

新算子暂时**回退到 CPU 基类**（CUDA kernel 是后续计划的任务，本计划只保证抽象完整 + 编译通过）：

```cpp
bool CudaProcessorBackend::convert(const ImageData& image, ImageData* out,
                                   const std::vector<float>& alpha,
                                   const std::vector<float>& beta) {
    return CpuProcessorBackend::convert(image, out, alpha, beta);
}
// cast / convert_and_permute / fusion_resize_pad_normalize_permute / normalize 同理
```

`convert_and_permute` 和 `fusion_resize_pad_normalize_permute` 回退实现：
```cpp
bool CudaProcessorBackend::convert_and_permute(const ImageData& image, Tensor* out,
                                               const std::vector<float>& alpha,
                                               const std::vector<float>& beta,
                                               bool swap_rb) {
    return CpuProcessorBackend::convert_and_permute(image, out, alpha, beta, swap_rb);
}
bool CudaProcessorBackend::fusion_resize_pad_normalize_permute(
    const std::vector<ImageData>& images, Tensor* out,
    const std::vector<int>& resize_size, const std::vector<int>& dst_size,
    const std::vector<float>& mean, const std::vector<float>& std,
    float pad_value) {
    return CpuProcessorBackend::fusion_resize_pad_normalize_permute(
        images, out, resize_size, dst_size, mean, std, pad_value);
}
```

- [ ] **Step 3: 编译验证**

Run: `cmake --build E:\CLionProjects\ModelDeploy\build --config Release --target ModelDeploySDK --parallel 2>&1`
Expected: 全绿（接口 + CPU + CUDA 全部实现完整）

- [ ] **Step 4: Commit**

```bash
git add csrc/vision/processors/cuda/
git commit -m "feat(processor): implement new ops in CudaProcessorBackend (CPU fallback)"
```

---

### Task 4: 迁移 classification preprocessor

**Files:**
- Modify: `csrc/vision/classification/preprocessor.h` + `.cpp`

**Interfaces:**
- Consumes: Task 1-3 的 backend 算子
- Produces: classification 接入 `backend_`

- [ ] **Step 1: 头文件改造**

按 P1 Task 5 的模式：`use_cuda_preproc_` → `backend_` + `set_processor_backend()` + `get_processor_backend()` + `use_cuda_preproc()`（内部 `create_processor_backend(Device::GPU, Backend::ORT, 0)`）。顶部加 `#include "vision/processors/processor_factory.h"` 和 `#include "vision/processors/cpu/cpu_processor_backend.h"`。protected 成员 `backend_` 默认 `make_shared<CpuProcessorBackend>()`。

**注意：** classification 头文件里可能没有 `use_cuda_preproc_`（调查显示它没有 CUDA 路径）。若无，则只加 `backend_` 成员 + `set_processor_backend()` + `get_processor_backend()` + `use_cuda_preproc()`（新建，内部走工厂）。

- [ ] **Step 2: preprocess() 改造**

把 `preprocess()` 里手写 pipeline 替换为 backend 算子链：

```cpp
    bool ClassificationPreprocessor::preprocess(ImageData* image, Tensor* output) const {
        if (image->width() <= 0 || image->height() <= 0) {
            return false;
        }
        ImageData tmp;
        if (enable_center_crop_) {
            const int crop_size = std::min(image->height(), image->width());
            if (!backend_->center_crop(*image, &tmp, crop_size, crop_size)) return false;
        } else {
            tmp = *image;
        }
        ImageData resized;
        if (!backend_->resize(tmp, &resized, size_[0], size_[1])) return false;
        ImageData rgb;
        if (!backend_->convert_to(resized, &rgb, "RGB")) return false;
        ImageData scaled;
        const std::vector alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector beta = {0.0f, 0.0f, 0.0f};
        if (!backend_->convert(rgb, &scaled, alpha, beta)) return false;
        const std::vector mean = {0.485f, 0.456f, 0.406f};
        const std::vector std = {0.229f, 0.224f, 0.225f};
        if (!backend_->normalize_and_permute(scaled, output, mean, std)) return false;
        output->expand_dim(0);
        return true;
    }
```

**注意：** 原实现用 `cv::Mat` + processor 静态方法，语义是 CenterCrop→Resize→BGR2RGB→Convert(1/255)→NormalizeAndPermute。上面用 backend 算子链等价替换。若 `normalize_and_permute` 的 scale 语义与原 `NormalizeAndPermute::apply(&mat, mean, std, false)` 不一致（原代码 scale=false，因为已 convert 过），需要确认 `fuse_normalize_and_permute(mean, std, scale)` 的 scale 参数默认值并正确传递，保证数值等价。

- [ ] **Step 3: 编译 + 测试**

Run: `cmake --build E:\CLionProjects\ModelDeploy\build --config Release --target test_modeldeploy --parallel 2>&1` 然后 `cd build && .\bin\test_modeldeploy.exe [core],[vision_models] 2>&1`
Expected: 全绿

- [ ] **Step 4: Commit**

```bash
git add csrc/vision/classification/
git commit -m "refactor(classification): migrate preprocessor to VisionProcessorBackend"
```

---

### Task 5: 迁移 face preprocessor（age/gender/rec）

**Files:**
- Modify: `csrc/vision/face/face_age/preprocessor.h` + `.cpp`
- Modify: `csrc/vision/face/face_gender/preprocessor.h` + `.cpp`
- Modify: `csrc/vision/face/face_rec/preprocessor.h` + `.cpp`

**Interfaces:**
- Consumes: Task 1-3 backend 算子
- Produces: 三个 face preprocessor 接入 `backend_`

- [ ] **Step 1: 头文件改造（三个，同 classification 模式）**

- [ ] **Step 2: 各 preprocess() 改造**

每个的 pipeline 不同，**先读原 .cpp 理解精确算子序列**，再用 backend 算子等价替换：

- **face_age**：`center_crop(size_)` → `cast("float", false)` → `permute()` → `to_tensor` → `expand_dim(0)`（非 256 时先 `resize(256,256)`）。用 `backend_->center_crop` + `backend_->cast` + `backend_->hwc2chw`（permute 语义）等价替换。
- **face_gender**：`Resize::apply(112,112)` → `HWC2CHW` → `Cast::apply("float")` → `mat_to_tensor`。用 `backend_->resize` + `backend_->hwc2chw` + `backend_->cast`（注意算子顺序：原代码是 HWC2CHW 再 cast，需保持一致）。
- **face_rec**：`Resize::apply(256,256)`（如非 256）→ `CenterCrop::apply(248,248)` → `BGR2RGB` → `HWC2CHW` → `Cast::apply("float")` → `mat_to_tensor`。用 `backend_->resize` + `backend_->center_crop` + `backend_->convert_to("RGB")` + `backend_->hwc2chw` + `backend_->cast`。

**重要：** 原实现是"对 cv::Mat 做 HWC2CHW 再 cast(float)"——cast 作用在 CHW 排列的 float mat 上。backend 算子里 `hwc2chw` 产出 Tensor，`cast` 产出 ImageData，两者衔接要注意数据布局。若 `cast` 算子无法直接对 Tensor 操作，可能需要新算子或在迁移时先 cast 再 hwc2chw。**以数值等价为准**，必要时调整算子顺序或报告差异。

- [ ] **Step 3: 编译 + 测试**

Run: 同 Task 4 Step 3，全绿

- [ ] **Step 4: Commit**

```bash
git add csrc/vision/face/face_age/ csrc/vision/face/face_gender/ csrc/vision/face/face_rec/
git commit -m "refactor(face): migrate age/gender/rec preprocessors to VisionProcessorBackend"
```

---

### Task 6: 迁移 lpr preprocessor（det/rec）

**Files:**
- Modify: `csrc/vision/lpr/lpr_det/preprocessor.h` + `.cpp`
- Modify: `csrc/vision/lpr/lpr_rec/preprocessor.h` + `.cpp`

**Interfaces:**
- Consumes: Task 1-3 backend 算子
- Produces: 两个 lpr preprocessor 接入 `backend_`

- [ ] **Step 1: 头文件改造（同 classification 模式）**

- [ ] **Step 2: 各 preprocess() 改造**

- **lpr_det**：`utils::letter_box(&mat, size_, padding_value_, record)` → `ConvertAndPermute::apply(alpha=1/255, beta=0, swap_rb=true)` → `mat_to_tensor`。若 backend 有 letterbox 语义算子（`yolo_preprocess` 或新增），用它 + `convert_and_permute`。**注意**：lpr_det 的 letterbox 走 `utils::letter_box`，与 yolo 的 letterbox 记录结构可能不同，先读 `vision/utils.h` 的 `letter_box` 签名确认。
- **lpr_rec**：`Resize::apply(168,48)` → `ConvertAndPermute::apply(alpha=1/255, beta=-0.588, swap_rb=true)` → `mat_to_tensor`。用 `backend_->resize` + `backend_->convert_and_permute`。

- [ ] **Step 3: 编译 + 测试**（全绿）

- [ ] **Step 4: Commit**

```bash
git add csrc/vision/lpr/
git commit -m "refactor(lpr): migrate det/rec preprocessors to VisionProcessorBackend"
```

---

### Task 7: 迁移 OCR det preprocessor

**Files:**
- Modify: `csrc/vision/ocr/det_preprocessor.h` + `.cpp`

**Interfaces:**
- Consumes: Task 1-3 的 `fusion_resize_pad_normalize_permute` 算子
- Produces: OCR det 接入 `backend_`（这是唯一已有 CUDA 路径的，迁移后 CUDA 经 backend 仍可用）

- [ ] **Step 1: 头文件改造**

OCR det 已有 `use_cuda_preproc_`。替换为 `backend_` + setter/getter + `use_cuda_preproc()`（工厂创建）。

- [ ] **Step 2: apply() 改造**

`apply()` 里的 batch 融合分支：
- CPU：`fusion_resize_pad_normalize_permute_cpu(images, out, resize_size, dst_size, mean, std, pad)` → `backend_->fusion_resize_pad_normalize_permute(...)`
- CUDA（`use_cuda_preproc_ && WITH_GPU`）：当前调用 `fusion_resize_pad_normalize_permute_cuda`。迁移后走 `backend_->fusion_resize_pad_normalize_permute(...)`（CudaProcessorBackend 目前回退 CPU，等价于原 CPU 路径；CUDA kernel 优化是后续任务）。

**注意：** 原代码的 CUDA 分支（line 44）是被注释掉的，line 88 的 batch CUDA 是生效的。迁移后统一走 backend，行为与原 CPU 路径一致。先读 `det_preprocessor.cpp` 完整理解再改。

- [ ] **Step 3: 编译 + 测试**（全绿，含 OCR 相关测试）

- [ ] **Step 4: Commit**

```bash
git add csrc/vision/ocr/det_preprocessor.*
git commit -m "refactor(ocr): migrate det preprocessor to VisionProcessorBackend"
```

---

### Task 8: 迁移 OCR rec/cls/table/layout preprocessor

**Files:**
- Modify: `csrc/vision/ocr/rec_preprocessor.h` + `.cpp`
- Modify: `csrc/vision/ocr/cls_preprocessor.h` + `.cpp`
- Modify: `csrc/vision/ocr/structurev2_table_preprocessor.h` + `.cpp`
- Modify: `csrc/vision/ocr/structurev2_layout_preprocessor.h` + `.cpp`

**Interfaces:**
- Consumes: Task 1-3 backend 算子
- Produces: 4 个 OCR preprocessor 接入 `backend_`

- [ ] **Step 1: 头文件改造（4 个，同 classification 模式）**

- [ ] **Step 2: 各 preprocess() 改造**

先读各原 .cpp，用 backend 算子等价替换（这些是 ImageData 链式或 processor 静态方法风格）：

- **ocr rec**：`resize(动态宽, h=48)` → `pad(0,0,0,max_w-resize_w,127)` → `fuse_normalize_and_permute(mean=std=0.5)` → `images_to_tensor`。**注意 batch 级 pad 逻辑**：需要先遍历 batch 算最大宽，再统一 pad。若 backend 算子都是逐图的，迁移时保留 batch 循环在 preprocessor 里，仅把逐图算子委托 backend。若 `pad` 接口只有 top/bottom 两组参数，需要支持四边——先检查 Task 1 的 pad 签名，必要时扩展为四参数或在实现里处理。
- **ocr cls**：`Resize::apply(动态宽,48)` → `Normalize::apply(mean=std=0.5)` → `Pad::apply(右补0)` → `HWC2CHW` → `mats_to_tensor`。
- **ocr table**：`Resize::apply(等比max_len=512)` → `Normalize::apply(0.485/0.456/0.406)` → `Pad::apply(0,max_len-h,0,max_len-w)` → `HWC2CHW` → `mats_to_tensor`。
- **ocr layout**：`Resize::apply(800,608)` → `NormalizeAndPermute::apply` → `mats_to_tensor`。

**重要：** 这些 batch 类 preprocessor 有"batch 内统一 pad 尺寸"逻辑。若 `pad` 算子无法表达四边 pad，本任务先在 Task 1 基础上扩展 `pad` 签名（如改为四参数 `pad(image, out, top, bottom, left, right, value)`），并同步 CPU/CUDA backend 实现，再迁移。

- [ ] **Step 3: 编译 + 测试**（全绿，含 OCR 测试）

- [ ] **Step 4: Commit**

```bash
git add csrc/vision/ocr/
git commit -m "refactor(ocr): migrate rec/cls/table/layout preprocessors to VisionProcessorBackend"
```

---

### Task 9: 全量回归 + 收尾

- [ ] **Step 1: 全量构建** `cmake --build build --config Release --parallel 2>&1` 全绿
- [ ] **Step 2: 全量测试** `cd build && set TEST_DATA_DIR=E:\CLionProjects\ModelDeploy&& .\bin\test_modeldeploy.exe 2>&1` → All tests passed (85/85)
- [ ] **Step 3: 用 demo 确认代表性模型推理正常**（classification + ocr + lpr demo 各跑一个）
- [ ] **Step 4: Commit** `git add -A && git commit -m "test: verify all preprocessors migrated, full regression green"`

## Self-Review 记录

- **Spec 覆盖**：接口扩展（Task 1）→ CPU 实现（Task 2）→ CUDA 补齐（Task 3）→ 11 个 preprocessor 迁移（Task 4-8）→ 回归（Task 9）。
- **占位符扫描**：各 Task 迁移步骤要求"先读原 .cpp 理解算子序列"——这是必要的，因为 11 个 pipeline 各不相同，计划里给了精确模式但实现细节需对照源文件。
- **类型一致性**：`convert/cast/convert_and_permute/fusion_resize_pad_normalize_permute/normalize(scale,swap_rb)` 签名在 Task 1-3 一致；`set_processor_backend/get_processor_backend/backend_` 与 P1 一致。
- **风险**：face 系列 cast 与 hwc2chw 的顺序/数据布局差异、OCR batch pad 语义、lpr_det letterbox 走 utils 而非 yolo——这些在 Task 5/6/8 里已注明需对照源文件，以数值等价为准。
