# P1: VisionProcessorBackend 抽象 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立前后处理算子抽象 `VisionProcessorBackend`，实现 CPU/CUDA 两个 backend，改造 5 个 yolo 系 preprocessor 从布尔开关切换到 backend 委派，行为零变化。

**Architecture:** `VisionProcessorBackend` 纯虚接口定义前后处理算子；`CpuProcessorBackend` 封装 ImageData 内置方法 + `yolo_preprocess_cpu`；`CudaProcessorBackend` 继承 CPU 实现并覆写 yolo 算子为 `yolo_preprocess_cuda`；工厂函数按 `Device/Backend` 创建。preprocessor 持有 `shared_ptr<VisionProcessorBackend>` 替代 `use_cuda_preproc_` 布尔开关。

**Tech Stack:** C++17, CMake GLOB_RECURSE（新文件自动纳入构建，无需改 CMake），CUDA kernel（现有 `yolo_preproc.cu`）

## Global Constraints

- **零行为变化**：重构后现有 Catch2 测试（`[core]` + `[vision_models]`）必须全绿，推理结果与重构前一致
- 现有公开 API `use_cuda_preproc()` 保留（demo/CAPI/pybind 都在用），内部改为创建 CudaProcessorBackend
- 新文件路径统一放在 `csrc/vision/processors/` 下
- CMake 用 `file(GLOB_RECURSE VISION_SOURCE csrc/vision/*.cpp)`，新 `.cpp` 自动纳入；**禁止**修改 CMakeLists.txt（除非绝对必要）
- `#include` 顺序遵循现有风格（标准库 → 第三方 → 项目内）
- 注释用中文，风格对齐现有代码（`//` 前缀，代码内不加无关注释）

---

### Task 1: VisionProcessorBackend 抽象接口

**Files:**
- Create: `csrc/vision/processors/processor_backend.h`

**Interfaces:**
- Consumes: `ImageData`（`csrc/vision/common/image_data.h`）、`Tensor`（`csrc/core/tensor.h`）、`LetterBoxRecord`（`csrc/vision/common/struct.h`）、`DetectionResult`（`csrc/vision/common/result.h`）、`Device/Backend`（`csrc/core/enum_variables.h`）
- Produces: `modeldeploy::vision::VisionProcessorBackend` 抽象类，Task 2-6 依赖

- [ ] **Step 1: 创建接口头文件**

```cpp
//
// Created by aichao on 2025/8/2.
//
#pragma once

#include <memory>
#include <vector>
#include "core/tensor.h"
#include "core/enum_variables.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision {

class VisionProcessorBackend {
public:
    virtual ~VisionProcessorBackend() = default;

    // YOLO 系融合算子（letterbox + resize + normalize + hwc2chw）
    virtual bool yolo_preprocess(const ImageData& image, Tensor* out,
                                 const std::vector<int>& dst_size,
                                 float pad_val, LetterBoxRecord* record) = 0;

    // NV12 直接输入（硬解码/摄像头常见格式）
    virtual bool yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                      const std::vector<int>& src_size,
                                      int step_y, int step_uv, Tensor* out,
                                      const std::vector<int>& dst_size,
                                      float pad_val, LetterBoxRecord* record) = 0;

    // 通用算子（输出中间图像，供多算子 pipeline 串联）
    virtual bool resize(const ImageData& image, ImageData* out,
                        int width, int height) = 0;
    virtual bool normalize(const ImageData& image, ImageData* out,
                           const std::vector<float>& mean,
                           const std::vector<float>& std) = 0;
    virtual bool convert_to(const ImageData& image, ImageData* out,
                            const std::string& dst_format) = 0;
    virtual bool center_crop(const ImageData& image, ImageData* out,
                             int width, int height) = 0;
    virtual bool pad(const ImageData& image, ImageData* out,
                     const std::vector<int>& top,
                     const std::vector<int>& bottom) = 0;
    virtual bool hwc2chw(const ImageData& image, Tensor* out) = 0;
    virtual bool normalize_and_permute(const ImageData& image, Tensor* out,
                                       const std::vector<float>& mean,
                                       const std::vector<float>& std) = 0;
    virtual bool nv12_to_bgr(const uint8_t* y, const uint8_t* uv,
                             int width, int height, ImageData* out) = 0;

    // 零拷贝窄口子：直接吃设备侧图像（硬解码路径专用）
    // device_image: Sophgo 实现为 sail::BMImage*；Cuda 实现为 cuda::GpuMat*；CPU 返回 false
    virtual bool process_device_image(void* device_image, int width, int height,
                                      Tensor* out, LetterBoxRecord* record) {
        (void)device_image; (void)width; (void)height; (void)out; (void)record;
        return false;
    }
};

} // namespace modeldeploy::vision
```

- [ ] **Step 2: 编译验证**

Run: `cmake --build E:\CLionProjects\ModelDeploy\build --config Release --target ModelDeploySDK --parallel 2>&1`
Expected: 编译通过（新头文件目前无使用者，仅验证无语法错误）

- [ ] **Step 3: Commit**

```bash
git add csrc/vision/processors/processor_backend.h
git commit -m "feat(processor): add VisionProcessorBackend abstract interface"
```

---

### Task 2: CpuProcessorBackend

**Files:**
- Create: `csrc/vision/processors/cpu/cpu_processor_backend.h`
- Create: `csrc/vision/processors/cpu/cpu_processor_backend.cpp`

**Interfaces:**
- Consumes: Task 1 的 `VisionProcessorBackend` 接口、`yolo_preprocess_cpu`（`csrc/vision/common/processors/yolo_preproc.h`）、`ImageData` 内置方法
- Produces: `CpuProcessorBackend` 类，Task 4 工厂使用

- [ ] **Step 1: 写头文件**

```cpp
//
// Created by aichao on 2025/8/2.
//
#pragma once

#include "vision/processors/processor_backend.h"

namespace modeldeploy::vision {

class CpuProcessorBackend : public VisionProcessorBackend {
public:
    CpuProcessorBackend() = default;
    ~CpuProcessorBackend() override = default;

    bool yolo_preprocess(const ImageData& image, Tensor* out,
                         const std::vector<int>& dst_size,
                         float pad_val, LetterBoxRecord* record) override;
    bool yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                              const std::vector<int>& src_size,
                              int step_y, int step_uv, Tensor* out,
                              const std::vector<int>& dst_size,
                              float pad_val, LetterBoxRecord* record) override;
    bool resize(const ImageData& image, ImageData* out,
                int width, int height) override;
    bool normalize(const ImageData& image, ImageData* out,
                   const std::vector<float>& mean,
                   const std::vector<float>& std) override;
    bool convert_to(const ImageData& image, ImageData* out,
                    const std::string& dst_format) override;
    bool center_crop(const ImageData& image, ImageData* out,
                     int width, int height) override;
    bool pad(const ImageData& image, ImageData* out,
             const std::vector<int>& top,
             const std::vector<int>& bottom) override;
    bool hwc2chw(const ImageData& image, Tensor* out) override;
    bool normalize_and_permute(const ImageData& image, Tensor* out,
                               const std::vector<float>& mean,
                               const std::vector<float>& std) override;
    bool nv12_to_bgr(const uint8_t* y, const uint8_t* uv,
                     int width, int height, ImageData* out) override;
};

} // namespace modeldeploy::vision
```

- [ ] **Step 2: 写实现**

```cpp
//
// Created by aichao on 2025/8/2.
//

#include "core/md_log.h"
#include "vision/processors/cpu/cpu_processor_backend.h"
#include "vision/common/processors/yolo_preproc.h"
#include "vision/common/processors/nv12_to_bgr.h"

namespace modeldeploy::vision {

bool CpuProcessorBackend::yolo_preprocess(const ImageData& image, Tensor* out,
                                          const std::vector<int>& dst_size,
                                          float pad_val, LetterBoxRecord* record) {
    return yolo_preprocess_cpu(image, out, dst_size, pad_val, record);
}

bool CpuProcessorBackend::yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                               const std::vector<int>& src_size,
                                               int step_y, int step_uv, Tensor* out,
                                               const std::vector<int>& dst_size,
                                               float pad_val, LetterBoxRecord* record) {
    return yolo_preprocess_nv12_cpu(src_y, src_uv, src_size, step_y, step_uv,
                                    out, dst_size, pad_val, record);
}

bool CpuProcessorBackend::resize(const ImageData& image, ImageData* out,
                                 int width, int height) {
    *out = image.resize(width, height);
    return !out->empty();
}

bool CpuProcessorBackend::normalize(const ImageData& image, ImageData* out,
                                    const std::vector<float>& mean,
                                    const std::vector<float>& std) {
    *out = image.normalize(mean, std);
    return !out->empty();
}

bool CpuProcessorBackend::convert_to(const ImageData& image, ImageData* out,
                                     const std::string& dst_format) {
    ColorConvertType type = ColorConvertType::CVT_PA_BGR2PA_RGB;
    if (dst_format == "RGB") {
        type = ColorConvertType::CVT_PA_BGR2PA_RGB;
    } else if (dst_format == "GRAY") {
        type = ColorConvertType::CVT_PA_BGR2GRAY;
    } else if (dst_format == "BGR") {
        *out = image.clone();
        return !out->empty();
    }
    *out = ImageData::cvt_color(image, type);
    return !out->empty();
}

bool CpuProcessorBackend::center_crop(const ImageData& image, ImageData* out,
                                      int width, int height) {
    *out = image.center_crop({width, height});
    return !out->empty();
}

bool CpuProcessorBackend::pad(const ImageData& image, ImageData* out,
                              const std::vector<int>& top,
                              const std::vector<int>& bottom) {
    *out = image.pad(top[0], bottom[0], top[1], bottom[1], 0.0f);
    return !out->empty();
}

bool CpuProcessorBackend::hwc2chw(const ImageData& image, Tensor* out) {
    image.to_tensor(out);
    return true;
}

bool CpuProcessorBackend::normalize_and_permute(const ImageData& image, Tensor* out,
                                                const std::vector<float>& mean,
                                                const std::vector<float>& std) {
    auto tmp = image.fuse_normalize_and_permute(mean, std);
    tmp.to_tensor(out);
    return true;
}

bool CpuProcessorBackend::nv12_to_bgr(const uint8_t* y, const uint8_t* uv,
                                      int width, int height, ImageData* out) {
    *out = ImageData(width, height, MdImageType::PKG_BGR_U8);
    if (out->empty()) return false;
    return nv12_to_bgr_cpu(y, uv, width, height, width, width, out->data());
}

} // namespace modeldeploy::vision
```

**注意：** 已核对真实签名——`nv12_to_bgr_cpu(src_y, src_uv, src_w, src_h, step_y, step_uv, dst_bgr)` 输出原始 `uint8_t*`；`MdImageType::PKG_BGR_U8` 表示打包 BGR；`ColorConvertType` 是 `enum class`，值为 `CVT_PA_BGR2PA_RGB`/`CVT_PA_BGR2GRAY`。

- [ ] **Step 3: 编译验证**

Run: `cmake --build E:\CLionProjects\ModelDeploy\build --config Release --target ModelDeploySDK --parallel 2>&1`
Expected: 编译通过

- [ ] **Step 4: Commit**

```bash
git add csrc/vision/processors/cpu/
git commit -m "feat(processor): add CpuProcessorBackend"
```

---

### Task 3: CudaProcessorBackend

**Files:**
- Create: `csrc/vision/processors/cuda/cuda_processor_backend.h`
- Create: `csrc/vision/processors/cuda/cuda_processor_backend.cpp`

**Interfaces:**
- Consumes: Task 2 的 `CpuProcessorBackend`、`yolo_preprocess_cuda`（`csrc/vision/common/processors/yolo_preproc.cuh`）
- Produces: `CudaProcessorBackend` 类，Task 4 工厂使用

- [ ] **Step 1: 写头文件**

```cpp
//
// Created by aichao on 2025/8/2.
//
#pragma once

#include "vision/processors/cpu/cpu_processor_backend.h"

namespace modeldeploy::vision {

// CUDA backend 继承 CPU 实现，仅覆写 yolo 系算子为 CUDA kernel
class CudaProcessorBackend : public CpuProcessorBackend {
public:
    CudaProcessorBackend() = default;
    ~CudaProcessorBackend() override = default;

    bool yolo_preprocess(const ImageData& image, Tensor* out,
                         const std::vector<int>& dst_size,
                         float pad_val, LetterBoxRecord* record) override;
    bool yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                              const std::vector<int>& src_size,
                              int step_y, int step_uv, Tensor* out,
                              const std::vector<int>& dst_size,
                              float pad_val, LetterBoxRecord* record) override;
};

} // namespace modeldeploy::vision
```

- [ ] **Step 2: 写实现**

```cpp
//
// Created by aichao on 2025/8/2.
//

#include "core/md_log.h"
#include "vision/processors/cuda/cuda_processor_backend.h"
#ifdef WITH_GPU
#include "vision/common/processors/yolo_preproc.cuh"
#endif

namespace modeldeploy::vision {

bool CudaProcessorBackend::yolo_preprocess(const ImageData& image, Tensor* out,
                                           const std::vector<int>& dst_size,
                                           float pad_val, LetterBoxRecord* record) {
#ifdef WITH_GPU
    return yolo_preprocess_cuda(image, out, dst_size, pad_val, record);
#else
    MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, fallback to cpu" << std::endl;
    return CpuProcessorBackend::yolo_preprocess(image, out, dst_size, pad_val, record);
#endif
}

bool CudaProcessorBackend::yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                                const std::vector<int>& src_size,
                                                int step_y, int step_uv, Tensor* out,
                                                const std::vector<int>& dst_size,
                                                float pad_val, LetterBoxRecord* record) {
#ifdef WITH_GPU
    return yolo_preprocess_nv12_cuda(src_y, src_uv, src_size, step_y, step_uv,
                                     out, dst_size, pad_val, record);
#else
    MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, fallback to cpu" << std::endl;
    return CpuProcessorBackend::yolo_preprocess_nv12(src_y, src_uv, src_size, step_y, step_uv,
                                                     out, dst_size, pad_val, record);
#endif
}

} // namespace modeldeploy::vision
```

**注意：** 实现前先读 `csrc/vision/common/processors/yolo_preproc.cuh` 确认 `yolo_preprocess_cuda` 和 `yolo_preprocess_nv12_cuda` 的确切签名。

- [ ] **Step 3: 编译验证**

Run: `cmake --build E:\CLionProjects\ModelDeploy\build --config Release --target ModelDeploySDK --parallel 2>&1`
Expected: 编译通过（当前 Windows 构建 WITH_GPU=ON，CUDA 路径会被编译）

- [ ] **Step 4: Commit**

```bash
git add csrc/vision/processors/cuda/
git commit -m "feat(processor): add CudaProcessorBackend (inherit CPU, override yolo ops)"
```

---

### Task 4: Processor 工厂

**Files:**
- Create: `csrc/vision/processors/processor_factory.h`
- Create: `csrc/vision/processors/processor_factory.cpp`

**Interfaces:**
- Consumes: Task 2/3 的 CPU/CUDA backend、`Device/Backend` 枚举
- Produces: `create_processor_backend(Device, Backend, int)` 工厂函数，Task 5/6 使用

- [ ] **Step 1: 写头文件**

```cpp
//
// Created by aichao on 2025/8/2.
//
#pragma once

#include <memory>
#include "core/enum_variables.h"
#include "vision/processors/processor_backend.h"

namespace modeldeploy::vision {

// 根据设备与后端创建对应的前后处理 backend
std::unique_ptr<VisionProcessorBackend> create_processor_backend(
    Device device, Backend backend, int device_id = -1);

} // namespace modeldeploy::vision
```

- [ ] **Step 2: 写实现**

```cpp
//
// Created by aichao on 2025/8/2.
//

#include "core/md_log.h"
#include "vision/processors/processor_factory.h"
#include "vision/processors/cpu/cpu_processor_backend.h"
#ifdef WITH_GPU
#include "vision/processors/cuda/cuda_processor_backend.h"
#endif

namespace modeldeploy::vision {

std::unique_ptr<VisionProcessorBackend> create_processor_backend(
    Device device, Backend backend, int device_id) {
    (void)backend;
    (void)device_id;
    switch (device) {
    case Device::GPU:
#ifdef WITH_GPU
        return std::make_unique<CudaProcessorBackend>();
#else
        MD_LOG_WARN << "GPU is not enabled, fallback to CPU processor backend." << std::endl;
        return std::make_unique<CpuProcessorBackend>();
#endif
    case Device::CPU:
    case Device::OPENCL:
    case Device::VULKAN:
    default:
        return std::make_unique<CpuProcessorBackend>();
    }
}

} // namespace modeldeploy::vision
```

- [ ] **Step 3: 编译验证**

Run: `cmake --build E:\CLionProjects\ModelDeploy\build --config Release --target ModelDeploySDK --parallel 2>&1`
Expected: 编译通过

- [ ] **Step 4: Commit**

```bash
git add csrc/vision/processors/processor_factory.h csrc/vision/processors/processor_factory.cpp
git commit -m "feat(processor): add processor backend factory"
```

---

### Task 5: 改造 5 个 yolo 系 preprocessor

**Files:**
- Modify: `csrc/vision/detection/preprocessor.h` + `.cpp`
- Modify: `csrc/vision/iseg/preprocessor.h` + `.cpp`
- Modify: `csrc/vision/obb/preprocessor.h` + `.cpp`
- Modify: `csrc/vision/pose/preprocessor.h` + `.cpp`
- Modify: `csrc/vision/face/face_det/preprocessor.h` + `.cpp`

**Interfaces:**
- Consumes: Task 4 的 `create_processor_backend`、Task 1 的 `VisionProcessorBackend`
- Produces: 5 个 preprocessor 持有 `shared_ptr<VisionProcessorBackend> backend_`，Task 6 使用 `set_processor_backend`

- [ ] **Step 1: 改 detection preprocessor 头文件**

把 `csrc/vision/detection/preprocessor.h` 中的 `bool use_cuda_preproc_ = false;` 替换为：

```cpp
        void use_cuda_preproc() {
            backend_ = create_processor_backend(Device::GPU, Backend::ORT, 0);
        }

        void set_processor_backend(std::shared_ptr<VisionProcessorBackend> backend) {
            backend_ = std::move(backend);
        }

        [[nodiscard]] std::shared_ptr<VisionProcessorBackend> get_processor_backend() const {
            return backend_;
        }
```

protected 成员 `bool use_cuda_preproc_` 替换为：
```cpp
        std::shared_ptr<VisionProcessorBackend> backend_ =
            std::make_shared<CpuProcessorBackend>();
```

同时头文件顶部 `#include` 增加：
```cpp
#include "vision/processors/processor_factory.h"
#include "vision/processors/cpu/cpu_processor_backend.h"
```

- [ ] **Step 2: 改 detection preprocessor 实现**

把 `preprocess()` 里的 `if (use_cuda_preproc_) { #ifdef WITH_GPU return yolo_preprocess_cuda(...); #else ... #endif } return yolo_preprocess_cpu(...);` 替换为：

```cpp
        return backend_->yolo_preprocess(image, output, size_, padding_value_[0], letter_box_record);
```

把 `run(src_y, src_uv, ...)` 里的 CUDA/CPU 分支替换为：

```cpp
        return backend_->yolo_preprocess_nv12(src_y, src_uv, src_size,
                                              step_y, step_uv, output,
                                              size_, padding_value_[0], letter_box_record);
```

移除不再使用的 `#include "vision/common/processors/yolo_preproc.h"`、`#include "vision/common/processors/pad.h"` 和 `#ifdef WITH_GPU #include ".../yolo_preproc.cuh" #endif`（如不再被引用）。

- [ ] **Step 3: 同模式改 iseg/obb/pose/face_det 四个 preprocessor**

对每个 preprocessor（`csrc/vision/iseg/preprocessor.h/.cpp`、`csrc/vision/obb/preprocessor.h/.cpp`、`csrc/vision/pose/preprocessor.h/.cpp`、`csrc/vision/face/face_det/preprocessor.h/.cpp`）重复 Step 1-2：
- 头文件：`use_cuda_preproc_` → `backend_` + `set_processor_backend()` + `get_processor_backend()` + `use_cuda_preproc()`（内部调工厂）
- 实现：preprocess() 内 CPU/CUDA 分支 → `backend_->yolo_preprocess(...)`
- 注意各 preprocessor 的 namespace 不同（`detection`/`detection`(iseg)/`detection`(obb)/`detection`(pose)/`face`）

**注意：** iseg/obb/pose 的 preprocessor 若有 NV12 变体 `run(src_y, src_uv, ...)`，同样替换为 `backend_->yolo_preprocess_nv12(...)`。face_det 的 preprocessor 若签名不同（如输出是 KeyPoints），按其实际算子调用替换。

- [ ] **Step 4: 编译验证**

Run: `cmake --build E:\CLionProjects\ModelDeploy\build --config Release --target test_modeldeploy --parallel 2>&1`
Expected: 编译通过

- [ ] **Step 5: 跑回归测试**

Run: `cd E:\CLionProjects\ModelDeploy\build && .\bin\test_modeldeploy.exe [core],[vision_models] 2>&1`
Expected: All tests passed

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/detection/ csrc/vision/iseg/ csrc/vision/obb/ csrc/vision/pose/ csrc/vision/face/face_det/
git commit -m "refactor(preprocessor): replace use_cuda_preproc flag with VisionProcessorBackend delegation"
```

---

### Task 6: 模型 initialize() 自动注入 backend

**Files:**
- Modify: `csrc/vision/detection/ultralytics_det.cpp`（initialize()）
- Modify: `csrc/vision/iseg/ultralytics_seg.cpp`（initialize()）
- Modify: `csrc/vision/obb/ultralytics_obb.cpp`（initialize()）
- Modify: `csrc/vision/pose/ultralytics_pose.cpp`（initialize()）
- Modify: `csrc/vision/face/face_det/scrfd.cpp`（initialize()）

**Interfaces:**
- Consumes: Task 5 的 `preprocessor_.set_processor_backend()`、`runtime_option.device/backend/device_id`

- [ ] **Step 1: 在每个模型的 initialize() 里注入 backend**

对 5 个模型的 `initialize()`（例如 `csrc/vision/detection/ultralytics_det.cpp`）：

```cpp
    bool UltralyticsDet::initialize() {
        if (!init_runtime()) {
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }
```

头文件顶部确认 `#include "vision/processors/processor_factory.h"` 已引入（直接 include 或经 preprocessor 头传递）。

**注意：** 若某模型 `initialize()` 中已有其它逻辑（如 OCR 预处理器特殊配置），仅在此方法末尾追加 set_processor_backend 一行，不改变原有逻辑。

- [ ] **Step 2: 编译验证**

Run: `cmake --build E:\CLionProjects\ModelDeploy\build --config Release --target test_modeldeploy --parallel 2>&1`
Expected: 编译通过

- [ ] **Step 3: 跑全部相关测试**

Run: `cd E:\CLionProjects\ModelDeploy\build && .\bin\test_modeldeploy.exe [core],[vision_models] 2>&1`
Expected: All tests passed（41 个 test case）

- [ ] **Step 4: 跑 demo 验证推理结果**

Run: `cd E:\CLionProjects\ModelDeploy\build\bin && .\demo_detection_cxx.exe 2>&1`
Expected: 检测结果正常输出，与重构前一致

- [ ] **Step 5: Commit**

```bash
git add csrc/vision/detection/ csrc/vision/iseg/ csrc/vision/obb/ csrc/vision/pose/ csrc/vision/face/face_det/
git commit -m "feat(model): auto-inject processor backend from runtime option in initialize"
```

---

### Task 7: 全量回归 + 收尾

**Files:**
- Test: `tests/`（如有必要新增 processor 工厂单测）
- Modify: `docs/optimization_prd.md` 或 README（如需要记录新抽象）

- [ ] **Step 1: 全量构建**

Run: `cmake --build E:\CLionProjects\ModelDeploy\build --config Release --parallel 2>&1`
Expected: 全 target 编译通过

- [ ] **Step 2: 全量测试**

Run: `cd E:\CLionProjects\ModelDeploy\build && ctest -C Release --output-on-failure 2>&1`
Expected: 除既有的 7 个环境相关失败（加密测试数据/路径）外全绿；核心 + 视觉模型测试全过

- [ ] **Step 3: 用 demo_benchmark 对比性能**

Run: `cd E:\CLionProjects\ModelDeploy\build\bin && .\demo_benchmark.exe 2>&1`
Expected: FPS 与重构前基本一致（误差 <5%），确认无性能回退

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: verify processor backend abstraction regression-free"
```

## Self-Review 记录

- **Spec 覆盖**：P1 全部内容（抽象接口 / CPU / CUDA / 工厂 / preprocessor 改造 / 枚举与 RuntimeOption）已映射到 Task 1-6；OCR det_preprocessor 因 CUDA 路径本身是注释掉的，留到后续任务（spec P1 边界已注明"5 个 yolo 系 preprocessor"）。
- **占位符扫描**：无 TBD/TODO；所有代码块完整。
- **类型一致性**：`yolo_preprocess`/`yolo_preprocess_nv12`/`process_device_image` 签名在 Task 1-5 间一致；`set_processor_backend`/`get_processor_backend`/`backend_` 在 Task 5-6 间一致。
- **风险**：`nv12_to_bgr_cpu`、`yolo_preprocess_cuda` 等外部函数签名需在实现时核对（Task 2/3 已注明先读头文件）。
