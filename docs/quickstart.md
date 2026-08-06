# ModelDeploy 快速开始

本文带你从零开始：构建 SDK → 编写第一个检测程序 → 运行。

## 1. 环境要求

| 项 | 要求 |
|----|------|
| 操作系统 | Windows 10/11 / Linux (x86_64, aarch64) |
| 编译器 | MSVC 2022 / GCC ≥ 9，需完整 C++17 支持 |
| CMake | ≥ 3.16 |
| 架构 | 64 位（不支持 32 位） |

## 2. 构建 SDK

```bash
# 克隆
git clone https://github.com/ChaoII/ModelDeploy.git
cd ModelDeploy

# Windows: 在 "x64 Native Tools Command Prompt for VS 2022" 中执行
# 纯 CPU 构建（推荐 Ninja）
cmake -S . -B build -G Ninja ^
    -DBUILD_AUDIO=ON -DBUILD_VISION=ON ^
    -DBUILD_CAPI=OFF -DBUILD_PYTHON=OFF ^
    -DENABLE_MNN=OFF -DENABLE_ORT=ON -DENABLE_TRT=OFF ^
    -DWITH_GPU=OFF -DCMAKE_INSTALL_PREFIX=install

# Linux
cmake -S . -B build -G Ninja \
    -DBUILD_AUDIO=ON -DBUILD_VISION=ON \
    -DBUILD_CAPI=OFF -DBUILD_PYTHON=OFF \
    -DENABLE_MNN=OFF -DENABLE_ORT=ON -DENABLE_TRT=OFF \
    -DWITH_GPU=OFF -DCMAKE_INSTALL_PREFIX=install

# 编译 + 安装
cmake --build build --config Release --parallel
cmake --install build
```

安装后生成 `install/` 目录：`include/`（头文件）+ `lib/`（动态库）。

### 常用构建选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `BUILD_VISION` | ON | 视觉模块 |
| `BUILD_AUDIO` | ON | 音频模块 |
| `BUILD_CAPI` | ON | C API |
| `BUILD_PYTHON` | ON | Python 绑定 |
| `ENABLE_ORT` | ON | OnnxRuntime 后端 |
| `ENABLE_MNN` | ON | MNN 后端 |
| `ENABLE_TRT` | OFF | TensorRT 后端（需 `WITH_GPU=ON`） |
| `WITH_GPU` | ON | CUDA 支持 |
| `ENABLE_SOPHGO` | OFF | Sophgo TPU 后端 |
| `BUILD_TESTS` | OFF | Catch2 测试 |
| `BUILD_ENCRYPTION` | ON | 模型加密（需 OpenSSL） |

> **MSVC 注意**：根 `CMakeLists.txt` 已自动为 SDK 设置 `/utf-8` 编译选项，无需手动加。

## 3. 编写第一个检测程序

### 3.1 创建工程

`CMakeLists.txt`：

```cmake
CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(test_modeldeploy C CXX)
set(CMAKE_CXX_STANDARD 17)

if (MSVC)
    add_compile_options(/utf-8)
endif ()

set(MD_DIR "E:/CLionProjects/ModelDeploy/build/install")
set(MD_INC_DIR "${MD_DIR}/include")
set(MD_LIB_DIR "${MD_DIR}/lib")
include_directories(${MD_INC_DIR})
link_directories(${MD_LIB_DIR})

add_executable(test_modeldeploy ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(test_modeldeploy ModelDeploySDK)
```

### 3.2 检测代码

`main.cpp`：

```cpp
#include "modeldeploy/vision.h"
#include <vector>

int main() {
    // 1. 配置运行时：ORT 后端 + CPU
    modeldeploy::RuntimeOption option;
    option.use_ort_backend();
    option.use_cpu();
    option.set_cpu_thread_num(4);

    // 2. 加载检测模型
    modeldeploy::vision::detection::UltralyticsDet yolo11_det(
        "yolo11n.onnx", option);
    if (!yolo11_det.is_initialized()) return -1;

    // 3. 设置输入尺寸与阈值
    yolo11_det.get_preprocessor().set_size({640, 640});
    yolo11_det.get_postprocessor().set_conf_threshold(0.25f);

    // 4. 读图 + 推理
    auto img = modeldeploy::ImageData::imread("test_person.jpg");
    std::vector<modeldeploy::vision::DetectionResult> result;
    yolo11_det.predict(img, &result);

    // 5. 打印结果
    for (const auto& r : result) {
        printf("label=%d score=%.4f box=[%.0f %.0f %.0f %.0f]\n",
               r.label_id, r.score, r.box.x, r.box.y, r.box.width, r.box.height);
    }

    // 6. 可视化
    const auto label_map = yolo11_det.get_label_map("names");
    auto vis = modeldeploy::vision::vis_det(img, result, 0.25, label_map, "", 12, 0.3, false);
    vis.imwrite("vis_result.jpg");
    return 0;
}
```

### 3.3 性能测试

```cpp
// 预热
std::vector<modeldeploy::vision::DetectionResult> result;
for (int i = 0; i < 10; ++i) yolo11_det.predict(img, &result);

// 计时 100 帧
modeldeploy::TimerArray timers;
for (int i = 0; i < 100; ++i) yolo11_det.predict(img, &result, &timers);
timers.print_benchmark();
// 输出形如：
// [Preprocess ]: avg = 1.34 ms
// [Inference  ]: avg = 6.69 ms
// [Postprocess]: avg = 7.35 ms
// [Total      ]: avg = 15.27 ms
```

## 4. 切换后端

同一套代码切换后端只需改 `RuntimeOption`：

```cpp
// CPU
option.use_ort_backend(); option.use_cpu();

// CUDA + TRT
option.use_ort_backend(); option.use_gpu(0);
option.enable_trt = true; option.enable_fp16 = true;

// 纯 TensorRT engine
option.use_trt_backend(); option.use_gpu(0);
option.enable_fp16 = true;

// MNN
option.use_mnn_backend();

// Sophgo TPU
option.use_sophgo_backend(0);
option.sophgo_option.bmodel_path = "model.bmodel";
```

详见 [后端详解](./backends.md)。

## 5. Python 快速上手

```python
import modeldeploy

option = modeldeploy.RuntimeOption()
option.use_ort_backend()
option.use_cpu()

model = modeldeploy.vision.detection.UltralyticsDet("yolo11n.onnx", option)
model.get_preprocessor().set_size([640, 640])
model.get_postprocessor().set_conf_threshold(0.25)

import cv2
img = cv2.imread("test_person.jpg")
results = model.predict(img)
print(results)
```

## 6. 更多示例

| 功能 | 示例 |
|------|------|
| 目标检测 | `examples/demo_det/demo_detection_cxx.cpp` |
| 批量推理 | `examples/demo_det/demo_detection_batch.cpp` |
| 多线程 | `examples/demo_det/demo_multi_thread_compare.cpp` |
| 实例分割 | `examples/demo_iseg/demo_instance_seg_cxx.cpp` |
| 人脸/OCR/车牌 | `examples/demo_face` / `demo_ocr` / `demo_lpr` |
| 模型加密工具 | `examples/tools/model_encrypted.cpp` |

完整示例列表见 [README](../../README.md) 或 `examples/` 目录。

## 7. 常见问题

- **MSVC 报编码错误**：确保用 `x64 Native Tools Command Prompt`，根 CMake 已加 `/utf-8`。
- **32 位系统**：不支持，SDK 全部基于 64 位。
- **找不到 OpenCV**：SDK 内置静态 OpenCV，无需额外安装；若系统 OpenCV 版本过旧，用 `-DOpenCV_DIR` 指定。
- **GPU 推理不生效**：确认 `use_gpu()` + 对应后端（ORT 需 `enable_trt` 或用 TRT backend），并检查 GPU 驱动/CUDA。
