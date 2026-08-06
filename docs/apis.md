# ModelDeploy 多语言 API

ModelDeploy 提供 **C++ / Python / C / C# / Rust** 五种语言的绑定。核心逻辑全在 C++ SDK，其余语言是薄封装，行为一致。

## 1. C++（首选）

完整功能，支持全部模型与后端。

```cpp
#include "modeldeploy/vision.h"

modeldeploy::RuntimeOption option;
option.use_ort_backend();
option.use_cpu();

auto det = modeldeploy::vision::detection::UltralyticsDet("yolo11n.onnx", option);
auto img = modeldeploy::ImageData::imread("test.jpg");
std::vector<modeldeploy::vision::DetectionResult> result;
det.predict(img, &result);
```

编译链接：见 [快速开始](./quickstart.md#3-编写第一个检测程序)。

## 2. Python（pybind11）

### 2.1 安装

```bash
cd ModelDeploy
pip install build
python -m build
pip install dist/modeldeploy-*.whl
```

构建后生成 `.pyi` 存根：

```bash
pybind11-stubgen modeldeploy
```

### 2.2 使用

```python
import modeldeploy

option = modeldeploy.RuntimeOption()
option.use_ort_backend()
option.use_cpu()
option.use_sophgo_backend(0)   # 或 Sophgo

# 目标检测
model = modeldeploy.vision.detection.UltralyticsDet("yolo11n.onnx", option)
model.get_preprocessor().set_size([640, 640])
model.get_postprocessor().set_conf_threshold(0.25)

import cv2
img = cv2.imread("test.jpg")
results = model.predict(img)
for r in results:
    print(r.label_id, r.score, r.box)
```

### 2.3 已绑定模块

- **核心**：`RuntimeOption`、`Runtime`、`Tensor`、`BaseModel`、`Device`、`Backend`
- **视觉模型**：`UltralyticsDet/Seg/Obb/Pose`、`Classification`、`Scrfd`、`SeetaFace*`、`LprPipeline`、`PaddleOCR`、`PedestrianAttribute` 等
- **结果结构**：`DetectionResult`、`InstanceSegResult`、`OCRResult`、`KeyPointsResult` 等
- **可视化**：`vis_det`、`vis_iseg`、`vis_ocr` 等
- **音频**：`Kokoro`（TTS）

### 2.4 性能测试

```python
import time
results = None
for _ in range(loop_count):
    results = model.predict(image)
print(f"{loop_count / elapsed} FPS")
```

## 3. C API（`md_*` 前缀）

面向 C/C++ 嵌入式、其他语言 FFI 桥接。统一返回 `MDStatusCode`。

```c
#include "modeldeploy/md_model_capi.h"

// 创建模型
MDModel model = md_create_detection_model("yolo11n.onnx", md_create_default_runtime_option());

// 设置输入尺寸
md_set_detection_input_size(model, 640, 640);

// 推理
MDDetectionResults results;
md_detection_predict(model, img, &results);

// 读取结果
int n = md_get_detection_result_size(results);
// ...

// 释放
md_free_detection_results(results);
md_free_detection_model(model);
```

### 接口分组

| 模块 | 接口 |
|------|------|
| 检测 | `md_create_detection_model` / `md_detection_predict` |
| 分类 | `md_create_classification_model` / `md_classification_predict` |
| 分割 | `md_create_instance_seg_model` / `md_instance_seg_predict` |
| 姿态 | `md_create_keypoint_model` / `md_keypoint_predict` |
| 旋转框 | `md_create_obb_model` / `md_obb_predict` |
| 人脸 | `md_create_face_det/rec/age/gender/as_*_model` |
| 车牌 | `md_create_lpr_*_model` |
| OCR | `md_create_ocr_model` / `md_ocr_model_predict` |
| 行人属性 | `md_create_attr_model` / `md_attr_predict` |
| 图像 | `md_read_image` / `md_save_image` / `md_from_bgr24` 等 |
| 绘制 | `md_draw_rect` / `md_draw_polygon` / `md_draw_text` |

编译需 `BUILD_CAPI=ON`。

## 4. C#（.NET）

C# 绑定封装 C API，命名空间 `ModelDeploy`。解决方案见 `csharp/ModelDeploy.sln`。

```csharp
using ModelDeploy;

var option = new MDRuntimeOption();
option.UseOrtBackend();
option.UseCpu();

var model = new MDDetectionModel("yolo11n.onnx", option);
model.SetInputSize(640, 640);

var results = model.Predict(img);
foreach (var r in results) {
    Console.WriteLine($"{r.LabelId} {r.Score} {r.Box}");
}
```

主要项目：
- `ModelDeploy` — C# 绑定库
- `ModelDeployExample` — 示例
- `ModelDeployUnitTest` — 单元测试

## 5. Rust

Rust 绑定通过 FFI 封装 C API。`rust/modeldeploy/`：

```rust
use modeldeploy::runtime::RuntimeOption;

let mut option = RuntimeOption::new();
option.sophgo_backend(0);   // 或 ort_backend() 等

// 调用 C API 的模型接口
```

主要文件：
- `rust/modeldeploy/src/runtime.rs` — RuntimeOption 封装
- `rust/modeldeploy/src/ffi.rs` — FFI 声明（`MD_BACKEND_*` 常量等）

## 6. API 一致性

所有绑定共享同一套 C++ 语义：

| 概念 | C++ | Python | C |
|------|-----|--------|---|
| 运行时配置 | `RuntimeOption` | `RuntimeOption` | `MDRuntimeOption` |
| 检测模型 | `UltralyticsDet` | `UltralyticsDet` | `MDModel` |
| 检测结果 | `DetectionResult` | `DetectionResult` | `MDDetectionResults` |
| 推理 | `predict()` | `predict()` | `md_*_predict()` |

后端/设备/精度配置在所有语言中保持一致（见 [RuntimeOption 配置详解](./runtime_option.md)）。
