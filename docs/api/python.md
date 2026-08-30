# Python 绑定（pybind11）

## 1. 安装

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

## 2. 使用

```python
import modeldeploy

option = modeldeploy.RuntimeOption()
option.use_ort_backend()
option.use_cpu()

# 设备（OPENCL/VULKAN 需显式 MNN 后端，否则 fail-closed）
option.use_mnn_backend()
option.set_device(modeldeploy.Device.OPENCL, 0)
option.set_device(modeldeploy.Device.VULKAN, 0)
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

### 设备帧 NV12（`ImageData.from_device_nv12`）

零拷贝借用 host/device NV12 帧，设备语义经 `dev` 参数与 C/C/C#/Rust 对齐（缺省 CPU）：

```python
import numpy as np
y  = np.zeros(h * step_y, dtype=np.uint8)
uv = np.zeros((h // 2) * step_uv, dtype=np.uint8)
# host NV12（CPU 缺省）
img_cpu = modeldeploy.ImageData.from_device_nv12(y, uv, w, h, dev=modeldeploy.Device.CPU)
# 设备 NV12（OPENCL/VULKAN 等，需显式 use_mnn_backend()）
img_dev = modeldeploy.ImageData.from_device_nv12(y, uv, w, h, dev=modeldeploy.Device.OPENCL)
```
y/uv 指向调用方内存、不拷贝，调用方须保证缓冲在 `predict` 期间存活。

## 3. 已绑定模块

- **核心**：`RuntimeOption`、`Runtime`、`Tensor`、`BaseModel`、`Device`、`Backend`
- **视觉模型**：`UltralyticsDet/Seg/Obb/Pose`、`Classification`、`Scrfd`、`SeetaFace*`、`LprPipeline`、`PaddleOCR`、`PedestrianAttribute` 等
- **结果结构**：`DetectionResult`、`InstanceSegResult`、`OCRResult`、`KeyPointsResult` 等
- **可视化**：`vis_det`、`vis_iseg`、`vis_ocr` 等
- **音频**：`Kokoro`（TTS）

## 4. 性能测试

```python
import time
results = None
for _ in range(loop_count):
    results = model.predict(image)
print(f"{loop_count / elapsed} FPS")
```
