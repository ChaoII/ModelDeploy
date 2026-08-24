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
