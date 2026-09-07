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
import cv2
import modeldeploy as md

option = md.RuntimeOption()
option.use_ort_backend()
option.use_cpu()
option.set_cpu_thread_num(4)

# 设备（OPENCL/VULKAN 需显式 MNN 后端，否则 fail-closed）
option.use_mnn_backend()
option.set_device(md.Device.OPENCL, 0)   # 或 md.Device.VULKAN / md.Device.GPU / md.Device.TPU

# 其它后端（均无参数）
# option.use_ncnn_backend()  /  option.use_trt_backend()  /  option.use_sophgo_backend()
# GPU：option.use_gpu(0) 或 option.set_device(md.Device.GPU, 0)

# 目标检测
model = md.vision.UltralyticsDet("yolo11n.onnx", option)
model.get_preprocessor().set_size([640, 640])
model.get_postprocessor().set_conf_threshold(0.25)

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

### TTS（Kokoro）

```python
import modeldeploy

option = modeldeploy.RuntimeOption()
option.use_ort_backend()
option.use_cpu()

# Kokoro：24kHz，构造参数较多（onnx/tokens/lexicons/voices.bin/jieba/norm_dir）
kokoro = modeldeploy.audio.Kokoro("kokoro.onnx", "tokens.txt", ["lex_en.txt", "lex_zh.txt"],
                                  "voices.bin", "dict/", "text_normalization/", option)
wav = kokoro.predict("你好，世界。", "zf_001", 1.0)      # 返回 float32 列表

# 统一流式：predict_stream 返回 chunks 列表（每块 float32 数组）
chunks = kokoro.predict_stream("你好，世界。", "zf_001", 1.0, chunk_frames=120)
```

`chunk_frames == 0` 等价一次性合成（单块回调整段）；`predict_stream(text, voice, speed, chunk_frames)` 返回逐块 float32 数组，平铺可 `np.concatenate`。超长文本走 `predict_stream`（分块，`>120` 字符触发多块）。

## 3. 已绑定模块
- **核心**：`RuntimeOption`、`Runtime`、`Tensor`、`BaseModel`、`Device`、`Backend`
- **视觉模型**：`UltralyticsDet/Seg/Obb/Pose`、`Classification`、`Scrfd`、`SeetaFace*`、`LprPipeline`、`PaddleOCR`、`PedestrianAttribute` 等
- **结果结构**：`DetectionResult`、`InstanceSegResult`、`OCRResult`、`KeyPointsResult` 等
- **可视化**：`vis_det`、`vis_iseg`、`vis_ocr` 等
- **音频**：`Kokoro`（TTS，`predict_stream` 返回 chunks 列表）、`SenseVoice` 等

## 4. 性能测试

```python
import time
results = None
for _ in range(loop_count):
    results = model.predict(image)
print(f"{loop_count / elapsed} FPS")
```
