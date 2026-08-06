# ModelDeploy 模型详解

按功能点介绍 ModelDeploy 支持的全部模型：接口、输入输出、示例。所有模型类均继承 `BaseModel`，提供统一的 `predict` / `batch_predict` / `clone` 接口。

## 1. 目标检测（Detection）

### 1.1 UltralyticsDet（YOLO 系列）

通用目标检测，支持 YOLOv5/v8/v9/v11/v12 等 Ultralytics 系列。

```cpp
#include "modeldeploy/vision.h"

modeldeploy::RuntimeOption option;
option.use_ort_backend(); option.use_cpu();

auto det = modeldeploy::vision::detection::UltralyticsDet("yolo11n.onnx", option);
det.get_preprocessor().set_size({640, 640});
det.get_postprocessor().set_conf_threshold(0.25f);
det.get_postprocessor().set_nms_threshold(0.5f);

auto img = modeldeploy::ImageData::imread("test.jpg");
std::vector<modeldeploy::vision::DetectionResult> result;
det.predict(img, &result);
// result[i]: {box(x,y,w,h), label_id, score}
```

**关键接口**：

| 接口 | 说明 |
|------|------|
| `batch_predict(images, results)` | 批量推理 |
| `clone()` | 多线程克隆 |
| `predict_nv12(y, uv, w, h, ...)` | NV12 直通（GPU 场景） |
| `get_preprocessor().set_size({640,640})` | 输入尺寸 |
| `get_postprocessor().set_conf_threshold()` | 置信度阈值 |
| `get_label_map("names")` | 类别名文件 |

**示例**：`examples/demo_det/`
- `demo_detection_cxx.cpp` — 基础用法
- `demo_detection_batch.cpp` — 批量推理
- `demo_detection_multi_thread.cpp` — 多线程
- `demo_detection_sophgo.cpp` — Sophgo TPU

## 2. 实例分割（Instance Segmentation）

### 2.1 UltralyticsSeg

```cpp
auto seg = modeldeploy::vision::detection::UltralyticsSeg("yolo11n-seg.onnx", option);
seg.get_preprocessor().set_size({640, 640});

std::vector<modeldeploy::vision::InstanceSegResult> result;
seg.predict(img, &result);
// result[i]: {box, mask(二值图), label_id, score}
```

mask 以 `Mask` 结构保存（shape `{h, w}`，uint8 0/1），可用 `vis_iseg` 可视化。

**示例**：`examples/demo_iseg/demo_instance_seg_cxx.cpp`

## 3. 姿态估计（Pose / Keypoints）

### 3.1 UltralyticsPose

```cpp
auto pose = modeldeploy::vision::detection::UltralyticsPose("yolo11n-pose.onnx", option);
pose.get_preprocessor().set_size({640, 640});

std::vector<modeldeploy::vision::KeyPointsResult> result;
pose.predict(img, &result);
// result[i]: {box, keypoints[N][3](x,y,conf), score}
```

**示例**：`examples/demo_kps/demo_pose_cxx.cpp`

## 4. 旋转框检测（Oriented Bounding Box）

### 4.1 UltralyticsObb

```cpp
auto obb = modeldeploy::vision::detection::UltralyticsObb("yolo11n-obb.onnx", option);
obb.get_preprocessor().set_size({640, 640});

std::vector<modeldeploy::vision::ObbResult> result;
obb.predict(img, &result);
// result[i]: {rotated_box(四点), label_id, score}
```

**示例**：`examples/demo_obb/demo_obb_cxx.cpp`

## 5. 图像分类（Classification）

### 5.1 Classification

```cpp
auto cls = modeldeploy::vision::classification::Classification("yolo11n-cls.onnx", option);
cls.get_preprocessor().set_size({224, 224});

std::vector<modeldeploy::vision::ClassifyResult> result;
cls.predict(img, &result);
// result[i]: {label_ids, scores}
```

支持 `set_topk`、`set_multi_label`（多标签分类）。

**示例**：`examples/demo_cls/demo_classification_cxx.cpp`

## 6. 人脸（Face）

人脸模块包含完整人脸应用链：检测、识别、年龄、性别、防伪。

| 模型类 | 功能 | 输出 |
|--------|------|------|
| `Scrfd` | 人脸检测 | 框 + 5 关键点 |
| `SeetaFaceID` | 人脸特征提取 | 512 维 embedding |
| `SeetaFaceAge` | 年龄估计 | 年龄段 |
| `SeetaFaceGender` | 性别判定 | 男/女 |
| `SeetaFaceAsFirst` | 防伪一阶段 | 分类结果 |
| `SeetaFaceAsSecond` | 防伪二阶段 | 模糊/活体/翻拍概率 |
| `SeetaFaceAsPipeline` | 防伪流水线 | 一二阶段串联 |
| `FaceRecognizerPipeline` | 识别流水线 | 检测 + 特征一体化 |

```cpp
// 人脸检测
auto det = modeldeploy::vision::face::Scrfd("scrfd.onnx", option);
det.get_preprocessor().set_size({640, 640});

std::vector<modeldeploy::vision::FaceDetectionResult> result;
det.predict(img, &result);
// result[i]: {box, landmarks[5]}

// 人脸识别流水线（检测+特征）
auto rec = modeldeploy::vision::face::FaceRecognizerPipeline("det.onnx", "rec.onnx", option);
rec.set_det_threshold(0.5);
rec.cls_batch_size = 8;
```

**示例**：`examples/demo_face/`（demo_face_det / rec / age / gender / as_pipeline / rec_pipeline）

## 7. 车牌识别（License Plate）

| 模型类 | 功能 |
|--------|------|
| `LprDetection` | 车牌检测（框 + 关键点） |
| `LprRecognizer` | 车牌字符识别（字符串 + 颜色） |
| `LprPipeline` | 检测 + 识别串联 |

```cpp
auto lpr = modeldeploy::vision::lpr::LprPipeline("det.onnx", "rec.onnx", option);
std::vector<modeldeploy::vision::LprResult> result;
lpr.predict(img, &result);
// result[i]: {box, plate(字符串), color}
```

**示例**：`examples/demo_lpr/`

## 8. OCR（文字识别）

### 8.1 PaddleOCR（完整流水线）

```cpp
auto ocr = modeldeploy::vision::ocr::PaddleOCR(
    "det.onnx", "cls.onnx", "rec.onnx", option);
ocr.get_preprocessor().set_max_side_len(960);

std::vector<modeldeploy::vision::OCRResult> result;
ocr.predict(img, &result);
// result[i]: {text(识别文本), score, box(文本框四点), cls_label, cls_score}
```

### 8.2 单模块

| 模型类 | 功能 |
|--------|------|
| `DBDetector` | 文本检测（DB，输出多边形） |
| `Recognizer` | 文本识别（CTC） |
| `Classifier` | 方向分类（0°/180°） |

### 8.3 表格结构

| 模型类 | 功能 |
|--------|------|
| `StructureV2Layout` | 版面分析 |
| `StructureV2SERViLayoutXLMModel` | 语义实体识别 |
| `StructureV2Table` | 表格结构识别（SLANet） |
| `PPStructureV2Table` | 表格流水线 |

```cpp
auto table = modeldeploy::vision::ocr::PPStructureV2Table(
    "det.onnx", "rec.onnx", "table.onnx", option);
std::vector<modeldeploy::vision::OCRResult> result;
table.predict(img, &result);
```

**示例**：`examples/demo_ocr/`

## 9. 行人属性（Pedestrian Attribute）

### 9.1 PedestrianAttribute

检测行人 + 多标签属性分类串联。

```cpp
auto attr = modeldeploy::vision::PedestrianAttribute("det.onnx", "cls.onnx", option);
attr.set_det_threshold(0.5);
attr.set_det_input_size({1280, 1280});
attr.set_cls_input_size({192, 256});
attr.cls_batch_size = 8;

std::vector<modeldeploy::vision::AttributeResult> result;
attr.predict(img, &result);
```

**示例**：`examples/demo_pipeline/demo_pedestrian_attribute_cxx.cpp`

## 10. 语音识别（ASR）

### 10.1 SenseVoice（流式识别）

```cpp
auto asr = modeldeploy::audio::asr::SenseVoice("sense_voice.onnx", option);
std::string text = asr.predict(wav_data);
```

### 10.2 AAsr（VAD + 识别流水线）

```cpp
auto asr = modeldeploy::audio::AAsr("vad.onnx", "sense_voice.onnx", option);
asr.predict(pcm_data, [](const std::string& text){ /* 回调 */ });
```

**示例**：`examples/demo_audio/demo_sense_voice_cxx.cpp`

## 11. 语音合成（TTS）

### 11.1 Kokoro

```cpp
auto tts = modeldeploy::audio::tts::Kokoro("kokoro.onnx", option);
auto audio = tts.predict("你好，世界");
audio.save_wav("out.wav");
```

支持中英混读（jieba 分词 + 文本正则化）。

**示例**：`examples/demo_audio/demo_kokoro_cxx.cpp`

## 12. VAD（语音活动检测）

### 12.1 SileroVAD

```cpp
auto vad = modeldeploy::audio::vad::SileroVAD("silero_vad.onnx", option);
std::vector<std::pair<int,int>> segments = vad.predict(pcm_data, sample_rate);
```

支持 16k/8k 采样率，32/64/96ms 窗口。

## 13. 模型与测试数据

- **测试模型**：`test_data/test_models/`（yolo11n 系列、人脸、OCR 等）
- **测试图片**：`test_data/test_images/`
- **测试数据下载**：见 AGENTS.md（需从 modelscope 单独下载）

## 14. 模型选择与转换

| 场景 | 推荐 |
|------|------|
| 通用检测 | YOLO11n/s/m（`.onnx`） |
| 分割 | YOLO11n-seg |
| 姿态 | YOLO11n-pose |
| 旋转框 | YOLO11n-obb |
| GPU 加速 | 转 `.engine`（TRT） |
| TPU 部署 | 转 `.bmodel`（Sophgo） |

各模型导出/转换教程见 [README](../../README.md) 与 [后端详解](./backends.md)。
