# C++ 绑定

首选，完整功能，支持全部模型与后端。核心逻辑全在 C++ SDK。编译链接见 [快速开始](../quickstart.md#3-编写第一个检测程序)。

## 1. 核心类型

| 类型 | 说明 |
|------|------|
| `RuntimeOption` | 运行时配置（后端/设备/精度/线程/动态 shape），见 [配置详解](../runtime_option.md) |
| `Tensor` | 张量（CPU/CUDA/TPU，支持外部内存零拷贝） |
| `ImageData` | 图像封装（统一平面存储，BGR HWC uint8 / NV12 / I420 等），见 [预处理](../preprocess.md) |
| `BaseModel` | 所有模型基类，提供 `predict` / `batch_predict` / `clone` |
| 结果结构 | `DetectionResult`、`SemSegResult`、`DepthResult`、`OCRResult`、`KeyPointsResult`、`FaceDetectionResult` 等 |

## 2. 检测示例

```cpp
#include "modeldeploy/vision.h"

modeldeploy::RuntimeOption option;
option.use_ort_backend(); option.use_cpu();

auto det = modeldeploy::vision::detection::UltralyticsDet("yolo11n.onnx", option);
det.get_preprocessor().set_size({640, 640});
det.get_postprocessor().set_conf_threshold(0.25f);

auto img = modeldeploy::ImageData::imread("test.jpg");
std::vector<modeldeploy::vision::DetectionResult> result;
det.predict(img, &result);
```

## 3. 更多模型（均使用同一 `RuntimeOption`）

| 能力 | 类 | 用法 |
|------|----|------|
| 实例分割 | `vision::detection::UltralyticsSeg` | 见 [models-分割](../models.md#2-实例分割instance-segmentation) |
| 语义分割 | `vision::detection::UltralyticsSem` | 见 [models-语义分割](../models.md#25-语义分割semantic-segmentation) |
| 深度估计 | `vision::detection::UltralyticsDepth` | 见 [models-深度](../models.md#26-深度估计depth-estimation) |
| 姿态估计 | `vision::detection::UltralyticsPose` | 见 [models-姿态](../models.md#3-姿态估计pose--keypoints) |
| 旋转框 | `vision::detection::UltralyticsObb` | 见 [models-旋转框](../models.md#4-旋转框检测oriented-bounding-box) |
| 分类 | `vision::Classification` | 见 [models-分类](../models.md#5-图像分类classification) |
| OCR | `vision::ocr::PaddleOCR` | 见 [models-OCR](../models.md#8-ocr文字识别) |
| 人脸 | `vision::face::Scrfd` / `InsightFaceAnalysis` | 见 [models-人脸](../models.md#6-人脸face) |
| 车牌 | `vision::lpr::LprPipeline` | 见 [models-车牌](../models.md#7-车牌识别license-plate) |
| ASR | `audio::asr::SenseVoice` | 见 [models-语音](../models.md#10-语音识别asr) |
| TTS（Kokoro） | `audio::tts::Kokoro` | 见 [models-TTS](../models.md#11-语音合成tts) |

> 各类模型完整 API 见 [models.md](../models.md)；后端/设备切换见 [RuntimeOption](../runtime_option.md) 与 [后端详解](../backends.md)。

### TTS 类签名

Kokoro 继承 `audio::tts::ITtsModel`，提供 `predict` / `predict_stream` / `get_sample_rate`（`predict_stream` 的 `cb` 签名 `bool(const float*, int, float progress)`，返回 `false` 中止；`chunk_frames == 0` 等价一次性合成）。

> `chunk_frames` 单位为 **UTF-8 字符数**（Kokoro，>120 字触发多块）。上层应传相对小的值以观察多次音频回调（如 120）。

- `audio::tts::Kokoro(model_onnx, tokens, lexicons, voices_bin, jieba_dir, norm_dir, opt)` —— 24kHz，`predict(text, voice, speed, &audio)`。

```cpp
modeldeploy::RuntimeOption option;
option.use_ort_backend(); option.use_cpu();

// Kokoro：24kHz；voice 来自 {model_dir}/voices/
modeldeploy::audio::tts::Kokoro kokoro("kokoro.onnx", "tokens.txt",
    {"lexicon-us-en.txt", "lexicon-zh.txt"}, "voices.bin", "dict/", "", option);
std::vector<float> wav;
kokoro.predict("你好，世界。", "zf_001", 1.0f, &wav);

// 统一流式合成
kokoro.predict_stream("你好，世界。", "zf_001", 1.0f, 120,
    [](const float* samples, int n, float progress) -> bool { return true; });
```

## 设备与设备帧

`RuntimeOption::set_device(Device::OPENCL/VULKAN)`(需显式 `use_mnn_backend()`,否则 fail-closed）:

```cpp
modeldeploy::RuntimeOption opt;
opt.use_mnn_backend();
opt.set_device(modeldeploy::Device::OPENCL, 0);   // == OK
opt.set_device(modeldeploy::Device::VULKAN, 0);   // == OK
```

设备帧 NV12：`ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, device)`(device 取 `Device::CPU/GPU/OPENCL/VULKAN/TPU`）——Python `ImageData.from_device_nv12(y, uv, w, h, dev=...)` 与 C/C#/Rust 均对齐此语义。

## 4. 工程配置

```cmake
CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(test C CXX)
set(CMAKE_CXX_STANDARD 17)
if (MSVC) add_compile_options(/utf-8) endif ()
include_directories("E:/.../install/include")
link_directories("E:/.../install/lib")
add_executable(test main.cpp)
target_link_libraries(test ModelDeploySDK)
```
