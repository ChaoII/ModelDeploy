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
| TTS（Audio8） | `audio::tts::Audio8` | 见 [models-Audio8](../models.md#112-audio8audio8-tts-preview-06b) |
| TTS（Qwen3） | `audio::tts::Qwen3Tts` | 见 [models-Qwen3](../models.md#113-qwen3-ttsqwen3-tts-tokenizer-12hz-06b) |

> 各类模型完整 API 见 [models.md](../models.md)；后端/设备切换见 [RuntimeOption](../runtime_option.md) 与 [后端详解](../backends.md)。

### TTS 类签名

三个 TTS 模型均继承 `audio::tts::ITtsModel`，统一提供 `predict` / `predict_stream` / `get_sample_rate`（`predict_stream` 的 `cb` 签名 `bool(const float*, int, float progress)`，返回 `false` 中止；`chunk_frames == 0` 等价一次性合成）。

> `chunk_frames` 单位为**模型相关**（非"秒"）：Audio8=AR 码帧（每帧 ≈ 2048 采样）、Qwen3=vq 码帧（每帧 ≈ 1920 采样）、Kokoro=UTF-8 字符数。上层应传相对小的值以观察多次音频回调（如 24/12/120）。Qwen3 在 AR 阶段还会回调 `progress` 空块（`*cb==nullptr, n==0`），消费端应跳过音频处理、仅记录进度。

- `audio::tts::Kokoro(model_onnx, tokens, lexicons, voices_bin, jieba_dir, norm_dir, opt)` —— 24kHz，`predict(text, voice, speed, &audio)`。
- `audio::tts::Audio8` —— **无 voices 参数**；默认构造后 `Load(model_dir, opt)`（`model_dir` 指向 `audio8_preview` 根目录，`voice` 来自 `{model_dir}/voices/`），**44100Hz**，`predict(text, voice, speed, &audio)`。
- `audio::tts::Qwen3Tts(model_dir, opt)` —— 24kHz，`predict(text, voice_or_speaker, speed, &audio)`；`clone(text, ref_audio, ref_text, lang, &audio)`（返回 `false` 表示失败，非法 `lang` / 参考音频无法编码等）。

```cpp
modeldeploy::RuntimeOption option;
option.use_ort_backend(); option.use_cpu();

// Audio8：44.1kHz；voice 来自 {model_dir}/voices/
modeldeploy::audio::tts::Audio8 audio8;
audio8.Load("{MODELDEPLOY_TTS_MODELS_DIR}/audio8_preview", option);
std::vector<float> a8;
audio8.predict("你好，世界。", "demo", 1.0f, &a8);

// Qwen3：24kHz；预置说话人或声音克隆
modeldeploy::audio::tts::Qwen3Tts qwen3("{MODELDEPLOY_TTS_MODELS_DIR}/qwen3_tts_0.6b", option);
std::vector<float> q3;
qwen3.predict("你好，世界。", "Vivian", 1.0f, &q3);
std::vector<float> clone;
qwen3.clone("你好，这是声音克隆。", "ref.wav", "参考音频的文本", "auto", &clone);

// 三模型统一流式合成
audio8.predict_stream("你好，世界。", "demo", 1.0f, 480,
    [](const float* samples, int n, float progress) -> bool { return true; });
```

### TTS 使用 CUDA 加速

Audio8 / Qwen3 两种模型为 **GPU-only**：经 `RuntimeOption::set_device(Device::GPU, 0)` 启用 ORT CUDAExecutionProvider；设备非 GPU 或 GPU 不可用（provider 缺失 / 初始化失败）时 `Load`/`predict` **直接返回失败，不落回 CPU 慢路径**。Audio8 另在模型内用 IoBinding 让 AR 的 KV 常驻显存，消除逐帧整块主机-设备搬运（见 `[tts-rtf]` 用例）。Kokoro 不受影响，默认 CPU。

```cpp
modeldeploy::RuntimeOption option;
option.set_device(modeldeploy::Device::GPU, 0);
modeldeploy::audio::tts::Audio8 audio8;
audio8.Load("{MODELDEPLOY_TTS_MODELS_DIR}/audio8_preview", option);

// Qwen3-TTS 同理：构造 Qwen3Tts 时传入同一个 option 即可启用 CUDA EP
```

**实测速度（RTX 4060 Ti，44.1/24 kHz，2026-09）**：Audio8 GPU RTF≈0.9–1.3（较 CPU 约 2.2×）；Qwen3 GPU RTF≈1.2（空闲；重载机器最高约 2.7，较 CPU 约 6–9×）。二者均未实现实时合成（RTF<1）。RTF 随文本长度/GPU/机器负载波动，以 `test_modeldeploy.exe "[tts-rtf]"` 回归门禁为准（Audio8≤1.5、Qwen3≤3.0）。

依赖 GPU 版 onnxruntime（providers 动态库 `onnxruntime_providers_cuda.dll` 等需在可执行搜索路径）+ CUDA 运行库（Windows 下 `CUDA\v13.x\bin\x64` 在 PATH）。不可用时按上段 fail-closed。

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
