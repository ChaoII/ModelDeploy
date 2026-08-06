# ModelDeploy 文档中心

ModelDeploy 是一个面向工业落地的多后端推理 SDK，支持 **目标检测 / 实例分割 / 姿态估计 / 旋转框 / 分类 / OCR / 人脸 / 车牌 / 行人属性 / 语音识别 / 语音合成 / VAD** 等模型，一套代码统一调用 **OnnxRuntime / TensorRT / MNN / Sophgo(算能 TPU)** 四种后端，并提供 **C++ / Python / C / C# / Rust** 多语言绑定。

## 文档导航

### 快速上手

| 文档 | 内容 |
|------|------|
| [快速开始](./quickstart.md) | 从源码构建、第一个检测程序、完整示例 |
| [README](../../README.md) | 构建命令、模型加密、混合精度、TRT/Sophgo 模型转换 |

### 核心概念

| 文档 | 内容 |
|------|------|
| [架构设计](./architecture.md) | 目录结构、`BaseModel` / `RuntimeOption` / `Tensor` / `ImageData` 核心抽象、推理链路 |
| [RuntimeOption 配置详解](./runtime_option.md) | 全部后端选择、设备选择、精度、线程、动态 shape 等配置项 |
| [预处理详解](./preprocess.md) | `ImageData` 图像类型、预处理算子、CPU/CUDA/BMCV 硬件加速、零拷贝链路 |

### 后端

| 文档 | 内容 |
|------|------|
| [后端详解](./backends.md) | OnnxRuntime / TensorRT / MNN / Sophgo 四种后端对比、模型格式、构建要求 |

### 模型

| 文档 | 内容 |
|------|------|
| [模型详解](./models.md) | 按功能点的全部模型（检测/分割/姿态/人脸/OCR/车牌/音频…）接口与用法 |

### 进阶

| 文档 | 内容 |
|------|------|
| [性能优化指南](./performance.md) | 推理提速、多线程与 clone、零拷贝、后端选型、实测数据 |
| [多语言 API](./apis.md) | C++ / Python / C / C# / Rust 绑定概览 |
| [模型加密](./encryption.md) | AES-256-CBC 模型加密与解密模型使用 |
| [多线程推理](../../docs/multi_thread.md) | `clone()` 多线程并发详解 |
| [Triton 推理服务](../examples/serving/) | Triton 部署（preprocess → pipeline → postprocess） |

## 支持矩阵

### 后端 × 设备

| 后端 | 模型格式 | CPU | CUDA GPU | OpenCL | TPU |
|------|---------|-----|----------|--------|-----|
| OnnxRuntime | `.onnx` | ✅ | ✅ | ✅ | — |
| TensorRT | `.engine` / `.onnx` | — | ✅ | — | — |
| MNN | `.mnn` | ✅ | ✅ | ✅ | — |
| Sophgo | `.bmodel` | — | — | — | ✅ (BM1688/CV186X) |

### 模型能力

| 功能 | 模型类 | 后端支持 |
|------|--------|---------|
| 目标检测 | `UltralyticsDet` | 全部 |
| 实例分割 | `UltralyticsSeg` | 全部 |
| 姿态估计 | `UltralyticsPose` | 全部 |
| 旋转框检测 | `UltralyticsObb` | 全部 |
| 图像分类 | `Classification` | 全部 |
| 人脸检测 | `Scrfd` | 全部 |
| 人脸识别 | `SeetaFaceID` / `FaceRecognizerPipeline` | 全部 |
| 人脸年龄/性别 | `SeetaFaceAge` / `SeetaFaceGender` | 全部 |
| 人脸防伪 | `SeetaFaceAsPipeline` | 全部 |
| 车牌识别 | `LprPipeline` | 全部 |
| 文字识别 OCR | `PaddleOCR` / `PPStructureV2Table` | 全部 |
| 行人属性 | `PedestrianAttribute` | 全部 |
| 语音识别 | `SenseVoice` / `AAsr` | 全部 |
| 语音合成 | `Kokoro` | 全部 |
| VAD | `SileroVAD` | 全部 |

## 快速入口

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

更多示例见 [examples](../../examples/)，构建与模型转换见 [README](../../README.md)。
