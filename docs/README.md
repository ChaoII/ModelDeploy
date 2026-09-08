# ModelDeploy 文档中心

ModelDeploy 是一个面向工业落地的多后端推理 SDK，支持 **目标检测 / 实例分割 / 姿态估计 / 旋转框 / 分类 / OCR / 人脸 / 车牌 / 行人属性 / 语音识别 / 语音合成 / VAD** 等模型，一套代码统一调用 **OnnxRuntime / TensorRT / MNN / ncnn / Sophgo(算能 TPU)** 五种后端，并提供 **C++ / Python / C / C# / Rust** 多语言绑定。

## 文档导航

### 快速上手

| 文档 | 内容 |
|------|------|
| [快速开始](./quickstart.md) | 从源码构建、第一个检测程序、完整示例 |
| [README](../README.md) | 项目概述、快速开始、支持矩阵 |

### 核心概念

| 文档 | 内容 |
|------|------|
| [架构设计](./architecture.md) | 目录结构、`BaseModel` / `RuntimeOption` / `Tensor` / `ImageData` 核心抽象、推理链路 |
| [RuntimeOption 配置详解](./runtime_option.md) | 全部后端选择、设备选择、精度、线程、动态 shape 等配置项 |
| [模型转换/量化](./conversion.md) | 混合精度、动态量化、TRT engine、Sophgo bmodel 转换 |
| [预处理详解](./preprocess.md) | `ImageData` 图像类型、预处理算子、CPU/CUDA/BMCV 硬件加速、零拷贝链路 |

### 后端

| 文档 | 内容 |
|------|------|
| [后端详解](./backends.md) | OnnxRuntime / TensorRT / MNN / ncnn / Sophgo 五种后端对比、模型格式、构建要求 |

### 模型

| 文档 | 内容 |
|------|------|
| [模型详解](./models.md) | 按功能点的全部模型（检测/分割/姿态/人脸/OCR/车牌/音频…）接口与用法 |
| [解决方案](./solutions.md) | 检测+跟踪组合成的业务方案（计数/热力/测速/车位/距离/打码/裁剪/健身/说话人/STT/TTS） |
| [工具](./tools.md) | 后处理公共底座（标注/检测容器/指标/切片/平滑/区域判断） |

### 视频编解码

| 文档 | 内容 |
|------|------|
| [视频编解码总览](./video/README.md) | 模块能力、支持矩阵、快速开始 |
| [使用教程](./video/guide.md) | 小白向：构建 + 解码/编码程序（C++/Python） |
| [技术报告](./video/design.md) | 架构分层、抽象方案、后端矩阵、背压/缓冲池/重连原理、性能 |
| [接口参考](./video/api.md) | 逐签名接口、参数、生命周期与 GPU 借用约定 |

### 进阶

| 文档 | 内容 |
|------|------|
| [性能优化指南](./performance.md) | 推理提速、多线程与 clone、零拷贝、后端选型、实测数据 |
| [海量视频流效率优化](./pipeline-efficiency.md) | CUDA PC / Jetson / Sophgo 跨平台视频管线实测与瓶颈分析 |
| [多语言 API](./api/README.md) | C++ / Python / C / C# / Rust 绑定概览 |
| [模型加密](./encryption.md) | AES-256-GCM 模型加密与解密模型使用 |
| [多线程推理](./multi_thread.md) | `clone()` 多线程并发详解 |
| [异步推理](./async_inference.md) | `AsyncModel` 异步投递（有界背压队列 / 攒批 / future+回调）+ 解码联动 `AsyncVideoInfer` |
| [嵌入式推理网关](./serving.md) | `ServingServer` REST 推理服务（健康/鉴权/限流/CORS/metrics/TLS(可选)/优雅停机）、模型仓库热更新；gRPC 二期 |
| [Triton 推理服务](../examples/serving/) | Triton 部署（preprocess → pipeline → postprocess） |

### 应用案例

| 文档 | 内容 |
|------|------|
| [AI 智能安防监控平台](./surveillance.md) | `application/surveillance` 跨平台监控应用架构（CUDA/Jetson/Sophgo） |
| [Web 模型演示](./web_demo.md) | `demo_server` 各模型族浏览器演示（ServingServer + 静态页 + 每族 E2E 验证） |

### 发布与开发

| 文档 | 内容 |
|------|------|
| [发布流程](./release.md) | 版本号 bump、打 tag、GitHub Release/资产上传规范 |

## 支持矩阵

### 后端 × 设备

| 后端 | 模型格式 | CPU | CUDA GPU | OpenCL | Vulkan | TPU |
|------|---------|-----|----------|--------|--------|-----|
| OnnxRuntime | `.onnx` | ✅ | ✅ | ✅ | — | — |
| TensorRT | `.engine` / `.onnx` | — | ✅ | — | — | — |
| MNN | `.mnn` | ✅ | ✅ | ✅ | ✅ | — |
| ncnn | `.param` / `.bin` | ✅ | — | — | ✅ | — |
| Sophgo | `.bmodel` | — | — | — | — | ✅ (BM1688/CV186X) |

> **模型覆盖**：上表"后端支持=全部"指 ORT / TRT / MNN / Sophgo 四后端全覆盖；**ncnn 后端当前仅覆盖 ultralytics YOLO 全系**（det/cls/obb/pose/seg/sem/depth），其余模型请用 ORT / MNN / TRT / Sophgo。

### 模型能力

| 功能 | 模型类 | 后端支持 |
|------|--------|---------|
| 目标检测 | `UltralyticsDet` | 全部 |
| 实例分割 | `UltralyticsSeg` | 全部 |
| **语义分割** | `UltralyticsSem` | 全部 |
| **深度估计** | `UltralyticsDepth` | 全部 |
| 姿态估计 | `UltralyticsPose` | 全部 |
| 旋转框检测 | `UltralyticsObb` | 全部 |
| 图像分类 | `Classification` | 全部 |
| 人脸检测 | `Scrfd` / `InsightFaceDet` | 全部 |
| 人脸识别 | `SeetaFaceID` / `FaceRecognizerPipeline` / `InsightFaceRecognition` | 全部 |
| 人脸分析 | `InsightFaceAnalysis` | 全部 |
| 人脸年龄/性别 | `SeetaFaceAge` / `SeetaFaceGender` | 全部 |
| 人脸防伪 | `SeetaFaceAsPipeline` | 全部 |
| 车牌识别 | `LprPipeline` | 全部 |
| 文字识别 OCR | `PaddleOCR` / `PPStructureV2Table` | 全部 |
| **公式识别** | `FormulaRecognizer` | 全部 |
| 文档理解(→Markdown) | `StructureV2Layout` + OCR + 表格 | 全部 |
| 行人属性 | `PedestrianAttribute` | 全部 |
| **多目标跟踪** | `ByteTracker` / `BotSortTracker` / `StrongSortTracker` | 全部 |
| **视频动作识别** | `TSN` / `StGcn` | 全部 |
| **行人 Re-ID** | `ReID` (OSNet) | 全部 |
| **声纹验证** | `SpeakerVerify` (ECAPA) | 全部 |
| **条码/二维码** | `BarcodeDetector` | 全部 |
| **手部关键点** | `HandKeypoint` | 全部 |
| **CV 解决方案** | `ObjectCounter` / `Heatmap` / `SpeedEstimator` / `ParkingManager` 等 | 全部 |
| **NLP** | `TextClassifier` / 分词 / 分句 / 关键词 | 全部 |
| 语音识别 | `SenseVoice` / `AAsr` / `ParaformerStreamingAsr` | 全部 |
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

更多示例见 [examples](../examples/)，构建与模型转换见 [README](../README.md)。
