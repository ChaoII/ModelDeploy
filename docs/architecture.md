# ModelDeploy 架构设计

## 1. 总体架构

ModelDeploy 采用 **统一推理抽象 + 多后端适配 + 模型无关预处理/后处理** 的分层架构：

```
┌─────────────────────────────────────────────────────────┐
│                    多语言绑定层                          │
│   Python (pybind11) │ C API │ C# │ Rust                 │
└──────────────────────────┬──────────────────────────────┘
┌──────────────────────────▼──────────────────────────────┐
│                    模型层 (BaseModel)                    │
│  UltralyticsDet/Seg/Pose/Obb │ Scrfd │ PaddleOCR │ ...  │
│  统一接口: predict / batch_predict / clone               │
│  ├─ Preprocessor  (输入尺寸/letterbox/归一化)            │
│  └─ Postprocessor (阈值/NMS/坐标还原/结果结构)           │
└──────────────────────────┬──────────────────────────────┘
┌──────────────────────────▼──────────────────────────────┐
│                   Runtime / Backend 层                   │
│  RuntimeOption → 选后端 + 设备 + 精度                    │
│  ┌────────┬─────────┬─────────┬─────────┐               │
│  │ ORT    │  TRT    │  MNN    │ Sophgo  │               │
│  │ .onnx  │ .engine │ .mnn    │ .bmodel │               │
│  └────────┴─────────┴─────────┴─────────┘               │
└──────────────────────────┬──────────────────────────────┘
┌──────────────────────────▼──────────────────────────────┐
│                   核心类型层                             │
│  Tensor (多维张量/设备内存) │ ImageData (图像) │ 结果结构 │
└─────────────────────────────────────────────────────────┘
```

## 2. 目录结构

```
ModelDeploy/
├── csrc/                    # C++ SDK 源码
│   ├── core/                # Tensor / 日志 / 枚举类型
│   ├── runtime/             # RuntimeOption + 四种后端
│   ├── vision/              # 视觉模型（检测/分割/姿态/人脸/OCR/车牌...）
│   ├── audio/               # 音频模型（ASR/TTS/VAD/文本正则化）
│   ├── base_model.h         # BaseModel 抽象基类
│   ├── vision.h             # 视觉公开头文件聚合
│   └── audio.h              # 音频公开头文件聚合
├── capi/                    # C API (md_* 前缀)
├── python/                  # Python 包
├── csharp/                  # C# 绑定
├── rust/                    # Rust 绑定
├── examples/                # 各功能 demo
├── cmake/                   # 第三方依赖查找模块
├── tools/                   # 模型加密 / Sophgo 转换脚本
└── tests/                   # Catch2 测试
```

## 3. 核心抽象

### 3.1 BaseModel（模型基类）

所有模型类继承自 `BaseModel`，统一提供：

| 接口 | 说明 |
|------|------|
| `predict(image, result)` | 单图推理 |
| `batch_predict(images, results)` | 批量推理 |
| `clone()` | 克隆实例（多线程用，共享 Session） |
| `get_preprocessor()` | 获取预处理器（设置输入尺寸/阈值等） |
| `get_postprocessor()` | 获取后处理器（置信度/NMS 阈值等） |
| `is_initialized()` | 是否初始化成功 |

模型层持有 `Runtime`（即 `Backend`），推理时按 `Preprocessor → Backend.infer → Postprocessor` 顺序执行。

### 3.2 RuntimeOption（运行时配置）

配置后端、设备、精度、线程、模型路径等。见 [RuntimeOption 配置详解](./runtime_option.md)。

### 3.3 Tensor（张量）

支持多设备内存的多维张量：

```
Tensor
├─ shape / dtype / device / name     # 元数据
├─ allocate(shape, dtype, device)    # CPU malloc / cudaMalloc / bmrt 设备内存
├─ from_external_memory(data, ...)   # 绑定外部内存（零拷贝，不拥有）
├─ data() / data_ptr<T>()            # 数据访问
├─ transpose/reshape/slice           # 惰性视图 (TensorView)
└─ device() == CPU/GPU/TPU           # 设备感知（零拷贝推理的关键）
```

### 3.4 ImageData（图像）

封装 OpenCV Mat，提供完整预处理算子链。见 [预处理详解](./preprocess.md)。

## 4. 推理链路

### 4.1 常规链路（CPU 输入）

```
ImageData → Preprocessor(letterbox+归一化) → CPU Tensor
    → Backend.infer(inputs, outputs)
        → ORT: BindInput+Run+GetOutput
        → TRT: enqueueV2
        → MNN: forward
        → Sophgo: bmrt_launch_tensor
    → Postprocessor(阈值+NMS+坐标还原) → 结果结构
```

### 4.2 零拷贝链路（设备端预处理）

```
ImageData → CudaProcessorBackend / SophgoProcessorBackend
    → GPU/TPU 设备内存上完成 letterbox+归一化
    → 产出 Device::GPU / Device::TPU 的 Tensor
    → Backend.infer 识别设备输入，跳过 H2D 拷贝直接推理
    → (Sophgo SOC) 输出 mmap 零拷贝读回
```

## 5. 后端设计模式

所有后端继承 `BaseBackend`，实现统一接口：

```cpp
class BaseBackend {
public:
    virtual bool init(const RuntimeOption&) = 0;      // 加载模型
    virtual bool infer(std::vector<Tensor>& inputs,
                       std::vector<Tensor>* outputs) = 0;  // 推理
    virtual std::unique_ptr<BaseBackend> clone(
        const RuntimeOption&, void* stream, int device_id) = 0;
};
```

`Runtime::infer()` 转发到具体 backend 的 `infer()`，因此**上层模型对后端无感知**，切换后端只需改 `RuntimeOption`。

## 6. 多线程架构

模型实例**线程不安全**（ORT 的 IoBinding、TRT 的 execution context 是独占的）。正确并发方式是 **clone**：

- 每个线程 `model.clone()` 获取独立实例
- 克隆体**共享 Session**（轻量），各自持有独立执行上下文
- 见 [多线程推理指南](./multi_thread.md)

## 7. 依赖与编译

- C++17 标准
- 核心依赖：OpenCV（内置静态库）、各后端推理库（onnxruntime / TensorRT / MNN / libsophon）
- 无外部运行时依赖（SDK 自带 OpenCV、pybind11、Catch2 等）

构建命令见 [README](../../README.md) 或 [快速开始](./quickstart.md)。
