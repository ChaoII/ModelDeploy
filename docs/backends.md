# ModelDeploy 后端详解

ModelDeploy 支持四种推理后端，一套上层 API 统一调用。本文详细对比各后端的使用方式、模型格式、配置项与适用场景。

## 1. 后端总览

| 后端 | 类名 | 模型格式 | 设备 | 编译开关 | 适用场景 |
|------|------|---------|------|---------|---------|
| OnnxRuntime | `OrtBackend` | `.onnx` | CPU / CUDA / OpenCL | `ENABLE_ORT` | 跨平台通用，模型生态最全 |
| TensorRT | `TrtBackend` | `.engine` / `.onnx` | NVIDIA GPU | `ENABLE_TRT + WITH_GPU` | 英伟达 GPU 最高性能 |
| MNN | `MnnBackend` | `.mnn` | CPU / GPU(多种) / Metal | `ENABLE_MNN` | 移动端 / 边缘设备 |
| Sophgo | `SophgoBackend` | `.bmodel` | 算能 TPU (BM1688/CV186X) | `ENABLE_SOPHGO` | 国产化 / 低功耗边缘 |

## 2. OnnxRuntime 后端

### 2.1 基本使用

```cpp
modeldeploy::RuntimeOption option;
option.use_ort_backend();
option.use_cpu();                    // CPU
// 或 option.use_gpu(0);             // CUDA
option.set_cpu_thread_num(4);

auto det = modeldeploy::vision::detection::UltralyticsDet("yolo11n.onnx", option);
```

### 2.2 GPU 推理（CUDA）

```cpp
option.use_ort_backend();
option.use_gpu(0);
// 可选：启用 FP16 + TensorRT EP（更快）
option.enable_fp16 = true;
option.enable_trt = true;
option.ort_option.trt_engine_cache_path = "./trt_engine";  // 缓存优化后的引擎
```

> **注意**：ORT 的 `enable_trt` 走的是 ORT 内置的 TensorRT Execution Provider（`TensorrtExecutionProvider`），与独立的 `TrtBackend`（`use_trt_backend()`）不同。前者在 ORT Session 内启用 TRT，后者直接用 TRT C++ API。

### 2.3 配置项（`ort_option`）

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `graph_optimization_level` | -1 | 图优化级别（-1 全开 / 0 关 / 1 基础 / 2 扩展 / 99 全） |
| `intra_op_num_threads` | -1 | 算子内线程数 |
| `inter_op_num_threads` | -1 | 图级并行线程数（需 `execution_mode=1`） |
| `execution_mode` | -1 | 图执行模式（0 顺序 / 1 并行） |
| `device` | CPU | 推理设备 |
| `device_id` | 0 | GPU 设备 ID |
| `enable_fp16` | false | FP16 推理（配合 TRT EP） |
| `optimized_model_filepath` | 空 | 优化后模型保存路径 |
| `trt_engine_cache_path` | 空 | TRT engine 缓存目录 |
| `log_severity_level` | -1 | 日志级别 |

## 3. TensorRT 后端

### 3.1 两种使用方式

**方式 A：在线构建（`TrtBackend`）**

```cpp
option.use_trt_backend();
option.use_gpu(0);
option.enable_fp16 = true;
// 动态 shape（可选，onnx 为动态输入时必设）
option.set_trt_min_shape("images:1x3x320x320");
option.set_trt_opt_shape("images:1x3x640x640");
option.set_trt_max_shape("images:4x3x1280x1280");

auto det = modeldeploy::vision::detection::UltralyticsDet("yolo11n.onnx", option);
```

首次运行会自动构建 engine 并缓存为 `model.onnx.engine`，后续加载直接用缓存。

**方式 B：预构建 engine（推荐）**

先用 `trtexec` 生成 `.engine`，再直接加载（避免在线构建耗时）：

```bash
trtexec --onnx=yolo11n.onnx \
        --saveEngine=yolo11n.engine \
        --fp16 \
        --minShapes=images:1x3x320x320 \
        --optShapes=images:1x3x640x640 \
        --maxShapes=images:4x3x1280x1280
```

```cpp
option.use_trt_backend();
option.use_gpu(0);
auto det = modeldeploy::vision::detection::UltralyticsDet("yolo11n.engine", option);
```

### 3.2 配置项（`trt_option`）

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `max_batch_size` | 32 | 最大 batch（TRT 8.x 已弃用，保留兼容） |
| `max_workspace_size` | 1GB | GPU 工作空间 |
| `enable_fp16` | false | FP16 推理 |
| `enable_log_info` | false | 打印构建日志 |
| `set_shape()` | — | 设置动态输入 shape 范围 |

### 3.3 动态 shape

动态输入模型必须设置 `min/opt/max` 三个 shape，否则构建失败：

```cpp
option.set_trt_min_shape("images:1x3x320x320");   // 最小
option.set_trt_opt_shape("images:1x3x640x640");   // 最优（常用尺寸）
option.set_trt_max_shape("images:4x3x1280x1280"); // 最大
```

> **提示**：`opt` 应设为你最常用的输入尺寸，TRT 会针对它做最深优化。

## 4. MNN 后端

### 4.1 基本使用

```cpp
option.use_mnn_backend();
option.use_cpu();
// 或 option.use_gpu(0);

auto det = modeldeploy::vision::detection::UltralyticsDet("model.mnn", option);
```

### 4.2 配置项（`mnn_option`）

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `forward_type` | AUTO | `MNN_FORWARD_CPU/AUTO/CUDA/OPENCL/VULKAN/METAL/NN` |
| `precision` | 正常 | 推理精度（正常/低/高） |
| `power_mode` | 正常 | 功耗模式 |
| `memory_mode` | — | 内存模式 |
| `cache_file_path` | 空 | 缓存文件（加速初始化） |

### 4.3 forward 类型

```cpp
// CUDA
option.mnn_option.forward_type = modeldeploy::mnn::MNN_FORWARD_CUDA;
// OpenCL
option.mnn_option.forward_type = modeldeploy::mnn::MNN_FORWARD_OPENCL;
// Vulkan
option.mnn_option.forward_type = modeldeploy::mnn::MNN_FORWARD_VULKAN;
```

> MNN 模型由 `.onnx/.torchscript` 通过 `MNNConvert` 转换，转换工具见 MNN 官方文档。

## 5. Sophgo 后端（算能 TPU）

### 5.1 环境要求

- 硬件：SE9-16 (BM1688) / CV186X 等算能 TPU
- 软件：libsophon（`libsophon-current` 含 bmlib/bmrt/bmcv），**不依赖 SOPHON-Sail**
- 编译：`-DENABLE_SOPHGO=ON`

### 5.2 基本使用

```cpp
option.use_sophgo_backend(0);
option.sophgo_option.bmodel_path = "yolo11n_bm1688.bmodel";

auto det = modeldeploy::vision::detection::UltralyticsDet(
    option.sophgo_option.bmodel_path, option);
det.get_preprocessor().set_size({1280, 1280});  // 与 bmodel 输入一致
```

### 5.3 配置项（`sophgo_option`）

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `device_id` | 0 | TPU 设备 ID |
| `bmodel_path` | 空 | `.bmodel` 路径（为空则用 `model_file`） |
| `use_device_input` | false | 设备内存直通（BMCV 零拷贝预处理） |

### 5.4 bmodel 生成

bmodel 由 ONNX 经 tpu-mlir 转换，见 [README](../../README.md#6-bmodel生成算能-sophgo-tpu) 和 [`tools/docker/sophgo/`](../../tools/docker/sophgo)：

```bash
cd tools/docker/sophgo
./build_docker.sh
docker run --rm -it -v <onnx目录>:/conv tpuc_dev:1.27 bash /conv/convert.sh \
    --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
    --chip bm1688 --quantize F16 --out yolo11n_bm1688.bmodel
```

> **注意**：tpu-mlir 对带 NMS 的 ONNX 有转换 bug，**转换前先把 NMS 从图中去掉**，NMS 由 SDK 后处理完成。模型输入需保持 SDK 默认的 letterbox + `/255` 归一化（即 `[0,1]`），**不要** `set_normalize(false)`。

### 5.5 零拷贝推理（BMCV）

Sophgo 后端支持 BMCV 设备端预处理零拷贝：

- 预处理在 TPU 设备内存上完成（vpp letterbox + convert_to）
- 通过 `Tensor::from_external_memory(..., Device::TPU)` 包装，`infer()` 识别 TPU 输入跳过 H2D 拷贝
- SOC 模式输出用 mmap 零拷贝读回

实测（BM1688, yolo11n 无 NMS, 1280×1280, INT8）：pre 22ms + infer 30ms + post 2ms ≈ 54ms/帧，检测 3 框与 ORT 一致。

## 6. 后端选择建议

| 需求 | 推荐后端 |
|------|---------|
| 快速原型 / 跨平台 | ORT (CPU) |
| 英伟达 GPU 最高性能 | TRT 或 ORT+TRT EP |
| 服务端多路视频 | TRT + 多线程 clone |
| 边缘盒子 / 低功耗 | Sophgo TPU |
| 移动端 | MNN |
| 国产化替代 | Sophgo |

## 7. 模型格式转换

| 目标 | 源 | 工具 |
|------|-----|------|
| `.engine` | `.onnx` | `trtexec` |
| `.mnn` | `.onnx` | `MNNConvert` |
| `.bmodel` | `.onnx` | tpu-mlir（见 `tools/docker/sophgo`） |
| 加密模型 | 任意 | `examples/tools/model_encrypted`（见 [模型加密](./encryption.md)） |
