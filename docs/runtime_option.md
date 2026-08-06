# RuntimeOption 配置详解

`RuntimeOption` 是 ModelDeploy 的运行时配置结构，控制后端选择、推理设备、精度、线程数、动态 shape 等。所有模型类通过它初始化。

## 1. 快速示例

```cpp
modeldeploy::RuntimeOption option;
option.use_ort_backend();        // 后端
option.use_gpu(0);               // 设备
option.enable_fp16 = true;       // FP16
option.enable_trt = true;        // ORT 启用 TRT EP
option.set_cpu_thread_num(4);    // CPU 线程
option.set_model_path("yolo11n.onnx", "password");  // 模型 + 可选解密密码
```

## 2. 后端选择

| 方法 | 后端 | 说明 |
|------|------|------|
| `use_ort_backend()` | OnnxRuntime | `.onnx`，最通用 |
| `use_trt_backend()` | TensorRT | `.engine` / `.onnx`，GPU 最高性能 |
| `use_mnn_backend()` | MNN | `.mnn`，移动端/边缘 |
| `use_sophgo_backend(device_id)` | Sophgo | `.bmodel`，算能 TPU |

```cpp
option.use_ort_backend();      // ORT
option.use_trt_backend();      // 纯 TRT
option.use_mnn_backend();      // MNN
option.use_sophgo_backend(0);  // Sophgo TPU
option.sophgo_option.bmodel_path = "model.bmodel";
```

> `backend` 成员直接设置也可：`option.backend = modeldeploy::Backend::TRT;`

## 3. 设备选择

| 方法 | 设备 | 说明 |
|------|------|------|
| `use_cpu()` | CPU | 默认 |
| `use_gpu(gpu_id)` | NVIDIA GPU | 需 CUDA，配合 TRT/ORT |
| `use_opencl(device_id)` | OpenCL | ORT/MNN 支持 |

```cpp
option.use_cpu();          // CPU
option.use_gpu(0);         // 第 0 块 GPU
option.use_opencl(0);      // OpenCL
```

## 4. 精度与优化

```cpp
option.enable_fp16 = true;   // FP16 推理（GPU）
option.enable_trt = true;    // ORT 启用 TensorRT Execution Provider
```

| 配置 | 说明 |
|------|------|
| `enable_fp16` | 半精度推理，GPU 场景显著加速 |
| `enable_trt` | 仅对 ORT 后端生效，内部启用 TRT EP |

## 5. 线程配置

```cpp
option.set_cpu_thread_num(8);   // CPU 推理线程数
```

> **注意**：使用 GPU 时 CPU 线程数基本无效（计算在 GPU）。多线程并发请用 `clone()`（见 [多线程指南](./multi_thread.md)）。

## 6. TRT 动态 shape

动态输入模型（如 `[-1,3,-1,-1]`）需设置三个 shape 范围：

```cpp
// 格式：<输入名>:<shape>
option.set_trt_min_shape("images:1x3x320x320");   // 最小
option.set_trt_opt_shape("images:1x3x640x640");   // 最优（常用尺寸）
option.set_trt_max_shape("images:4x3x1280x1280"); // 最大
```

> `opt` 设为最常用尺寸，TRT 会针对它深度优化。动态 shape 既适用于 `TrtBackend`，也适用于 ORT 的 TRT EP。

## 7. 模型加载与加密

```cpp
// 文件路径
option.set_model_path("yolo11n.onnx");

// 加密模型（需要密码）
option.set_model_path("yolo11n.mdenc", "123456");
// 或
option.password = "123456";

// 从内存加载
option.model_from_memory = true;
option.model_buffer = model_buffer;
```

详见 [模型加密](./encryption.md)。

## 8. 各后端 Option 结构

### 8.1 `ort_option`（OnnxRuntime）

| 成员 | 默认 | 说明 |
|------|------|------|
| `graph_optimization_level` | -1 | 图优化级别（-1 全开 / 0 关 / 1 基础 / 2 扩展 / 99 全） |
| `intra_op_num_threads` | -1 | 算子内线程数 |
| `inter_op_num_threads` | -1 | 图级并行线程数 |
| `execution_mode` | -1 | 图执行模式（0 顺序 / 1 并行） |
| `device` | CPU | 推理设备 |
| `device_id` | 0 | GPU 设备 ID |
| `enable_fp16` | false | FP16 |
| `optimized_model_filepath` | 空 | 优化后模型保存路径 |
| `trt_engine_cache_path` | 空 | TRT engine 缓存目录 |
| `external_stream` | nullptr | 外部 CUDA 流 |

```cpp
option.ort_option.graph_optimization_level = 99;
option.ort_option.trt_engine_cache_path = "./trt_engine";
```

### 8.2 `mnn_option`（MNN）

| 成员 | 默认 | 说明 |
|------|------|------|
| `forward_type` | AUTO | CPU/AUTO/CUDA/OPENCL/VULKAN/METAL/NN |
| `precision` | 正常 | 推理精度 |
| `power_mode` | 正常 | 功耗模式 |
| `cache_file_path` | 空 | 缓存文件 |

```cpp
option.mnn_option.forward_type = modeldeploy::mnn::MNN_FORWARD_CUDA;
```

### 8.3 `trt_option`（TensorRT）

| 成员 | 默认 | 说明 |
|------|------|------|
| `max_batch_size` | 32 | 最大 batch |
| `max_workspace_size` | 1GB | GPU 工作空间 |
| `enable_fp16` | false | FP16 |
| `enable_log_info` | false | 打印构建日志 |

### 8.4 `sophgo_option`（Sophgo）

| 成员 | 默认 | 说明 |
|------|------|------|
| `device_id` | 0 | TPU 设备 ID |
| `bmodel_path` | 空 | `.bmodel` 路径 |
| `use_device_input` | false | 设备内存直通 |

## 9. 完整配置模板

### CPU + ORT

```cpp
option.use_ort_backend();
option.use_cpu();
option.set_cpu_thread_num(4);
```

### GPU + ORT + TRT EP（推荐 GPU 方案）

```cpp
option.use_ort_backend();
option.use_gpu(0);
option.enable_fp16 = true;
option.enable_trt = true;
option.ort_option.trt_engine_cache_path = "./trt_engine";
```

### 纯 TensorRT

```cpp
option.use_trt_backend();
option.use_gpu(0);
option.enable_fp16 = true;
option.set_trt_min_shape("images:1x3x320x320");
option.set_trt_opt_shape("images:1x3x640x640");
option.set_trt_max_shape("images:4x3x1280x1280");
```

### MNN GPU

```cpp
option.use_mnn_backend();
option.use_gpu(0);
option.mnn_option.forward_type = modeldeploy::mnn::MNN_FORWARD_CUDA;
```

### Sophgo TPU

```cpp
option.use_sophgo_backend(0);
option.sophgo_option.bmodel_path = "model.bmodel";
```

## 10. 常见问题

- **后端编译开关未开**：如 `use_trt_backend()` 但 `ENABLE_TRT=OFF`，会运行时报错。需按后端要求配置 CMake 开关。
- **动态 shape 未设置**：TRT 加载动态输入模型报错，用 `set_trt_*_shape` 设置范围。
- **加密模型密码错误**：模型加载失败，检查 `password`。
- **GPU 不生效**：确认 `use_gpu()` 且后端支持 GPU（ORT 需 `enable_trt` 或 TRT backend）。
