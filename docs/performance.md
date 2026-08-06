# ModelDeploy 性能优化指南

涵盖推理提速、预处理加速、后处理优化、多线程与零拷贝，以及各平台的实测数据。

## 1. 推理提速

### 1.1 选择正确的后端

| 场景 | 后端 | 典型加速 |
|------|------|---------|
| NVIDIA GPU | TRT（或 ORT+TRT EP） | FP32 → FP16 约 2-4 倍 |
| 算能 TPU | Sophgo bmodel | 与 CPU 相比数倍到数十倍 |
| CPU | ORT（自动 SIMD） | 多线程 |

```cpp
// GPU 最优：ORT + TRT EP + FP16
option.use_ort_backend();
option.use_gpu(0);
option.enable_fp16 = true;
option.enable_trt = true;
option.ort_option.trt_engine_cache_path = "./trt_engine";
```

### 1.2 启用 FP16

GPU 场景 FP16 通常比 FP32 快 2-4 倍，且精度损失可忽略（检测/分割类任务）：

```cpp
option.enable_fp16 = true;
```

### 1.3 预构建 TRT engine

避免在线构建 engine（每次运行重新编译很慢），用 `trtexec` 预生成：

```bash
trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n.engine --fp16 \
        --minShapes=images:1x3x320x320 \
        --optShapes=images:1x3x640x640 \
        --maxShapes=images:4x3x1280x1280
```

### 1.4 INT8 量化（TPU）

Sophgo bmodel 用 INT8 比 F16 快约 4 倍（需校准表）。见 [后端详解](./backends.md#5-sophgo-后端算能-tpu)。

## 2. 预处理加速

### 2.1 问题：CPU 预处理在高分辨率下很慢

4K 图像（3840×2160）的 letterbox+归一化在 CPU 上可能达 20-30ms。原因是处理的是整幅图像的像素。

### 2.2 方案：硬件预处理

```cpp
// CUDA：预处理在 GPU 上
model.get_preprocessor().use_cuda_preproc();

// Sophgo：BMCV 设备端预处理（自动，无需配置）
option.use_sophgo_backend(0);
```

实测（BM1688, 1280 输入, INT8）：

| 阶段 | CPU 预处理 | BMCV 设备预处理 |
|------|-----------|----------------|
| Preprocess | 31ms | **22ms** |
| Inference | 48ms | **30ms**（零拷贝输入） |
| **Total** | 82ms | **54ms** |

### 2.3 降低输入分辨率

YOLO 输入尺寸直接决定推理量。640×640 → 320×320 推理量降 4 倍（精度略降）。小目标检测不建议过度降低。

## 3. 后处理优化

### 3.1 置信度阈值

无 NMS 模型（原始检测头输出）建议阈值 **≥ 0.5**。0.25 会带出大量低分候选，导致后处理 O(n²) NMS 变慢：

```cpp
model.get_postprocessor().set_conf_threshold(0.5f);
```

### 3.2 实例分割 mask

mask 生成曾把 160×160 的 mask resize 到整幅原图（4K = 830 万像素）再裁剪，极慢（100+ms）。SDK 已优化为**只对目标框区域处理**（约 8 倍加速）：

| 图 | 优化前 | 优化后 |
|----|--------|--------|
| 3840×2160 / 9 实例 | 113ms | **13.9ms** |
| 900×675 / 15 实例 | 14.7ms | **5.9ms** |

## 4. 零拷贝推理

### 4.1 原理

传统链路有 3 次不必要拷贝：
```
H2D 上传输入 → 推理 → D2H 读回输出
```

零拷贝链路：
```
预处理在设备内存完成 → 直接推理 → (SOC) mmap 读回
```

### 4.2 Tensor 设备感知

```cpp
// 设备预处理产出 Device::GPU/TPU Tensor
tensor.from_external_memory(device_ptr, shape, DataType::FP32,
    [](void*){}, Device::TPU, "input");
// infer() 识别设备输入，跳过 H2D
```

### 4.3 实测收益

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| Sophgo infer | 47.8ms（含 s2d 上传） | **29.6ms** |
| Sophgo total | ~81ms | **53.4ms** |

## 5. 多线程与 clone

### 5.1 模型实例线程不安全

同一实例不能多线程并发 `predict()`（IoBinding 独占）。用 `clone()` 为每线程建独立实例：

```cpp
auto c1 = model.clone();
auto c2 = model.clone();
// 每线程用自己的 clone
```

### 5.2 何时多线程有价值

| 场景 | 收益 |
|------|------|
| 单 GPU 单模型单路 | 无提升（GPU 已饱和） |
| 多路视频流 | 高（每路独立推理互不阻塞） |
| 多 GPU | 线性提升 |
| 延迟隐藏 | CPU 预处理与 GPU 推理重叠 |

实测（RTX 4060 Ti, YOLO11n 640, FP16）：单线程 83 FPS；4 路 clone 67 FPS（单 GPU 无提升，价值在多路）。

### 5.3 Sophgo TPU 多线程

实测（BM1688, 1280 INT8, 多实例 clone）：

| 线程数 | 吞吐 |
|--------|------|
| 1 | 18.7 FPS |
| 4 | 33.4 FPS |
| 8 | 33.5 FPS |

4 线程达吞吐上限（瓶颈是 TPU 单核算力 29ms）。

详见 [多线程推理指南](./multi_thread.md)。

## 6. 批量推理

`batch_predict` 一次推理多张图，利用 GPU/TPU 并行：

```cpp
std::vector<modeldeploy::ImageData> imgs = {img1, img2, img3};
std::vector<std::vector<modeldeploy::vision::DetectionResult>> results;
model.batch_predict(imgs, &results);
```

> 需模型支持 batch 输入（动态 shape 或固定 batch）。

## 7. 实测数据汇总

### 7.1 RTX 4060 Ti（GPU, ORT+TRT, FP16）

| 模型 | 输入 | Inference |
|------|------|-----------|
| YOLO11n-seg (NMS) | 640 | 6.5ms |
| YOLO11n-seg (裸TRT) | 640 | **2.5ms** |

### 7.2 BM1688（Sophgo TPU, INT8）

| 模型 | 输入 | Total |
|------|------|-------|
| yolo11n (无 NMS) | 1280 | **53.6ms** (pre 22 + infer 30 + post 2) |

### 7.3 CPU（ORT, 4 线程）

| 模型 | 输入 | Inference |
|------|------|-----------|
| YOLO11n-seg (NMS) | 640 | 43.7ms |

## 8. 性能分析工具

- `examples/demo_det/demo_benchmark.cpp` — 纯推理吞吐（排除预处理/后处理）
- `examples/demo_det/demo_profile.cpp` — 分阶段耗时
- `TimerArray` / `timers.print_benchmark()` — 各阶段统计（Preprocess/Inference/Postprocess/Total）
- `nvidia-smi` — GPU 利用率
- `bm-smi` — TPU 利用率

## 9. 优化优先级建议

1. **先看各阶段耗时**（`timers.print_benchmark()`），定位瓶颈在 pre/infer/post
2. **推理慢** → 换后端/FP16/INT8/降低分辨率
3. **预处理慢** → 硬件预处理（CUDA/BMCV）/降低分辨率
4. **后处理慢** → 提高置信度阈值 / 确认 mask 优化生效
5. **多路需求** → 多线程 clone / batch
