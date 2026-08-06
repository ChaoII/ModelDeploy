# ModelDeploy 预处理详解

预处理是推理前对输入图像的处理链。ModelDeploy 通过 `ImageData`（图像类型）+ `*Preprocessor`（模型预处理器）+ 处理器后端（CPU/CUDA/BMCV）实现。

## 1. ImageData（图像类型）

`ImageData` 是 ModelDeploy 的图像封装，底层基于 OpenCV Mat，默认 **BGR HWC uint8** 布局。

### 1.1 读取 / 保存

```cpp
// 读图
auto img = modeldeploy::ImageData::imread("test.jpg");

// 保存
img.imwrite("out.jpg");

// 显示
img.imshow("window");

// 内存编码 / 解码
std::vector<uchar> buf = modeldeploy::ImageData::imencode(img, ".jpg");
auto img2 = modeldeploy::ImageData::imdecode(buf);
```

### 1.2 构造

```cpp
// 空图
modeldeploy::ImageData img;
// 指定尺寸/类型
modeldeploy::ImageData img(640, 480, CV_8UC3);
// 从 cv::Mat
modeldeploy::ImageData img(cv_mat);
// 从裸内存（可共享外部内存，零拷贝）
modeldeploy::ImageData img = modeldeploy::ImageData::from_raw(data, w, h, CV_8UC3, /*copy=*/false);
```

### 1.3 常用属性

```cpp
img.width();        // 宽
img.height();       // 高
img.channels();     // 通道数
img.type();         // OpenCV 类型
img.data();         // 数据指针
img.bytes();        // 字节数
img.empty();        // 是否为空
img.clone();        // 深拷贝
```

## 2. 预处理算子

### 2.1 基础算子

```cpp
img.resize(w, h);                          // 缩放
img.center_crop({w, h});                   // 中心裁剪
img.letter_box({640, 640}, 114);           // 等比缩放 + 灰边填充
img.pad(top, bottom, left, right, value);  // 填充
img.normalize(mean, std, scale, swap_rb);  // 归一化
img.convert(alpha, beta);                  // 仿射变换（如 /255）
img.cast(CV_32FC3, scale);                 // 类型转换
img.cvt_color(CV_BGR2RGB);                 // 颜色转换
img.rotate(flag);                          // 旋转（0/1/2）
img.crop(rect);                            // 裁剪
img.permute();                             // HWC → CHW
```

### 2.2 融合算子（减少拷贝）

```cpp
img.fuse_normalize_and_permute();  // 归一化 + 转置一步
img.fuse_convert_and_permute();    // 仿射 + 转置一步
img.fuse_resize_and_pad();         // 缩放 + 填充一步
```

### 2.3 转 Tensor

```cpp
// 单图 → Tensor (CHW)
modeldeploy::Tensor tensor;
img.to_tensor(&tensor, /*copy=*/true);

// 多图 → batch Tensor (NCHW)
std::vector<modeldeploy::ImageData> imgs = {...};
modeldeploy::Tensor batch;
modeldeploy::ImageData::images_to_tensor(imgs, &batch);
```

## 3. 模型预处理器（Preprocessor）

每个模型类自带预处理器，负责把 `ImageData` 转成模型输入 Tensor，**内部自动完成 letterbox + /255 归一化**。

```cpp
auto& pre = model.get_preprocessor();

pre.set_size({640, 640});        // 模型输入尺寸（重要！）
pre.set_padding_value(114);      // letterbox 填充值（默认 114）
pre.set_scale_factor(1.0/255);   // 归一化系数
```

> **YOLO 系列约定**：SDK 默认预处理 = letterbox 等比缩放 + 填充 114 + `/255` 归一化到 `[0,1]`，与 Ultralytics 训练一致。**正常模型不需要** `set_normalize(false)`（除非特殊模型明确要求 `[0,255]` 输入）。

### 各模型预处理器接口

| 模型 | 预处理器 | 常用配置 |
|------|---------|---------|
| `UltralyticsDet/Seg/Pose/Obb` | `Ultralytics*Preprocessor` | `set_size`, `set_padding_value` |
| `Classification` | `ClassificationPreprocessor` | `set_size`, `set_center_crop` |
| `Scrfd` | `ScrfdPreprocessor` | `set_size` |
| `PaddleOCR` | `DBDetectorPreprocessor` 等 | `set_max_side_len` |
| `PedestrianAttribute` | `PedestrianAttributePreprocessor` | `set_det_input_size`, `set_cls_input_size` |

## 4. 处理器后端（硬件加速）

预处理默认在 CPU 执行。ModelDeploy 抽象了 `VisionProcessorBackend`，支持 GPU（CUDA）和 Sophgo（BMCV）硬件加速：

| 处理器后端 | 设备 | 加速点 |
|-----------|------|--------|
| `CpuProcessorBackend` | CPU | 默认，NEON/SVE/AVX2/AVX512 SIMD |
| `CudaProcessorBackend` | CUDA GPU | letterbox/normalize 在 GPU 上，输出 GPU Tensor |
| `SophgoProcessorBackend` | 算能 TPU | BMCV vpp 硬件，输出设备内存 |

### 4.1 CUDA 预处理（GPU）

```cpp
model.get_preprocessor().use_cuda_preproc();
```

预处理在 CUDA 上完成，产出 `Device::GPU` 的 Tensor，ORT backend 识别后直接 IoBinding 推理（**零拷贝输入**）。

### 4.2 Sophgo BMCV 预处理（TPU）

Sophgo 后端自动使用 BMCV 设备端预处理（无需手动配置）：

```cpp
option.use_sophgo_backend(0);
option.sophgo_option.bmodel_path = "model.bmodel";
// 自动走 BMCV：vpp letterbox + convert_to 在 TPU 设备内存完成
```

产出 `Device::TPU` 的 Tensor，`infer()` 识别后跳过 H2D 拷贝直接 `bmrt_launch_tensor`（**零拷贝**）。

## 5. 零拷贝推理链路

### 5.1 什么是零拷贝

传统链路：

```
ImageData(CPU) → 预处理(CPU) → CPU Tensor → H2D 上传 → 推理 → D2H 读回
```

零拷贝链路：

```
ImageData → 预处理(GPU/TPU设备内存) → Device Tensor → 直接推理(无 H2D) → mmap 读回
```

### 5.2 Tensor 设备感知

`Tensor` 通过 `device()` 表达数据所在设备，`from_external_memory()` 绑定外部设备内存：

```cpp
// 绑定设备内存（不拷贝、不拥有）
tensor.from_external_memory(device_ptr, shape, DataType::FP32,
    [](void*){}, Device::TPU, "input");
```

后端 `infer()` 识别 `device()==GPU/TPU` 且 `!owns_data` 的输入，跳过 H2D 拷贝直接推理。

### 5.3 实测效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| Sophgo (BM1688, 1280 INT8) | pre 31 + infer 48 + post 3 = 82ms | pre 22 + infer 30 + post 2 = 54ms |
| CUDA (RTX 4060 Ti, ORT) | — | infer 6.5ms（含零拷贝输入） |

## 6. 常见问题

- **为什么检测不到目标**：确认 `set_size` 与模型输入一致；YOLO 系列确认阈值（无 NMS 模型建议 ≥0.5）。
- **为什么 CPU 预处理慢**：高分辨率图（4K）预处理开销大，可用 CUDA/BMCV 硬件加速。
- **零拷贝不生效**：确认预处理产出的是 `Device::GPU/TPU` Tensor（`use_cuda_preproc()` 或 sophgo 后端），且未手动拷贝回 CPU。
- **mask 结果慢**：实例分割 mask 后处理对大图 resize 开销大，SDK 已优化为只对目标框区域处理（见 [性能优化](./performance.md)）。
