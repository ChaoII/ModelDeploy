# AI 智能安防监控平台 — 20路 25FPS+ 性能优化 PRD

## 文档信息

| 字段 | 内容 |
|------|------|
| 版本 | v1.0 |
| 作者 | Agent |
| 日期 | 2026-06-16 |
| 状态 | 草案 |
| 硬件目标 | RTX 4060 Ti 16GB / i7-12700 / 32GB RAM |
| 软件目标 | 20 路 RTSP 实时流，每路 25FPS，编码输出，HTTP-FLV 预览 |

---

## 1. 背景与现状

### 1.1 当前架构

```
输入(RTSP) → Decoder(CUVID) → NV12 → Process(InferGroup + Draw) → BGR → Encoder(NVENC/x264) → 输出(RTMP/RTSP/FLV)
                                       ↑
                               BatchScheduler(可选)
```

每路 `Pipeline` 启动 3 个线程：

| 线程 | 函数 | 主要耗时 |
|------|------|---------|
| decode_thread | `decode_loop()` | 1. 解码 CUVID ~1ms/帧（GPU）<br>2. GPU→CPU 拷贝 ~0.5ms/帧<br>3. NV12 行拷贝 ~0.3ms/帧 |
| process_thread | `process_loop()` → `do_process()` | 1. NV12→BGR CUDA kernel ~0.5ms/帧（GPU）<br>2. 推理 ~3-5ms/帧（GPU, ORT+TRT EP FP16）<br>3. **CPU 绘制（vis_det）~1-3ms/帧**<br>4. 结果组装 ~0.1ms/帧 |
| encode_thread | `encode_loop()` | 1. BGR→NV12 **CPU sws_scale ~1-2ms/帧**<br>2. NVENC 编码 ~0.5-1ms/帧（GPU）<br>3. 写入输出流 ~0.2ms |

### 1.2 实测数据（当前运行实例）

- GPU 显存占用：~5GB/16GB（含系统进程）
- NVENC 会话数：2路
- 平均 FPS 表现：~31 FPS（2 路合计数）

### 1.3 瓶颈总结

| 瓶颈 | 位置 | 每帧耗时 | 500 帧/秒总耗时 | 性质 |
|------|------|---------|----------------|------|
| CPU 绘制 | `vis_det` (OpenCV) | 1-3ms | 500-1500ms | **核心瓶颈，CPU 完全饱和** |
| CPU BGR→NV12 | `sws_scale` | 1-2ms | 500-1000ms | **核心瓶颈** |
| 逐帧推理 | BatchScheduler 串行调用 | 3-5ms/帧 | 按 batch 累积 | GPU 利用率不足 |
| GPU↔CPU 乒乓拷贝 | 多次跨 PCIe | ~2ms/帧 | 1000ms PCIe 带宽 | 延迟叠加 |
| NVENC 会话数 | 单卡最多 8 路并发 | — | — | 超 8 路需 fallback CPU 编码 |

---

## 2. 目标

### 2.1 性能目标

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| 最大路数 | ~6-8 路 | **20 路** |
| 单路帧率 | ~25 FPS | **25 FPS** |
| CPU 利用率 | ~80-100% | < 50% |
| GPU 利用率 | ~30-40% | ~60-80% |
| 端到端延迟 | ~100-150ms | < 200ms |
| 预览路编码 | 全部编码 | 仅预览路编码 |

### 2.2 质量目标

- 推理精度不变（保持 FP16）
- 编码画质不退化（预览路 2.5Mbps H.264）
- 报警截图功能不受影响
- 重连机制正常工作

---

## 3. 优化方案

### 3.1 Phase 1：零成本优化（无需 GPU 内核开发）

#### 3.1.1 非预览路跳过编码 + 绘制（P1）

**改造文件：**

| 文件 | 修改 |
|------|------|
| `application/config.hpp` | `TaskConfig` 新增 `bool enable_preview = true` |
| `application/pipeline.cpp:decode_loop()` | `enable_preview=false` 时不创建 `encoder_`，跳过 `encode_loop` |
| `application/pipeline.cpp:process_loop()` | 非预览路每 5 帧仅画一帧用于快照，其余只推理不画不编码 |
| `application/http_server.cpp` | 预览接口检查 `enable_preview`，非预览路返回 404 |
| `application/pipeline_manager.hpp` | `TaskStatus` 新增 `enable_preview` 字段 |

**数据流对比：**

```
预览路：decode → infer → draw(全量) → encode → push
报警路：decode → infer → draw(每5帧·仅快照) → 存储结果 → 不编码
```

**收益预估：** 若 20 路中仅 4 路预览，编码+绘制负载降至 20%。CPU 绘制耗时从 100% → 20%。

#### 3.1.2 推理间隔控制（P2）

**现状：** `ModelConfig.interval = 1` 已存在但未在流程中生效。

**改造文件：**

| 文件 | 修改 |
|------|------|
| `application/pipeline.cpp:do_process()` | 在调用 `infer_group_->run_models()` 前检查 `wall_time_sec` 与上次推理时间戳间隔 |
| `application/infer_group.cpp:should_process()` | 全局帧计数改为按模型独立计数，第 N 帧仅当 `N % interval == 0` 时推理 |

**实现逻辑：**

```cpp
// pipeline.cpp:do_process 内
bool need_infer = true;
static std::unordered_map<std::string, int64_t> last_infer_time;
auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(...).count();
auto it = last_infer_time.find(cfg_.id);
if (it != last_infer_time.end()) {
    int64_t interval_us = 1000000 / (cfg_.encoder.fps * interval_);
    need_infer = (now_us - it->second) >= interval_us;
}
if (need_infer) {
    infer_group_->run_models(...);
    last_infer_time[cfg_.id] = now_us;
}
```

**收益预估：** `interval=2` 推理负载减半，`interval=3` 减至 1/3。

#### 3.1.3 BatchScheduler 真批推理（P3）

**现状：** `batch_scheduler.cpp:143` 对 batch 内每帧依次调用 `prototype->infer()`。

**改造文件：**

| 文件 | 修改 |
|------|------|
| `application/inference_engine.hpp` | 新增 `bool batch_infer(const std::vector<ImageData>&, std::vector<InferResult>*)` |
| `application/inference_engine.cpp` | 实现 `batch_infer()`，调用 `det_model_->batch_predict()` |
| `application/batch_scheduler.cpp:process_batch()` | 收集 batch 内所有 BGR image 为 vector，一次性调用 `prototype->batch_infer()` |
| `application/batch_scheduler.hpp` | `max_batch_size_` 默认值从 8 改为 4（YOLO FP16 显存限制） |

**关键代码：**

```cpp
// BatchScheduler::process_batch() 改造后核心逻辑
void BatchScheduler::process_batch(
    std::vector<std::pair<BatchRequest, std::shared_ptr<BatchResult>>>& batch) {
    
    // 1. 取公共尺寸（不同尺寸则回退逐帧）
    int ref_w = batch[0].first.width;
    int ref_h = batch[0].first.height;
    for (auto& [req, _] : batch) {
        if (req.width != ref_w || req.height != ref_h) {
            // 尺寸不一致 → fallback 逐帧（代码略）
            return fallback_sequential(batch);
        }
    }
    
    // 2. NV12→BGR 转换（CUDA kernel，已有）
    std::vector<ImageData> bgr_images = batch_nv12_to_bgr(batch);
    
    // 3. 一次 batch_predict 完成 N 帧推理
    std::vector<std::vector<DetectionResult>> all_results;
    {
        std::lock_guard<std::mutex> lock(models_mtx_);
        // 要求原型模型内部 batch_predict 线程安全（ORT Session 支持）
        entry.prototype->det_model_->batch_predict(bgr_images, &all_results);
    }
    
    // 4. 拆分结果
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& [req, res] = batch[i];
        for (auto& det_result : all_results[i]) {
            InferResult r;
            r.model_name = entry.cfg.name;
            r.type = "detection";
            // ... 转换 DetectionResult → InferResult ...
            res->results.push_back(std::move(r));
        }
        res->bgr_image = bgr_images[i];
        res->ready = true;
    }
}
```

**`InferenceEngine` 新增方法：**

```cpp
// inference_engine.h
bool batch_infer(const std::vector<modeldeploy::vision::ImageData>& images,
                 std::vector<InferResult>* results);

// inference_engine.cpp
bool InferenceEngine::batch_infer(
    const std::vector<ImageData>& images,
    std::vector<InferResult>* results) {
    if (cfg_.type == "detection" && det_model_) {
        std::vector<std::vector<DetectionResult>> det_results;
        if (!det_model_->batch_predict(images, &det_results))
            return false;
        results->resize(det_results.size());
        for (size_t i = 0; i < det_results.size(); ++i)
            convert_det_results(det_results[i], &(*results)[i]);
        return true;
    }
    // fallback 逐帧
    ...
}
```

**收益预估：** 4 帧 batch 推理时，GPU 利用率从 4 次 kernel launch 降为 1 次，推理吞吐提升 40-60%。

#### 3.1.4 预编译 TRT Engine（P4）

**改造文件：**

| 文件 | 修改 |
|------|------|
| `application/inference_engine.cpp:load()` | `backend=="trt"` 路径已存在；优化点：自动检测 `.engine` 后缀走 TRT 路径 |
| `application/config.hpp` | `ModelConfig` 新增 `trt_cache_path` 字段 |
| 构建脚本 | 新增 `tools/export_trt_engine.py` 批量转换脚本 |

**`tools/export_trt_engine.py`（新建）：**

```python
#!/usr/bin/env python3
"""批量从 ONNX 生成 TRT .engine 文件"""
import subprocess, sys, os
from pathlib import Path

TRTEXEC = r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\bin\trtexec.exe"

models = [
    ("yolo11n_nms.onnx", (640, 640)),
    # ... 更多模型 ...
]

for onnx_path, (w, h) in models:
    engine_path = onnx_path.replace(".onnx", ".engine")
    cmd = [
        TRTEXEC, f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes=images:1x3x{h}x{w}",
        f"--optShapes=images:4x3x{h}x{w}",
        f"--maxShapes=images:4x3x{h}x{w}",
    ]
    subprocess.run(cmd, check=True)
    print(f"Generated: {engine_path}")
```

**收益预估：** 消除 ORT 调度开销，每帧节省 ~0.5-1ms。

---

### 3.2 Phase 2：GPU 绘制内核

#### 3.2.1 文件清单（新建）

| 文件 | 类型 | 内容 |
|------|------|------|
| `csrc/vision/common/processors/draw_gpu.cuh` | 头文件 | `draw_boxes_gpu()` 声明 |
| `csrc/vision/common/processors/draw_gpu.cu` | CUDA 实现 | Kernel + 主机包装函数 |

#### 3.2.2 接口设计

```cpp
// draw_gpu.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace modeldeploy::vision {

// GPU 绘制矩形框参数
struct GpuDrawBox {
    int x1, y1, x2, y2;    // 像素坐标
    float score;
    int label_id;
    uint8_t r, g, b;        // BGR 颜色
    char label[32];         // 预格式化的标签文字 "person: 0.95"
};

// 在 GPU BGR buffer 上绘制一批检测框
// bgr: 设备端 [H x W x 3] uint8 buffer（就地修改）
MODELDEPLOY_CXX_EXPORT bool draw_boxes_gpu(
    uint8_t* bgr, int width, int height,
    const GpuDrawBox* d_boxes, int num_boxes,
    float alpha = 0.15f,
    cudaStream_t stream = nullptr);

}
```

#### 3.2.3 Kernel 设计

**`kernel_draw_rect`：** 每个检测框分配 1 个 block，block 内线程负责框内及边框像素。

```
block(256)  ← 128 用于框内填充 + 128 用于边框+文字
grid(num_boxes)
```

**伪代码：**

```cuda
__global__ void kernel_draw_boxes(
    uint8_t* __restrict__ bgr, int width, int height,
    const GpuDrawBox* __restrict__ boxes, int num_boxes,
    float alpha)
{
    int idx = blockIdx.x; // 框索引
    if (idx >= num_boxes) return;
    auto& box = boxes[idx];
    int box_w = box.x2 - box.x1;
    int box_h = box.y2 - box.y1;

    int tid = threadIdx.x;
    // 区域 A: 框内行（用于填充 + 边框）
    // 区域 B: 文字行

    // 略：逐线程映射到矩形区域像素，判断是否在边框上 → 画线，
    // 否则画填充色 * alpha + 原像素 * (1-alpha)
    // 文字区域：使用内嵌 8x16 bitmap font 画 label
}
```

**文字渲染方案：** 内嵌一个简化的 8x16 ASCII bitmap font（约 256 字符 × 16 字节），无需 FreeType 依赖。首个版本支持 0-9、A-Z、a-z、句点、冒号即可。

#### 3.2.4 主机包装函数

复用 `nv12_to_bgr.cu` 的既有模式：
- `thread_local` workspace 管理 device memory
- `cudaPointerGetAttributes` 检测输入是否已在 device
- 调用方传入 stream，或内部创建临时 stream

#### 3.2.5 集成到 DrawEngine

```cpp
// draw_engine.hpp 新增
void draw_gpu(uint8_t* gpu_bgr, int width, int height,
              const std::vector<InferResult>& results);

// draw_engine.cpp 新增
void DrawEngine::draw_gpu(...) {
    // 1. 将 InferResult 转换为 GpuDrawBox 数组（CPU 端组装，开销很小）
    std::vector<GpuDrawBox> h_boxes;
    for (auto& r : results) {
        for (auto& b : r.boxes) {
            GpuDrawBox gb;
            gb.x1 = b.x; gb.y1 = b.y;
            gb.x2 = b.x + b.w; gb.y2 = b.y + b.h;
            gb.score = b.score;
            gb.label_id = b.label_id;
            gb.r = color_map[b.label_id][0];
            gb.g = color_map[b.label_id][1];
            gb.b = color_map[b.label_id][2];
            snprintf(gb.label, 32, "%s:%.2f",
                     label_name.c_str(), b.score);
            h_boxes.push_back(gb);
        }
    }
    // 2. 拷贝到 device
    GpuDrawBox* d_boxes;
    cudaMalloc(&d_boxes, h_boxes.size() * sizeof(GpuDrawBox));
    cudaMemcpy(d_boxes, h_boxes.data(), ...);
    // 3. 调用 GPU 绘制
    draw_boxes_gpu(gpu_bgr, width, height, d_boxes, h_boxes.size());
    // 4. 清理
    cudaFree(d_boxes);
}
```

#### 3.2.6 Pipeline 集成

```cpp
// pipeline.cpp:do_process() 新增 GPU 路径分支
if (cfg_.device == "gpu" && use_gpu_draw) {
    // gpu_bgr 是推理后结果在 GPU 上的 BGR buffer
    draw_engine_->draw_gpu(gpu_bgr, pf.width, pf.height, results);
    // 结果留在 GPU，供下一步 GPU 编码使用
} else {
    // 现有 CPU 绘制路径
    draw_engine_->draw(bgr_image, results);
}
```

#### 3.2.7 收益预估

| 指标 | CPU 绘制 | GPU 绘制 |
|------|----------|----------|
| 每帧耗时 | 1-3ms | ~0.1-0.2ms |
| 20路×25FPS CPU 耗时 | ~1000ms/s | ~50ms/s |
| CPU 核心占用 | ~8 核 | <0.5 核 |

---

### 3.3 Phase 3：GPU BGR→NV12 编码前转换

#### 3.3.1 文件清单（新建）

| 文件 | 内容 |
|------|------|
| `csrc/vision/common/processors/bgr_to_nv12.cuh` | `bgr_to_nv12_cuda()` 声明 |
| `csrc/vision/common/processors/bgr_to_nv12.cu` | CUDA 实现 |

#### 3.3.2 接口设计

```cpp
// bgr_to_nv12.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace modeldeploy::vision {

// 将 GPU BGR [HxWx3] 平面格式 → GPU NV12 [HxWx1.5] 紧凑格式
// output 需预分配 H*W*3/2 字节
// 色彩空间: BT.709 limited range（与编码器一致）
MODELDEPLOY_CXX_EXPORT bool bgr_to_nv12_cuda(
    const uint8_t* bgr, int width, int height,
    uint8_t* nv12, cudaStream_t stream = nullptr);

}
```

#### 3.3.3 Kernel 设计

**`kernel_bgr_to_nv12_y`：** 每个像素一个 thread，计算 Y 值并写入 NV12 Y plane。

```cuda
// BT.709 限范围 YUV 转换
// Y = 16 + (0.2126*R + 0.7152*G + 0.0722*B)
// U = 128 + (-0.1146*R - 0.3854*G + 0.5000*B)
// V = 128 + (0.5000*R - 0.4542*G - 0.0458*B)

__global__ void kernel_bgr_to_nv12_y(
    const uint8_t* __restrict__ bgr,
    uint8_t* __restrict__ nv12,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    int R = bgr[idx + 2];
    int G = bgr[idx + 1];
    int B = bgr[idx];

    int Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16;
    nv12[y * width + x] = clamp_u8(Y);
}

__global__ void kernel_bgr_to_nv12_uv(
    const uint8_t* __restrict__ bgr,
    uint8_t* __restrict__ nv12,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width / 2 || y >= height / 2) return;

    // 2x2 块 average
    int sum_r = 0, sum_g = 0, sum_b = 0;
    for (int dy = 0; dy < 2; dy++)
        for (int dx = 0; dx < 2; dx++) {
            int idx = ((y*2+dy) * width + (x*2+dx)) * 3;
            sum_r += bgr[idx + 2];
            sum_g += bgr[idx + 1];
            sum_b += bgr[idx];
        }
    int R = sum_r / 4, G = sum_g / 4, B = sum_b / 4;

    int U = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128;
    int V = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128;

    size_t uv_offset = width * height; // Y plane 后接 UV
    nv12[uv_offset + (y * width + x) * 2 + 0] = clamp_u8(U);
    nv12[uv_offset + (y * width + x) * 2 + 1] = clamp_u8(V);
}
```

#### 3.3.4 集成到 StreamEncoder

```cpp
// stream_encoder.cpp 新增
bool StreamEncoder::encode_from_gpu(const uint8_t* gpu_bgr, int width, int height) {
    // 1. 分配 GPU NV12 buffer（复用缓存）
    // 2. 调用 bgr_to_nv12_cuda
    // 3. cudaMemcpy NV12 → enc_frame_->data
    // 4. avcodec_send_frame / avcodec_receive_packet
    // 5. av_interleaved_write_frame
}
```

---

### 3.4 Phase 4（可选）：全 GPU 零拷贝流水线

#### 3.4.1 改造 StreamDecoder 输出 GPU NV12

`stream_decoder.cpp:read_one_frame()` 新增 `bool read_one_frame_gpu(DecodedFrameGpu* out)` 返回 GPU device 指针。

```cpp
struct DecodedFrameGpu {
    uint8_t* d_nv12;     // 设备端 NV12 紧凑 buffer
    int width, height;
    int64_t pts;
};
```

解码器从 `av_hwframe_transfer_data`（CPU 下载）改为：
- CUVID 解码 → 帧留在 GPU（`AVFrame->data[0]/data[1]` 为 CUDA 指针）
- `cudaMemcpy` 到固定大小的 device buffer 即可
- 跳过 GPU→CPU→GPU 的来回拷贝

#### 3.4.2 完整零拷贝路径

```
CUVID(GPU NV12) 
  → nv12_to_bgr_cuda(GPU BGR, 已在 GPU)
  → batch_predict(GPU input→GPU output, 全 GPU)
  → draw_boxes_gpu(GPU BGR 上就地绘制)
  → bgr_to_nv12_cuda(GPU NV12)
  → cudaMemcpy(NV12 → NVENC input buffer)
  → nvenc encode
```

**收益预估：** GPU↔CPU 来回 4 次拷贝（~2ms/帧 × 500FPS = ~1000ms/s PCIe 传输） → 0 次拷贝。

---

## 4. 优先级与工作量

| 优先级 | 阶段 | 工作量 | 依赖 | 单路收益 | 多路收益 | 复杂度 |
|--------|------|--------|------|---------|---------|--------|
| P0 | 1.1 非预览路跳编码 | 0.5天 | 无 | 编码节省 ~2ms | +40-60% 路数 | 低 |
| P0 | 1.2 推理间隔 | 0.5天 | 无 | 推理减半~3倍 | +50-200% 有效容量 | 低 |
| P0 | 1.3 真批推理 | 2天 | 无 | −30% | +30-50% GPU 利用率 | 中 |
| P1 | 1.4 预编译 TRT | 1天 | 无 | −0.5~1ms | +10-20% | 低 |
| **P1** | **2 GPU 绘制** | **3-5天** | **无** | **−1~3ms CPU** | **核心瓶颈消除** | **高** |
| P2 | 3 GPU BGR→NV12 | 2-3天 | Phase 2 | −1~2ms CPU | 核心瓶颈消除 | 中 |
| P3 | 4 全 GPU 零拷贝 | 4-5天 | Phase 2+3 | −2ms 延迟 | 延迟极限优化 | 高 |

---

## 5. 非功能性要求

### 5.1 向后兼容

- 所有优化默认为**关闭**，通过 `enable_preview`、`use_gpu_draw` 等开关控制
- CPU 绘制路径完整保留作为 fallback
- 非 GPU 系统自动回退到 CPU 路径（`#ifdef WITH_GPU` 条件编译）

### 5.2 错误处理

- GPU 绘制失败时：记录日志并回退 CPU 绘制
- CUDA OOM 时：减少 batch size 或关闭 GPU 绘制
- NVENC 会话数超限时：自动 fallback 到 `libx264`，或在 `enable_preview=false` 时静默跳过编码

### 5.3 监控

在 `PerfStats` 中新增指标：

| 指标 | 类型 | 说明 |
|------|------|------|
| `gpu_draw_us` | int64 | GPU 绘制耗时 |
| `gpu_nv12_us` | int64 | GPU BGR→NV12 转换耗时 |
| `gpu_decode_us` | int64 | GPU 解码耗时（含下载） |
| `preview_enabled` | bool | 是否为预览路 |

---

## 6. 验收标准

### 6.1 功能验收

| 测试用例 | 预期结果 |
|----------|----------|
| 20 路 RTSP 同时接入 | 全部初始化成功，无崩溃 |
| 每路 25 FPS 推流 | 编码输出 PTS 连续，播放器平滑 |
| 非预览路不输出流 | 编码器未启动，仅缓存报警截图 |
| `interval=3` 推理减负 | 推理负载降至 1/3，帧率保持 25 |
| Batch 推理精度对齐 | 单帧推理 vs 4 帧 batch 推理结果一致 |

### 6.2 性能验收

| 指标 | 基准（优化前） | 目标（Phase 1） | 目标（Phase 2+3） |
|------|---------------|----------------|------------------|
| 20 路 CPU 利用率 | N/A | < 70% | < 40% |
| 20 路 GPU 利用率 | N/A | < 70% | < 80% |
| 单路端到端延迟 | ~120ms | < 200ms | < 150ms |
| 编码质量 (SSIM) | — | > 0.95 | > 0.95 |
| 推理精度 (mAP) | baseline | 不变 | 不变 |
| 显存占用 20 路 | N/A | < 14GB | < 14GB |

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| ORT Session 并发推理非线程安全 | 中 | 高 | 加互斥锁 serialize，或在 BatchScheduler 单线程中执行 |
| TRT Engine 动态 shape 不稳定 | 低 | 高 | 固定输入尺寸，不使用动态 shape |
| 非预览路报警截图缺失 | 低 | 中 | 保留每 5 帧一帧的绘制+快照 |
| NVENC 并发会话数不够 | 中 | 中 | 超过 8 路时自动 fallback x264 或限制预览路数 |
| 显存不足 20 路 | 低 | 高 | 设置最大 batch 数，超出时拒绝创建任务 |

---

## 8. 附录

### 8.1 关键文件索引

| 文件 | 职责 |
|------|------|
| `application/config.hpp` | 所有配置结构体定义 |
| `application/pipeline.hpp/cpp` | 三段流水线核心 |
| `application/inference_engine.hpp/cpp` | 推理引擎封装 |
| `application/batch_scheduler.hpp/cpp` | 批量调度器 |
| `application/draw_engine.hpp/cpp` | 绘制引擎 |
| `application/stream_encoder.hpp/cpp` | H.264 编码器 |
| `application/stream_decoder.hpp/cpp` | FFmpeg 解码器 |
| `application/infer_group.hpp/cpp` | 多模型并行推理组 |
| `csrc/vision/common/processors/nv12_to_bgr.cuh/.cu` | NV12→BGR GPU kernel |
| `csrc/vision/detection/ultralytics_det.h/.cpp` | YOLO 检测模型 |
| `csrc/vision/common/visualize/vis_det.cpp` | CPU 绘制实现 |

### 8.2 数据流对比

```
优化前：
  CUVID → GPU→CPU拷贝 → NV12(cpu) → CUDA kernel → BGR(gpu) → 推理(gpu)
  → BGR(cpu) → vis_det(cpu) → SWS BGR→NV12(cpu) → NVENC(gpu)

P1+P2+P3 后：
  CUVID → nv12_to_bgr_cuda(gpu) → 推理(gpu) → draw_boxes_gpu(gpu)
  → bgr_to_nv12_cuda(gpu) → cudaMemcpy → NVENC(gpu)
                                     ↑ 仅一次 GPU→CPU 拷贝
```

### 8.3 硬件参考数据

| 操作 | 硬件 | 耗时（1080p） |
|------|------|-------------|
| CUVID 解码 H.264 | GPU | ~0.5-1ms |
| GPU→CPU 拷贝 3MB | PCIe 4.0×16 | ~0.15ms |
| CPU→GPU 拷贝 3MB | PCIe 4.0×16 | ~0.15ms |
| NV12→BGR CPU (cv::cvtColor) | CPU | ~3-5ms |
| NV12→BGR GPU (kernel) | GPU | ~0.3-0.5ms |
| YOLO11n FP16 640×640 | GPU (RTX 4060 Ti) | ~2-4ms |
| vis_det 10 objects CPU | CPU | ~1-2ms |
| draw_boxes GPU 10 objects | GPU | < 0.2ms |
| BGR→NV12 CPU (sws_scale) | CPU | ~1-2ms |
| BGR→NV12 GPU (kernel) | GPU | ~0.2-0.3ms |
| NVENC H.264 编码 | GPU (8th gen) | ~0.5-1ms |