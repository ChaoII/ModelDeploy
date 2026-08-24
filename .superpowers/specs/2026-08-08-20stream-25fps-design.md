# 20 路 25fps 性能优化（子项目 A）— 设计文档

日期：2026-08-08
分支：main

## 背景与目标

NEXUS 多路智能监控应用（`application/`）目标是 20 路 RTSP 实时流每路 25 FPS。当前实测仅支持 ~6-8 路。`docs/optimization_prd.md`（v1.0）是权威规格，定义了 4 阶段优化方案。本设计按 PRD 分阶段完整实施。

**硬件目标**：RTX 4060 Ti 16GB / i7-12700 / 32GB RAM
**模型**：yolo11n_nms（TRT engine，已生成于 `test_data/test_models/trt/yolo11n_nms.engine`）
**测试**：本地视频文件模拟 20 路（file 源，StreamHub 只共享网络源）

### 验收标准（PRD §6）
| 指标 | 当前 | 目标 |
|------|------|------|
| 最大路数 | ~6-8 | **20** |
| 每路 FPS | ~25 | **25** |
| CPU 利用率 | 80-100% | **<50%**（Phase1）/<40%（Phase2+3） |
| GPU 利用率 | 30-40% | 60-80% |
| 端到端延迟 | 100-150ms | <200ms |
| VRAM @20路 | ~5GB | <14GB |

### 非功能要求（PRD §5）
- 所有优化默认关闭（`enable_preview`、`use_gpu_draw` 开关），CPU fallback 保留
- GPU kernel 用 `#ifdef WITH_GPU` 保护
- GPU 绘制失败 → 日志 + CPU fallback
- CUDA OOM → 降低 batch 大小
- NVENC 超 8 路 → x264 fallback 或静默跳过编码

## 现状分析（探索结论）

架构已就位：3 线程 pipeline（decode/process/encode）、StreamHub 共享解码器、clone 模型、FP16/TRT、有界队列。**PRD Phase 1 大部分已实现**：

已实现：`enable_preview` 跳过编码/绘制、per-model `interval`、TRT engine 自动检测、`batch_scheduler` 真批推理逻辑、CUVID 设备指针暴露、`predict_nv12` GPU 直接预处理、`PendingFrame` 设备指针字段。

**剩余核心工作**（本设计范围）：

| # | 工作 | 位置 | PRD |
|---|------|------|-----|
| 1 | 接线 BatchScheduler（2 处 nullptr → &batch_scheduler_，main 里 start） | pipeline_manager.cpp:171,459 + main.cpp | §3.1.3 |
| 2 | 传设备指针（GPU-direct 分支激活） | pipeline.cpp:349 | §3.4 |
| 3 | GPU 绘制内核 `draw_boxes_gpu`（全新） | csrc/vision/common/processors/draw_gpu.cuh/.cu | §3.2 |
| 4 | GPU BGR→NV12 内核 `bgr_to_nv12_cuda`（全新） | csrc/vision/common/processors/bgr_to_nv12.cuh/.cu | §3.3 |
| 5 | P4 全零拷贝（D2D 设备缓冲 + GPU BGR 批量） | pipeline.cpp + batch_scheduler.cpp | §3.4 |

## 设计

### P1：接线 BatchScheduler（接通已实现的批量推理）

**改动**：
- `pipeline_manager.cpp:171,459`：`nullptr` → `&batch_scheduler_`
- `main.cpp`：`load_from_directory` 后调用 `start_batch_scheduler()`（先确保 model_library_ 已注册）
- 修复 `batch_scheduler.cpp:110-123` 的 buffer 别名 bug（`from_raw(copy=false)` + 复用 `bgr_buf_` → 改为每次拷贝或双缓冲，防止下一批覆盖）
- 修复 `pipeline.cpp:339-341` busy-wait（`while(!ready) yield()` → CV 等待或超时轮询降频）

**验证**：20 路本地文件任务，`batch_size=4`，GPU 利用率上升，推理吞吐提升。

### P2：GPU 绘制内核 `draw_boxes_gpu`

**新文件**：
- `csrc/vision/common/processors/draw_gpu.cuh` / `draw_gpu.cu`

**接口**：
```cpp
struct GpuDrawBox {
    int x1, y1, x2, y2;
    float score;
    int label_id;
    uint8_t r, g, b;
    char label[32];
};
MODELDEPLOY_CXX_EXPORT bool draw_boxes_gpu(
    uint8_t* bgr, int width, int height,
    const GpuDrawBox* d_boxes, int num_boxes,
    float alpha = 0.15f, cudaStream_t stream = nullptr);
```

**实现**：
- kernel：1 block 每 box，block 256 线程 = 128 fill + 128 border/text
- 内嵌 8x16 ASCII 位图字体（无 FreeType 依赖）
- host wrapper 复用 `nv12_to_bgr.cu` 模式（`thread_local` workspace、`cudaPointerGetAttributes`、caller stream）

**应用层**：
- `DrawEngine::draw_gpu(...)`（InferResult → GpuDrawBox 转换）
- `pipeline.cpp`：`if (cfg_.device=="gpu" && use_gpu_draw)` → `draw_gpu`，else 现有 CPU draw

### P3：GPU BGR→NV12 内核 `bgr_to_nv12_cuda`

**新文件**：
- `csrc/vision/common/processors/bgr_to_nv12.cuh` / `.cu`

**接口**：
```cpp
MODELDEPLOY_CXX_EXPORT bool bgr_to_nv12_cuda(
    const uint8_t* bgr, int width, int height,
    uint8_t* nv12, cudaStream_t stream = nullptr);
```
输出 H×W×3/2 字节，**BT.709 limited range**。

**实现**：
- `kernel_bgr_to_nv12_y`（每像素，`Y=(66R+129G+25B+128)>>8+16`）
- `kernel_bgr_to_nv12_uv`（2×2 平均，UV 在 `width*height` 偏移，交错 `(y*width+x)*2`）

**应用层**：
- `StreamEncoder::encode_from_gpu(const uint8_t* gpu_bgr, int w, int h)`：分配 GPU NV12 → `bgr_to_nv12_cuda` → cudaMemcpy → `avcodec_send_frame` → `av_interleaved_write_frame`

### P4：全 GPU 零拷贝

**改动**：
- `stream_decoder.cpp`：新增 `read_one_frame_gpu(DecodedFrameGpu*)`，CUVID 帧留在 GPU（`av_hwframe_transfer_data` 替换为 `cudaMemcpy` D2D 到固定设备缓冲）
- `PendingFrame::gpu_nv12` 填充（D2D 拷贝保证设备指针 pipeline 安全，避免 `read_hw_frame_` 复用问题）
- `batch_scheduler.cpp`：GPU-resident BGR 批量（`yolo_preproc.cu:450` 的 H2D 需改为设备指针路径）
- 完整路径：`CUVID → nv12_to_bgr_cuda(GPU BGR) → batch_predict(GPU→GPU) → draw_boxes_gpu(in-place) → bgr_to_nv12_cuda(GPU NV12) → cudaMemcpy → NVENC`

**依赖**：P4 依赖 P2+P3 的 kernel。

## 组件改动清单

| 文件 | 改动 |
|------|------|
| `application/pipeline_manager.cpp` | 2 处 BatchScheduler 接线 |
| `application/main.cpp` | start_batch_scheduler 调用 |
| `application/batch_scheduler.cpp` | buffer 别名修复 + busy-wait 修复 |
| `application/pipeline.cpp` | 设备指针传递 + GPU draw 分支 + GPU encode 分支 |
| `application/infer_group.cpp` | GPU-direct 分支实际激活 |
| `application/draw_engine.hpp/.cpp` | `draw_gpu` 方法 |
| `application/stream_encoder.hpp/.cpp` | `encode_from_gpu` 方法 |
| `application/config.hpp` | `use_gpu_draw` 开关 |
| `csrc/vision/common/processors/draw_gpu.cuh/.cu` | 新 GPU 绘制 kernel |
| `csrc/vision/common/processors/bgr_to_nv12.cuh/.cu` | 新 GPU BGR→NV12 kernel |
| `application/perf_stats.hpp/.cpp` | 新增 `gpu_draw_us`/`gpu_nv12_us` 等指标 |
| `application/http_server.cpp` | web_ui 路径缓存 + 指标暴露 |

## 构建与验证

```bash
# 构建（开启 surveillance + TRT + GPU）
cmake -S . -B build -DBUILD_SURVEILLANCE=ON -DENABLE_ORT=ON -DENABLE_TRT=ON -DWITH_GPU=ON -DBUILD_TESTS=ON
cmake --build build --parallel
```

**验证流程**：
1. 用本地视频文件（test_data/test_video60.mp4）创建 20 个任务（REST 或 tasks.json 种子）
2. 每任务 `input_url = E:/videos/camXX.mp4`（file 源，`block_on_in_full_`），模型 `yolo11n_nms.engine`（TRT）
3. 监控 `/api/v1/metrics`：20 路、每路 25fps、CPU%<50%、VRAM<14GB
4. 每个阶段（P1→P4）单独验证吞吐提升

## 错误处理

- GPU kernel 初始化失败 → 日志 + CPU fallback（`use_gpu_draw` 开关）
- CUDA OOM → 捕获 + 降 batch_size
- NVENC >8 路 → x264 fallback
- BatchScheduler 不可用 → pipeline 回退 `infer_group_->run_models()` 单帧路径

## 风险

- P2/P3 新 CUDA kernel 是本任务最大工作量（PRD 估 3-5d + 2-3d）
- P4 的 GPU 批量 BGR 需改 `yolo_preproc.cu` 的 H2D 路径（batch 目前 host-BGR only）
- 20 路各自解码器（file 源不共享）→ decode 线程多，但 file 解码快
- NVENC 会话数限制（消费卡 8 路）——超限需 x264 fallback
