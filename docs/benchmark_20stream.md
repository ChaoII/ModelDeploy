# 20 路视频分析吞吐基准

日期：2026-08-10（更新）
硬件：RTX 4060 Ti 16GB / i7-12700 / 32GB RAM
模型：yolo11n_nms（TRT FP16，b8 动态 shape engine）

## TRT Engine 基准（SDK 级）

| Engine | 配置 | 单帧 | batch4 | batch8 |
|--------|------|------|--------|--------|
| yolo11n_nms.engine | 固定 1x3x640 | 11.5ms (87fps) | 15.3ms/批 (262fps) | — |
| yolo11n_nms_b8.engine | 动态 1-8x3x640 | 3.4ms (293fps) | 3.3ms/批 (1196fps) | 6.9ms/批 (580fps) |

**关键**：动态 shape engine 单帧快 3.4×，batch4 吞吐最高。

## 吞吐实测（本地文件模拟，enable_preview=false）

### 优化历程

| 配置 | 20 路每路 fps | 总吞吐 | 说明 |
|------|---------|--------|------|
| b4 engine + 单帧路径 | 9.7 | 194fps | BatchScheduler 未聚合 |
| b8 + batch8 | 12.1 | 242fps | max_batch=8 |
| b8 + batch4 | 13.0 | 260fps | 最优 batch 配置 |
| **+ need_bgr 省拷贝 + stream 复用** | **9.3** | **186fps** | process_batch 15.4ms（受 20 路系统内存限制） |

### 路数扩展性（干净环境）

| 路数 | 每路 fps | 总吞吐 | 说明 |
|------|---------|--------|------|
| 1 | 26.2 | 26 | 单路达标 |
| 2 | 23.3 | 47 | |
| 5 | 32.4 | 162 | 每路超 25 |
| 10 | **28.1** | **225fps** | **每路达标 25+** |
| 20 | 9.3 | 186fps | **受系统内存限制**（32GB 不够） |

### 资源占用（20 路）

- GPU：53% 利用率，VRAM 13.3GB/16GB
- 系统内存：surveillance Private 14.4GB（+其他进程 ≈ 32GB 满）
- process_batch 平均 15.4ms/批（batch_avg 3.06）

## 结论

- **10 路稳定 28fps/路（225fps 总吞吐），每路超 PRD 的 25fps 目标**
- **20 路受系统内存（32GB）物理限制**：surveillance 14.4GB + GPU 13.3GB + 其他进程导致 swap，fps 降 9.3
- 要 20 路全 25fps 需：64GB 系统内存，或降低每路内存占用（GPU workspace 复用、锁页内存、decode 限速）
- 用户决策：接受物理极限，最大化吞吐（已从 6-8 路扩展到 10 路达标 + 20 路可跑）

## 已实施优化（commit 87de734, a116b37）

1. BatchRequest.need_bgr：非预览路不传回 BGR 结果（省深拷贝）
2. process_batch 每帧独立 BGR 缓冲 + copy=false（消除跨批复用 + 深拷贝）
3. BatchScheduler 复用 CUDA 流（避免每批 create/destroy）
4. metrics 暴露 batch_avg_size / batch_avg_process_ms 监控

## 剩余优化空间（未做）

1. **锁页内存**：nv12_buf_ 改 cudaMallocHost（H2D 加速）
2. **process 轮询改 CV 通知**（省 CPU）
3. **decode 限速匹配处理率**（file 源 decode 0.8ms 超快，浪费 CPU）
4. **P4 batch GPU 零拷贝**：设备 NV12 → GPU BGR → batch_predict 设备输入（消除双重 PCIe）
5. **减少每路内存**：GPU workspace 池化（FramePool 未用）
