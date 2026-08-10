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

## P4 零拷贝 / 异步推理实验（2026-08-10，均回退）

| 实验 | 结果 | 结论 |
|------|------|------|
| SDK batch 设备 BGR 输入检测 | 提交保留（b158dc4）| host 输入无副作用，为后续增强能力 |
| process_batch 设备零拷贝（nv12→GPU BGR→batch 设备输入）| fps 6.8 vs host 11 | D2D 拷贝 + 系统内存压力抵消 PCIe 节省，收益为负 |
| 异步批量推理（process 不阻塞等结果）| fps 3-7 或统计失真 | req 自持拷贝开销 + batch 队列积压，收益为负 |

结论：当前环境（RTX 4060 Ti + 32GB + 内存压力 FreeGB≈5GB）下，同步 host BGR 批处理 + stream 复用是最优实现，20 路稳定 7-11fps/路。

## 320 分辨率实验（2026-08-10）

转换 320×320 动态 engine（yolo11n_nms_b8_320.engine）：单帧 2.5ms(398fps)，batch4 2.69ms/批(1488fps)。

| 路数 | 每路 fps | 总吞吐 | vs 640 |
|------|---------|--------|--------|
| 10 路 @320 | 29.7 | 297fps | +6% |
| 20 路 @320 | 13.4 | 267fps | **+50%**（640 为 7-11） |

结论：320 分辨率显著提升 20 路吞吐（+50%），GPU 53% 仍有空间。
batch4 仍优于 batch8（320：1488 vs 611 fps）。配置：max_batch=4 + 320 engine + stream 复用。

## NV12→BGR 批处理实验（2026-08-10，回退）

新增 SDK nv12_to_bgr_batch_cuda（一次上传/一次 kernel/一次下载/一次同步，消除逐帧 cudaStreamSynchronize），process_batch 用它。实测 process_ms 14.7→31ms（fps 13.5→6.6），收益为负：多一次 bgr_flat→逐帧 memcpy + 3D 网格 kernel 效率抵消 sync 节省。已回退，逐帧 + stream 复用仍最优。
