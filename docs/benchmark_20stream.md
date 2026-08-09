# 20 路视频分析吞吐基准

日期：2026-08-09
硬件：RTX 4060 Ti 16GB / i7-12700 / 32GB RAM
模型：yolo11n_nms（TRT FP16）

## TRT Engine 基准（SDK 级）

| Engine | 配置 | 单帧 | batch4 | batch8 |
|--------|------|------|--------|--------|
| yolo11n_nms.engine | 固定 1x3x640 | 11.5ms (87fps) | 15.3ms/批 (262fps) | — |
| yolo11n_nms_b8.engine | 动态 1-8x3x640 | 3.4ms (293fps) | 3.3ms/批 (1196fps) | 6.9ms/批 (580fps) |

**关键**：动态 shape engine（min=1, opt=4, max=8）的单帧远快于固定 1x3x640（3.4ms vs 11.5ms），且 batch4 吞吐最高（1196fps）。

## 20 路实测（本地文件模拟，enable_preview=false）

| 配置 | 每路 fps | 总吞吐 | 说明 |
|------|---------|--------|------|
| b4 engine + 单帧路径 | 9.7 | 194fps | BatchScheduler 未聚合（首帧即处理 bug 已修） |
| b8 engine + batch8 | 12.1 | 242fps | max_batch=8, timeout=8ms |
| **b8 engine + batch4（最优）** | **13.0** | **260fps** | max_batch=4, timeout=4ms |

运行时资源：GPU 56%，VRAM 13GB/16GB，CPU 64%，105 线程。

## 结论

- **单卡 RTX 4060 Ti 无法达到 20 路 × 25fps = 500fps 全帧推理**（物理极限受 batch 聚合调度 + CPU 开销限制，实测 260fps）
- 当前配置较 PRD 基线（~6-8 路）扩展到 **20 路，每路 13fps**
- 用户决策：接受物理极限，最大化吞吐

## 剩余优化空间（未做，留后续）

1. **process 轮询改 CV 通知**（pipeline.cpp:397-399 的 200µs sleep → condition_variable）省 CPU，可能提升 GPU 利用
2. **decode 限速匹配处理率**（file 源 decode 0.8ms/帧超快，25fps 输入只处理 13fps，decode 浪费 CPU）
3. **P4 batch GPU 零拷贝**：process_batch 用设备 NV12 → GPU BGR → batch_predict 设备输入（当前 host NV12 双重 PCIe）
4. **NVENC 会话数**：>8 路预览需 x264 fallback（当前 enable_preview=false 不涉及）
