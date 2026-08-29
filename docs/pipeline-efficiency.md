# 推理视频管线跨平台效率最佳实践

> 适用范围：ModelDeploy 的「解码 → 前处理 → 推理 → 后处理 → 绘制 → 编码」端到端视频管线，
> 面向三种主流部署环境——**主机 CUDA（PC）**、**Jetson L4T（NVIDIA 嵌入式）**、**Sophgo 算能 TPU**。
> 本文基于本仓库在三种环境下的实测与代码路径归纳，重点回答：**哪个环节是瓶颈、怎么做才最高效、如何规避**。

---

## 1. 管线解剖：每一帧到底做了什么

所有平台共享同一条数据流，瓶颈几乎都出现在「**内存拷贝 / 主机↔设备边界**」上，而非推理本身：

```
                ┌──────────────────────── [设备侧] ────────────────────────┐
 视频源 ──► 解码 ──► 帧交付 ──► 前处理 ──► 推理 ──► 后处理 ──► 绘制 ──► 编码 ──► 输出
           (硬件/软)            (缩放/归一/通道)   (模型)    (nms/解码) (叠加)   (硬件/软)
```

| 阶段 | 典型做法 | 主要成本 | 关键决策点 |
|---|---|---|---|
| 解码 | 硬解（nvdec/bmdec/nvv4l2）或软解 | 高性能任务→VPU | **输出帧放在哪一侧（设备/主机）** |
| 帧交付 | read_one_frame / 回调 | 拷贝/映射 | **是否是“宿主映射的设备缓冲”** ⚠️ |
| 前处理 | CUDA kernel / BMCV / CPU | 访存 | 是否在推理设备上做、输入源是否连续堆 |
| 推理 | TRT / ORT-CUDA / TPU(bmrt) | 模型计算 | 精度（FP32/FP16/INT8）、batch 静态性 |
| 后处理 | CPU/CUDA | 稀疏、通常很小 | 类数取 shape，NMS 是否在模型内 |
| 绘制 | 设备就地写 NV12 | 访存 | 是否零拷贝写回设备帧 |
| 编码 | 硬编（nvenc/bmh264enc/nvv4l2）或软编 | VPU | 输入是否设备帧、输入帧是否“连续紧凑” |

> **最核心、最容易被忽略的一条经验**（本项目踩过的最大坑）：
> **“宿主 (host) 映射的设备缓冲”被下游逐步、逐元素访问时极慢**，慢到足以抵消全部加速收益。
> 典型案例：Sophgo bmdec 的 **GBM/DMA-BUF 宿主映射帧**被 BMCV 逐次访问 → 前处理 **66ms/帧**；
> 而拷成一句紧致连续堆缓冲（~1-2ms）后立即降到 **8.4ms**。NVIDIA 的 nvbuffer / NVMM 属同类，务必警惕。

---

## 2. 跨平台通用准则（先读这个）

无论哪个平台，按优先级踩这 7 条：

1. **把数据留在它该在的那一侧。** 设备帧就一路设备侧（前处理/推理/绘制/编码全用设备指针），
   不要让任何一环把它拷回主机（除非模型本身是 CPU 的）。
2. **识别并消灭“宿主映射的设备缓冲”。** 设备驱动输出这类内存时（GBM/DMA-BUF、NVMM、部分 CUDA
   interop），拷成**紧致连续堆缓冲再交给下游**（尤其是会被逐像素/反复访问的前处理和绘制）。
   单次拷贝 ~1-2ms 的代价 < 反复访问宿主映射的几十 ms。
3. **推理模型尽量量化。** 设备推理的 I/O 精度是平方级杠杆：Sophgo 上同一个检测模型，
   **FP32 bmodel 纯 TPU 16.8ms，INT8 只要 3.36ms**（≈5 倍）。凡是带宽/算力受限的设备优先 INT8/FP16。
4. **烘焙后端与引擎，避免在线编译。** TensorRT 用 `trtexec` 离线生成 `.engine` 并缓存，
   在线从 ONNX 构建要数分钟；SOPHGO 直接用编译好的 bmodel。
5. **解码器的“节拍毫秒”不等于“解码耗时”。** GStreamer `appsink` 常按源帧率同步放行
   （如 25fps → 39.9ms/帧），这是**节奏不是成本**；别把它在统计里当解码瓶颈。
6. **善用异步/背压，但读懂是成本还是排队。** 异步编码队列、丢帧背压能平滑吞吐；
   单独文件任务要显式触发 EOS/`stop` 才能刷出 mp4 的 moov（否则输出缺索引）。
7. **构建/运行时一次性配置要对。** Sophgo 必须设 `GST_PLUGIN_PATH=/opt/sophon/sophon-gstreamer_2.2.0/lib`、
   纯 Sophgo 构建 `ENABLE_SOPHGO=ON + ENABLE_ORT=OFF`；Jetson 需 `ENABLE_TRT=ON + TBB`。

---

## 3. 各平台分述

### 3.1 主机 CUDA（PC，独立 GPU）

**目标形态：全设备、零拷贝。**

```
nvdec (GStreamer nvh264dec, GST_MAP_CUDA) ──► 设备 NV12
        │（帧留在 GPU，device 指针）
        ▼
CUDA 前处理 kernel（yolo_preproc.cu，零拷贝吃设备平面）──► GPU FP32 tensor（内存池）
        ▼
ORT-CUDA / TRT 推理（CUDA memory 零拷贝绑定）──► 结果
        ▼
CUDA 后处理 + CUDA 就地绘制（写回设备 NV12）
        ▼
nvenc (nvh264enc，从 CUDA memory 直接编码)
```

**关键机制**：`CudaProcessorBackend::yolo_preprocess_nv12` 用 `cudaPointerGetAttributes`
识别设备指针后**零拷贝**消费；ORT CUDA EP 以 CUDA MemoryInfo 零拷贝绑定输入张量；`device_only`
解码直通让整条链没有一次主机回读。

**瓶颈与规避：**

| 瓶颈 | 如何避免 |
|---|---|
| **H2D / D2H 拷贝**（最常见的隐形杀手） | 反编译一律 `GST_MAP_CUDA` 设备直通；模型输入用设备指针；不调任何 host 回读 |
| TRT 在线构建（分钟级） | 用 `trtexec` 离线生成 `.engine`，启动时直接加载；缓存 |
| FP32 I/O 浪费显存带宽 | 支持时用 FP16/INT8 engine |
| 主机映射的设备缓冲被 CPU 访问 | 同第 2 节：拷成连续堆或保持设备侧 |
| 一旦混入 CPU 模型 | 那一支就地转 host NV12；**不要**为单个 CPU 模型把整条 GPU 链拉回主机 |

**⚠️ 进程级陷阱（GStreamer–CUDA 互操）**：GStreamer 首次创建 CUDA context
（`gst_cuda_context_new` / CUDA memory）后，其 nvcodec 会朝进程内**余下所有** `nvh264enc`
管道全局注册 CUDA 缓冲——此进程级状态**不可逆**，后续管道会缺 moov、软解打不开。
**必须把带 `[gst-cuda]` 的用例放进单独进程**，不与其它 nvcodec 管道混跑。

**实测锚点**：RTX 40 系（SM 86），默认 CUDA 架构已设 86；设备 NV12 → CUDA kernel → ORT-CUDA
全 `predict()` 零拷贝链路在仓库 `[gpu]` 套件通过（辅助断言 IoU>0.5 校验正确性）。

---

### 3.2 Jetson L4T（NVIDIA 嵌入式集成 GPU）

**目标形态：解码/编码上硬件，前/推/后留在集成 GPU。**

```
nvv4l2decoder（VPU 硬解）──► nvvidconv ──► 主机 NV12
        │（⚠️ L4T 无 CUDA 零拷贝：NvBufSurface 提取依赖私有 nvmm buffer-pool，本 SDK 不含，
        │   故 convert 为主机 NV12；decode 仍是硬件加速）
        ▼
H2D 上传该帧 ──► CUDA 前处理（GPU）──► TRT GPU 推理 ──► CUDA 绘制
        ▼
nvv4l2enc（VPU 硬编）
```

**瓶颈与规避：**

| 瓶颈 | 如何避免 |
|---|---|
| **解码→主机的固定边界**（L4T 无法避免的 host NV12） | 接受这一处 host↔device；其余全部留在 GPU。不要因此退化成纯 CPU 管线 |
| H2D 上传成为每帧固定开销 | 用连续 NV12 + 包式上传；这是 L4T 的固有成本，硬解/硬编已经给你省了 VPU 上的大钱 |
| TRT 引擎 | 离线 `trtexec` 生成并缓存 |
| 集成 GPU 算力有限 | 用 INT8/FP16 engine；避免每次都 `create_processor_backend`（建流/分配池是 CUDA 同步操作），**复用 processor backend**（本仓库 `DrawEngine::backends_` 已按 device 缓存复用） |

**注意**：Jetson 自动检测（`/etc/nv_tegra_release`），强制 `WITH_GPU=ON`、`ENABLE_TRT=ON`，需要 TBB。
GStreamer–CUDA 互操同样要**独立进程**隔离（同上）。

---

### 3.3 Sophgo 算能 TPU

**目标形态：硬解/硬编上 VPU，前处理/推理在 TPU，帧交付后先做一次堆拷贝。**

```
bmdec（VPU 硬解）──► GBM/DMA-BUF 宿主映射 NV12 ⚠️
        │（❌ 直接交给 BMCV：前处理 66ms/帧）
        ▼  read_one_frame 无条件拷成紧致连续堆 NV12（~1-2ms）
BMCV 前处理（NV12 letterbox+归一 → TPU INT8 tensor，设备到设备，快）──► ~8.4ms
        ▼
TPU bmrt 推理（INT8 bmodel）──► 3.36ms 纯 TPU
        ▼
CPU 后处理（nms/解码, 0.2ms，稀疏）
        ▼
TPU/BMCV 就地绘制 ──► bmh264enc（VPU 硬编, appsrc 喂入）
```

**这是三种环境里“瓶颈最容易爆”的一种**，核心就那么几件事：

| 瓶颈 | 如何避免（本仓库实测方案） |
|---|---|
| **GBM/DMA-BUF 宿主映射帧被 BMCV 逐像素访问 → 前处理 66ms/帧**（本项目第一大坑） | `GstDecoder::read_one_frame` 对**标准主机 NV12 无条件**拷入紧致连续堆缓冲（去门控），GBM 帧 66→**8.4ms**；软解 CPU 帧另多 ~1-2ms 可忽略 |
| **FP32 模型浪费 TPU**（纯 TPU 16.8ms vs int8 3.36ms） | 用 **INT8 bmodel**（≈5 倍）；本仓库检测模型 input_dtype 为 FP32 但内部 int8，output `[1 5 8400]`（4+1 单类）可直接支持 |
| 软解软编吃掉 CPU | 用原生硬编（`bmh264enc` 实测 86.9fps）+ 硬解（`bmdec` 1500/1500 实时）。须设 `GST_PLUGIN_PATH` 且 SDK `-DENABLE_GSTREAMER=ON` |
| 显式 SOPHGO 分支没被走到 | 即便后端收到 `hw_accel=Auto`，soft `decodebin` 在 Sophgo 会自动选中 `bmdec` 产 GBM 帧——堆拷贝去门控后**这条路线同样受益**；显式 `sophgo` 分支仍可用于 fail-closed |
| 分辨 89.8ms 的“推理耗时” | 那是 FP32 模型 + CPU 后处理 + GBM 前处理混合；换 int8 + 堆拷贝后 run_models 70→12.4ms |
| bmodel 静态 batch=1 | 分类模型需 `set_cls_batch_size(1)`，否则静态形状不匹配推理失败（AGENTS 约定） |
| mp4 输出缺 moov | 单文件任务跑完要 `POST .../stop` 触发 encoder EOS 刷 moov |
| 编码喂入慢 | `encode(const VideoFrame&)` 按 `ImageData::device()` 路由（CPU 软编/TPU 占位 fail-closed）；喂入紧凑连续帧后编码 45.8→1.24ms |

**实测全链路数字（1080p30 源，zhgd INT8，`enable_preview=true`）：**

| 指标 | 改造前 | 改造后 |
|---|---|---|
| SDK 前处理 | 66ms（GBM） | **8.4ms**（堆拷贝） |
| run_models（推理组） | 70.4ms | **12.4ms** |
| 编码 | 45.8ms | **1.24ms** |
| 绘制 | 46ms | **3.5ms** |
| **fps** | 7.8 | **≈25（实时）** |
| 端到端 avg_total | 128ms | **17.1ms** |
| SDK 丢帧 | 0 | 0 |

> 解码 39.9ms 是 appsink 按 25fps 的**节拍**，不是解码成本；瓶颈全都被堆拷贝 + int8 + 硬编解消掉了。

---

## 4. 三平台横向对比

| 维度 | 主机 CUDA | Jetson L4T | Sophgo TPU |
|---|---|---|---|
| 推理设备 | 独立 GPU | 集成 GPU | TPU(bmrt) |
| 解码 | nvdec（CUDA 零拷贝直通） | nvmm HW（host NV12，无零拷贝） | bmdec VPU（GBM 宿主映射 ⚠️） |
| 前处理 | CUDA kernel（零拷贝） | CUDA（需 H2D） | BMCV（需先堆拷贝） |
| 编码 | nvenc（CUDA 直通） | nvv4l2enc（HW） | bmh264enc（HW, appsrc） |
| 模型精度 | FP16/INT8 可选 | INT8/FP16 重要 | **INT8 必需**（5× 差距） |
| 头号瓶颈 | 不必要的 H2D/D2H、TRT 在线构建 | 解码 host NV12 固定边界 | **GBM 帧 + FP32 模型** |
| 首要规避手段 | 全设备零拷贝 + 预烘焙 engine | 其余环节留 GPU + 复用后端 | **堆拷贝 + int8 bmodel + 硬编解** |
| 实测结果 | `[gpu]` 套件零拷贝链路通过 | HW 解/编，GPU 前/推/后 | **fps≈25 实时**，run_models 12.4ms |
| 进程隔离 | GStreamer–CUDA 互操独立进程 | 同左 | 无 CUDA，无该问题 |

---

## 5. 通用检查清单（部署新平台/新模型时逐条过）

- [ ] 解码输出帧在设备侧时，是否任其被 CPU/逐步访问？（→ 应拷成连续堆或保持设备侧）
- [ ] 模型 I/O 是否量化到设备支持的最优精度（INT8/FP16）？有无 FP32 浪费？
- [ ] TRT 是否离线生成了 `.engine` 并缓存？（禁止在线构建进上线路径）
- [ ] 前处理/绘制是否复用 processor backend（不每帧 `create_processor_backend`）？
- [ ] 解码“节拍毫秒”是否被误当成解码耗时计入统计/瓶颈？
- [ ] 编码输入是不是连续紧凑帧？进的是硬编而非退化软编？
- [ ] 输出容器（mp4）是否保证在任务结束触发 EOS 刷出索引（moov）？
- [ ] GStreamer–CUDA 互操用例是否在独立进程运行？
- [ ] 运行时一次性配置（`GST_PLUGIN_PATH`、`LD_LIBRARY_PATH`、后端 ON/OFF）是否正确？

---

## 6. 一句话总结

> **别让“宿主映射的设备缓冲”被逐渐访问，别用 FP32 跑设备推理，把解/编的体力活交给 VPU、
> 把它俩之间的一切留在设备侧并复用后端——三平台都通用的最高效管线，就是把“不必要的拷贝和宿主访问”彻底消干净。**
