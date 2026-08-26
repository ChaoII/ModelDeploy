# AI 智能安防监控平台（application/surveillance）架构说明

本文档描述 `application/` 目录下 AI 智能安防监控平台（导出为 `surveillance` 可执行文件）
在 **application-sdk-rewrite** 重写后的最新架构。重写把原先手写的编解码/帧池/调度模块整体
下沉到 SDK 的 `csrc/video` 模块，应用侧只保留「单线程检测关键路径」与薄封装，显著减少自研
代码与维护面，同时获得 GPU 硬编解码直通、缓冲池与自动重连等 SDK 原生能力。

> 阅读前提：了解 SDK 视频模块（[视频编解码总览](./video/README.md)、[技术报告](./video/design.md)、
> [接口参考](./video/api.md)）与 `ImageData` 零拷贝模型（[预处理详解](./preprocess.md)）。

---

## 1. 整体架构

每个通道（任务）是一条独立 `Pipeline`，数据处理链路固定为：

```
SDK VideoDecoder（异步，经 VideoSource） 
   │  解码回调逐帧投递
   ▼
有界队列（满时丢最旧帧，保最新、控延迟；容量 3）
   │  det_cv_ 唤醒
   ▼
应用 detect_loop 线程（每通道 1 个，单线程关键路径）
   ├─ InferGroup.run_models：逐模型跑同一帧（每路独立模型组，无跨通道 batch）
   │     · detection 模型 → 收集 SDK DetectionResult
   │     · face/classification 等非检测模型 → InferResult
   ├─ 检测绘制：SDK UltralyticsDet::draw_result 直接在设备 NV12 帧内绘制（GPU 零拷贝）
   ├─ 非检测绘制：DrawEngine 标注（face/classification 走主机路径，见 §4）
   └─ VideoSink::encode（SDK VideoEncoder.encode_async，异步排队编码）
```

线程模型：**每通道共 2 类应用线程** —— SDK 解码线程（`VideoSource`/`VideoDecoder` 内部）+ 1 个
`detect_loop` 线程；编码走 SDK 异步（`VideoSink::encode_async`），不占应用线程。SDK 内部
解码/编码各自异步，应用无需自行管理缓冲池与线程。

关键点：

- **SDK 解码**：`VideoSource` 是对 `modeldeploy::video::VideoDecoder` 的薄封装
  （`application/video_source.hpp`），`open()` 同步完成、之后回调逐帧异步投递
  （`set_callback` + `start`）。
- **有界丢最旧队列**：`det_queue_` 容量 `det_max_size_ = 3`（`pipeline.hpp:102`），满时
  `pop_front()` 丢最旧帧，保证帧率与低延迟（背压策略，见 SDK design.md）。
- **单检测线程**：`Pipeline::detect_loop()`（`pipeline.cpp:244`）对每帧串行执行
  「推理 → 绘制 → encode_async」，是应用侧唯一关键路径；`perf_stats` 分 Infer/Draw/Encode
  度量，并将 SDK 编解码统计摄入 HTTP（`ingest_sdk`）。
- **SDK 编码**：`VideoSink` 是 `modeldeploy::video::VideoEncoder` 的薄封装
  （`application/video_sink.hpp`），`encode()` 内部走 `encode_async`，GPU 直编为 D2D 拷贝。

### 单路配置 → 线程/资源总览

| 组件 | 实现 | 线程 | 说明 |
|------|------|------|------|
| 解码 | `VideoSource` → SDK `VideoDecoder` | SDK 内部异步 | 缓冲池、自动重连、device_only 直通 |
| 有界队列 | `det_queue_`（容量 3） | — | 满则丢最旧帧 |
| 检测关键路径 | `detect_loop`（1 线程/通道） | 1 | 推理 + 绘制 + encode_async |
| 编码 | `VideoSink` → SDK `VideoEncoder` | SDK 异步 | encode_async，GPU D2D 直编 |

---

## 2. 已删除的自研编解码/调度模块

重写前应用侧有一批手写模块，现已整体删除（`git rm`），由 SDK `csrc/video` 的薄封装取代：

| 已删除模块 | 取代物 |
|-----------|--------|
| `stream_decoder.{hpp,cpp}` | SDK `VideoDecoder`（经 `video_source`） |
| `stream_encoder.{hpp,cpp}` | SDK `VideoEncoder`（经 `video_sink`） |
| `stream_hub.{hpp,cpp}` | 每通道独立 `Pipeline` + SDK 编解码 |
| `frame_pool.{hpp,cpp}` | SDK 解码缓冲池（`video_decoder` 内部） |
| `sophgo_decoder.{hpp,cpp}` | SDK 解码器（Sophgo 平台自动选择硬件路径） |
| `batch_scheduler.{hpp,cpp}` | 每路独立模型组（`InferGroup` 独立实例，不做跨通道 batch） |

对应测试 `test_stream_decoder.cpp` / `test_stream_encoder.cpp` / `test_frame_pool.cpp` 一并删除。

新增的 SDK 薄封装（`application/`）：

- `video_source.{hpp,cpp}` —— `VideoSource`：封装 `VideoDecoder` 生命周期与回调。
- `video_sink.{hpp,cpp}` —— `VideoSink`：封装 `VideoEncoder` 生命周期与 `encode_async`。
- `video_codec.{hpp,cpp}` —— `EncodeTopology`/`parse_topology`、`hw_from_string`、
  `video_codec_fill_decoder` / `video_codec_fill_encoder`：把应用 `DecoderConfig`/`EncoderConfig`
  映射为 SDK `VideoDecoderConfig`/`VideoEncoderConfig`。

### 保留未动的管理面模块

`http_server`、`pipeline_manager`、`perf_stats`、`inference_engine`、`draw_engine`、
`config`、`web_ui.html` 等本就与 SDK 深度协同，逻辑未回退；其中 `inference_engine`/`draw_engine`
本质上已是 SDK 模型/绘制的调用端，保持「不变」而非占位。

---

## 3. 配置

### 3.1 编码拓扑（EncodeTopology）

`TaskConfig.topology` 支持枚举（由 `parse_topology` 解析，`video_codec.hpp`）：

| 值 | 含义 | 状态 |
|----|------|------|
| `per_channel` | 每路独立编码推流 | **MVP 默认** |
| `mosaic` | 多路拼合一路（mosaic 编码） | Phase-2（仅保留字段，未实现） |
| `both` | 同时输出独立路 + 合流 | Phase-2（仅保留字段，未实现） |

### 3.2 编码器尺寸/帧率

`EncoderConfig`（`config.hpp`）新增输出尺寸与帧率控制：

| 字段 | 默认 | 说明 |
|------|------|------|
| `out_width` / `out_height` | 0 | 0 = 跟随源宽高；非零 = 强制输出分辨率 |
| `out_fps` | 0 | 0 = 跟随源帧率；非零 = 强制输出帧率 |
| `fps` | 0 | 帧率别名（0 = 自动匹配源帧率） |

其余编码参数沿既有：`codec`（auto/libx264/h264_nvenc）、`preset`、`tune`、`format`
（auto/rtsp/rtmp/flv/mp4）、`bitrate_kbps`、`gop`、`low_latency` 等。

### 3.3 解码器

`DecoderConfig`：`hw_accel`（cuda/none，其它字符串映射为 SDK `Auto`）、`device_only`
（true 时跳过 D2H、仅暴露设备指针）、`rtsp_transport`、`reconnect_delay_ms`、
`max_reconnects`、`timeout_us`。

---

## 4. GPU 零拷贝渲染/推理/编码链路

在 NVIDIA 平台上，`hw_accel="cuda"` + `device_only=true` + 硬编 codec 三者齐备时启用
**GPU-direct 门控**（`pipeline.cpp:78-92`），实现全程设备态、无 GPU↔主机像素往返：

```
decode（设备 NV12，device_only=true）
   │  解码帧为 GPU 显存 ImageData
   ▼
infer（InferGroup 在 ImageData.device 上零拷贝推理）
   │
   ▼
draw（SDK UltralyticsDet::draw_result 就地写入设备 NV12 帧内；非检测走 DrawEngine）
   │
   ▼
encode（GPU 直编 D2D，编码器以设备帧为输入，h264_nvenc）
```

- 检测绘制：`UltralyticsDet::draw_result(frame, dets, threshold)` 直接在设备 NV12 帧内绘制，
  不产生主机拷贝。
- 非检测（face/classification）标注：`DrawEngine` 走 CPU 路径。对 CPU NV12 帧先
  `CVT_NV122PKG_BGR` 转 BGR 标注、再重建 NV12 交付编码；对设备 NV12 帧走
  `draw_engine_->draw_gpu` 就地绘制，保住 GPU 直编 D2D（见 `draw_non_det`，`pipeline.cpp:206`）。
- 快照（HTTP JPEG）为低频操作（`snapshot_interval_=2` 帧之一），对设备帧 `toCpu` 深拷贝回读，
  不占每帧关键路径。

**退化路径**：当源为 CPU 软解（`hw_accel="none"`）或设备不支持 GPU 直通时，自动退化为
「CPU 解码 → 主机推理/绘制 → 软编（libx264）」；Sophgo 平台走硬件加速的 BMCV 搬运/推理，
由 SDK 按平台选择，无需应用改动。

> 应用侧不直接持有 CUDA 设备帧的像素所有权 —— SDK 解码器内部的缓冲池负责回收复用。

---

## 5. 各平台配置示例

以下为 `TaskConfig` 关键字段的 JSON 片段（完整结构见 `config.cpp` 的序列化）。

### 5.1 NVIDIA 桌面 / Jetson（GPU 直通）

```jsonc
{
  "decoder": {
    "hw_accel": "cuda",        // CUDA 硬解
    "device_only": true,       // 跳过 D2H，仅暴露设备指针（GPU 直通）
    "rtsp_transport": "tcp"
  },
  "encoder": {
    "codec": "h264_nvenc",     // Jetson 亦可用 "auto" / "nvh264enc"
    "format": "flv",
    "bitrate_kbps": 2500,
    "gop": 12
  }
}
```

### 5.2 CPU（全软编软解）

```jsonc
{
  "decoder": {
    "hw_accel": "none",        // 软解
    "device_only": false
  },
  "encoder": {
    "codec": "libx264",
    "format": "flv",
    "preset": "ultrafast",
    "tune": "zerolatency",
    "bitrate_kbps": 2500
  }
}
```

### 5.3 Sophgo（算能 TPU）

```jsonc
{
  "decoder": {
    "hw_accel": "sophgo",      // 映射为 SDK Auto，由平台选择硬件加速路径
    "device_only": false
  },
  "encoder": {
    "codec": "libx264",        // TPU 侧建议软编，避免额外硬件编码依赖
    "format": "flv",
    "bitrate_kbps": 2500
  }
}
```

> **Sophgo 分类模型注意**：Sophgo int8 bmodel 多属 batch=1 静态形状，分类模型在 SDK 侧需
> `set_cls_batch_size(1)`（AGENTS 约定），否则静态 batch 不匹配会推理失败。

---

## 6. 构建

应用由顶层 CMake 的 `BUILD_SURVEILLANCE=ON` 激活（`application/CMakeLists.txt`，需要
FFmpeg，默认 `E:/develop/ffmpeg`）。

```bash
cmake -S . -B build -G Ninja \
  -DBUILD_SURVEILLANCE=ON \
  -DBUILD_SURVEILLANCE_TESTS=ON \
  -DWITH_GPU=ON        # NVIDIA 平台；CPU/Sophgo 可 OFF/按需
cmake --build build --target surveillance --parallel     # 应用
cmake --build build --target surveillance_test --parallel # 测试
```

依赖关系：

- `BUILD_SURVEILLANCE=ON` 时，**根 CMake 自动强制 `BUILD_VIDEO=ON`**（SDK 视频模块为应用
  编解码所必需），无需手动打开。
- `BUILD_SURVEILLANCE_TESTS=ON` 生成 `surveillance_test`（Catch2，单二进制）。

测试：

```bash
cd build && bin\surveillance_test.exe          # 全量
cd build && bin\surveillance_test.exe [pipeline]  # 仅流水线
```

---

## 7. 8 路本地文件冒烟

`application/tools/smoke_8ch.ps1` 提供 8 通道本地文件集成冒烟：启动 `surveillance.exe`、
经 HTTP REST 逐路 `start`、观测约 5s 后断言每路 `sdk_frames_out` 达到工作阈值。

```powershell
pwsh application/tools/smoke_8ch.ps1
```

要点：

- 输入为 `test_data/bench_videos/cam00..cam07.mp4`（缺失时回退复用上一路）。
- 每路任务挂 1 个 dummy 模型（不存在路径，加载失败 → 进入 "preview only" 原帧编码路径），
  以满足 `TaskConfig::validate()` 的 models 非空约束。
- 默认脚本解码侧 `hw_accel="cuda"`（GPU 冒烟默认）；**CPU 冒烟**可将
  `decoder.hw_accel` 改为 `"none"`、`encoder.codec` 保持 `"libx264"`，即全软编软解验证。
- 断言：`frames_out ≥ 25 × SampleSec × 0.6`；脚本退出即清理进程与临时目录。
