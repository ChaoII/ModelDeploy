# 视频编解码 · 技术报告

> 面向想**理解设计与实现**的读者。纯使用向请先看 [guide.md](./guide.md)，接口签名看 [api.md](./api.md)。
> 本报告基于 `csrc/video/` 当前实现撰写（VideoFrame 统一帧模型，C++/Python 门面）。

## 1. 总体架构：四层解耦

视频模块没有把 FFmpeg/GStreamer 揉进一坨，而是做了**四层**分层，每一层的职责单一、不泄漏下层类型：

```
┌─────────────────────────────────────────────────────────────┐
│  门面层  VideoDecoder / VideoEncoder (C++，用户直接调用)        │
│   - 后端无关；接口签名里没有任何 FFmpeg/GStreamer 原生类型       │
└────────────────────────────┬────────────────────────────────┘
                             │ 持有/包装
┌────────────────────────────▼────────────────────────────────┐
│  路由层（解码）DecodePipeline                                │
│   - 统一三类工程化能力：有界背压队列 + 帧缓冲池 + 重连状态机     │
│   - 只认 DecoderBackend 抽象，不 care 底层是谁                │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  后端抽象层  DecoderBackend / EncoderBackend（纯虚接口）       │
└────────────────────────────┬────────────────────────────────┘
                             │ 实现于
┌────────────────────────────▼────────────────────────────────┐
│  后端实现层  FfmpegDecoder / GstDecoder / FfmpegEncoder /     │
│              GstEncoder + factory 工厂 + hw_probe 硬件探测     │
└─────────────────────────────────────────────────────────────┘
```

**为什么这样分？**

- **门面层**：稳定、好记的公共 API。底层换成什么库都不影响调用方。
- **路由层（DecodePipeline）**：把「抽帧、排队、复用、重连」这些**与具体后端无关的工程动作**集中到一处，后端实现只管「打开、读一帧、关」这种原子的码流操作。这样不管 FFmpeg 还是 GStreamer 都能共享同一套背压/缓冲池/重连能力。
- **后端抽象层**：`DecoderBackend` / `EncoderBackend` 定义统一契约，是「鸭子类型」的 C++ 表达（纯虚接口 + 工厂创建）。
- **后端实现层**：真正碰 FFmpeg/GStreamer 的地方，全部隔离在各自 `.cpp` 里，编译开关 `ENABLE_FFMPEG` / `ENABLE_GSTREAMER` 控制。

---

## 2. 核心抽象与数据模型

### 2.1 统一帧模型 `VideoFrame`

```cpp
struct VideoFrame {
    modeldeploy::vision::ImageData image;  // 内嵌视觉模块图片类型
    uint64_t pts_ms = 0;                   // 毫秒时间戳
};
```

- 解码统一输出 `ImageData`，格式为 **NV12**（视频界最常用 YUV420 半平面采样）。
- 时间戳 `pts_ms` 随帧一起带出，供追帧、对齐、测速等场景使用。
- 编码输入也统一收 `VideoFrame`，按 `image.device()`（CPU / GPU）路由处理路径。

### 2.2 零拷贝适配层 `IPlaneView`（adapter）

后端解出来的平面（Y / UV）通过统一的中间表示传入：

```cpp
struct IPlaneView {
    const uint8_t* y; const uint8_t* uv; int y_step; int uv_step;
    int width; int height;
    modeldeploy::Device device;
    std::shared_ptr<void> owner;   // 保活解码 buffer
};
```

`make_image_from_planes_view()` 据此**零拷贝**包出 `ImageData`（`owner` 负责保活底层解码缓冲）。这样「解码 → 预处理 → 推理」全程不需要把像素缓冲区翻来覆去地复制，是性能的关键设计之一。

---

## 3. 后端与编解码器矩阵

### 3.1 后端

| 后端 `CodecBackend` | 编译开关 | 说明 |
|---------------------|----------|------|
| `FFmpeg`（默认） | `-DENABLE_FFMPEG=ON` | 成熟、协议/容器支持全，适合本地文件与 RTSP/RTMP |
| `GStreamer` | `-DENABLE_GSTREAMER=ON` | 插件化流水线，适合 GPU 内存直通（CUDA memory） |
| `Auto` | — | 自动探测可用后端；只启用 FFmpeg 时即选 FFmpeg |

> **默认后端是 `FFmpeg`**（`VideoCodecConfig` 里 `backend = CodecBackend::FFmpeg`）。
> 若某个后端被编译为 OFF，创建该后端的门面会返回 `nullptr`（fail-closed），不会偷偷回退成另一个、给你静默换行为。

### 3.2 解码器

| 后端 | 软解（CPU） | 硬解（GPU） | 硬解自动回退 |
|------|------------|------------|:---:|
| FFmpeg | 通用软解 | `h264_cuvid` / `hevc_cuvid` / `av1_cuvid`（CUVID）、VAAPI(*compile-gated*) | ✅ |
| GStreamer | 通用软解 | `nvcodec`（如 `nvh264dec`） | ✅ |

- 硬解输出可能是 **CPU NV12**（如 cuvid 直接输出主机内存）或 **设备内存**（`device_only=true` 时 GPU 上）。
- `HwAccel::Auto` + 运行时硬解失败 → **零帧自动回退软解**（不会挂会话）。

### 3.3 编码器

| 后端 | 软编 | 硬编 |
|------|------|------|
| FFmpeg | `libx264` | `h264_nvenc` |
| GStreamer | `x264enc` | `nvh264enc`、`vaapih264enc` |

- `codec` 值集合：`auto` / `libx264` / `x264enc` / `h264_nvenc` / `nvh264enc` / `vaapih264enc`。
- `format`（输出容器）集合：`auto` / `mp4` / `flv` / `rtmp` / `rtsp`。
- 编码输入按 `image.device()` 路由：**CPU 输入由 SDK 拷贝**（无生命周期约束）；**GPU 输入（`device=GPU` 的设备 NV12）SDK 只借用、直通编码，不持有、不拷贝** —— 调用方必须保证这些显存平面在 `close()` 前有效（见 api.md 的 GPU 借用契约）。

---

## 4. 三大工程化能力（路由层的价值所在）

### 4.1 有界背压异步队列

异步解码时，后台解码线程把帧推入**有界队列**（容量 `async_queue_size`，默认 30）。队列满时的策略 `Backpressure`：

| 策略 | 行为 |
|------|------|
| `Block`（默认） | 解码线程阻塞，直到队列有空间 —— 用「慢下来」换取不丢帧 |
| `Drop` | 丢弃新帧（`dropped` 计数 +1） —— 用「丢帧」换取恒定吞吐 |
| `OverwriteOldest` | 覆盖最旧帧 —— 追求「最新画面」，适合实时预览 |

> 背压是**有界队列**的配套：没有背压上限，慢消费会被无限排队吞掉内存。三策略给了「保吞吐 / 保完整 / 保最新」三种取向。

### 4.2 帧缓冲池（零拷贝复用）

`VideoFrame` 容器默认走**缓冲池**（`pooling=true`）：交付给回调/调用方后**回收到池**，供下一帧复用，避免每帧反复 new/delete 大块像素缓冲。

- 测速/调试可观察：`pool_hits()`（复用命中次数）、`pool_returns()`（归还次数）、`stats().dropped`。
- `pooling=false` 时退化为每帧新建（内存更简单、但更慢），一般不需要关。

### 4.3 重连状态机

网络流（RTSP 等）难免断。路由层内置状态机：

```
Idle → Opening → Running ⇄ Reconnecting ⤵
                        ↓                Eof / Error / Closed
```

- 配置：`reconnect_delay_ms`（默认 5000）、`max_reconnects`（默认 10）、`timeout_us`（默认 10s）、`rtsp_transport`（默认 `tcp`）。
- **错误哲学**：区分**可重连的瞬态失败**（网络闪断 → 进入 `Reconnecting`，按间隔重试重开同一地址直到 `max_reconnects`）与**永久失败**（`ErrorCode::PermanentFailure` / `BackendUnavailable`，直接停）。
- `stats().reconnect_count` / `error_count` 可观测重连与出错次数。

---

## 5. 状态与统计

### 5.1 会话状态 `State`

`Idle / Opening / Running / Reconnecting / Eof / Error / Closed`，原子化，可跨线程安全轮询（`state()`）。

### 5.2 错误码 `ErrorCode`

`Ok / OpenFailed / ReadFailed / EncodeFailed / BackendUnavailable / NotInitialized / InvalidArgument / PermanentFailure`
—— 一个项目/一次失败到底是「能重试」还是「永久不行」，用类型直接区分，避免调用方靠猜。

### 5.3 统计 `VideoStats`

```cpp
struct VideoStats {
    uint64_t frames_in, frames_out, dropped;
    double avg_decode_ms, avg_encode_ms;
    uint64_t reconnect_count, error_count;
};
```

---

## 6. GPU 直通与设备直编

### 6.1 解码设备直通（`device_only`）

`set_device_only(true)` / `device_only=true` 后，硬解（CUDA）输出保持**设备内存 NV12**，不做显卡→主机的拷贝往返。追求「解码结果直接留在显存喂给 GPU 推理」的全 GPU 流水线时用。

### 6.2 编码 GPU 直编（`gpu_direct_input`）

`VideoEncoderConfig::gpu_direct_input=true`（配合 `hw_accel=Cuda` 且 `nvenc`/`nvh264enc`）时，`encode(const VideoFrame&)` 直接以 **device=GPU 的设备 NV12 直编**，不经主机往返。仅设备路径生效，CPU 编码不受影响。

> ⚠️ **生命周期借用契约**：GPU 直编时 SDK **借用调用方显存平面**编码（不做拷贝），调用方必须保证这些平面在 `close()` 之前一直有效。CPU 输入由 SDK 拷贝，无此约束。

---

## 7. 性能优化方向

以下方向按「投入产出比」排列，均已在架构上支持：

1. **能硬解就硬解**：`hw_accel=Cuda/Vaapi` 是单帧解码性能提升最明显的一步；`Auto` 会自动优先并兜底回退。
2. **解码结果留在设备内存**：`device_only` + GPU 推理 = 免主机往返，适合全 GPU 管线（带宽与延迟双赢）。
3. **编码同样用硬编 + 设备直编**：`h264_nvenc`/`nvh264enc` + `gpu_direct_input`，避免「编码前把画面搬回主机」。
4. **保零拷贝的帧复用它不浪费**：依赖 `pooling`（默认开），不要随意关；中间转换用 `cvt_color` 按需做、不要做多余的往返转格式。
5. **异步 + 恰当的背压**：高通量用 `set_callback + start()` 异步，选对 `Backpressure`：
   - 分析后处理慢但不许丢 → `Block`；
   - 实时监控画面要最新 → `OverwriteOldest`；
   - 吞吐优先、允许丢帧 → `Drop`。
6. **调码率/关键帧**：`bitrate_kbps`、`gop`、`max_b_frames`、`preset`（如编码默认 `ultrafast`）影响编码速度与体积的平衡；`low_latency` 影响端到端延迟。
7. **合理并发**：`clone()`-式多实例或解码/推理并行线程分摊 CPU/GPU，见整体 [性能指南](../performance.md)。

---

## 8. 已知限制与注意事项

- **GStreamer CUDA 的进程级限制**：GStreamer 一旦在进程内创建 CUDA context，其 nvcodec 会对本进程后续所有编码管道全局注册 CUDA 缓冲，可能影响其他行为。这是 GStreamer–CUDA 的进程级特性，非 SDK 逻辑缺陷。测试里把 GStreamer CUDA 互操作用例隔离到独立进程。
- **Sophgo 编码**：GStreamer 无对应编码插件，显式选 Sophgo 编码会 **fail-closed**（报 `unsupported-codec`），未实现未验证。
- **语言暴露**：视频模块暴露于 **C++ / C API / C# / Rust / Python**（全部解码+编码全功能）。C/C#/Rust 经 C API 阈值 `capi/md_capi.cpp`（`md_video_*`）承载，Python 经 `csrc/pybind/video/video_pybind.cpp` 直接绑定 C++ 门面。
- **编译/链接**：Windows 下需把 `build/bin` 与 FFmpeg 的 `bin` 加入 DLL 搜索路径（`PATH` / `os.add_dll_directory`），否则 `import modeldeploy` 或运行时可能找不到依赖 DLL。

---

## 9. 相关源码入口

| 关注点 | 文件 |
|--------|------|
| 门面 | `csrc/video/video_decoder.h/.cpp`、`video_encoder.h/.cpp` |
| 路由层（背压/缓冲池/重连） | `csrc/video/decode_pipeline.h/.cpp` |
| 后端抽象 | `csrc/video/backend/decoder_backend.h`、`encoder_backend.h` |
| 后端实现 | `csrc/video/backend/ffmpeg_*.cpp`、`gst_*.cpp` |
| 工厂 + 能力探测 | `csrc/video/factory.h/.cpp`、`hw_probe.h/.cpp` |
| 配置 / 状态 / 统计 | `csrc/video/video_codec_config.h`、`video_common.h` |
| 统一帧 / 零拷贝适配 | `csrc/video/video_frame.h`、`adapter.h/.cpp` |
| Python 绑定 | `csrc/pybind/video/video_pybind.cpp` |
| 使用示例 | `examples/demo_action/`（VideoDecoder 抽帧 + TSN 推理） |

继续看 [api.md](./api.md) 了解每个接口的签名与约定。
