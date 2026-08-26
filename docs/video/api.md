# 视频编解码 · 接口参考（API）

> 这是**逐签名、逐参数、逐约定**的接口手册，面向 SDK 接口调用者（把每个接口用法抠清楚）。
> 想看整体流程，先去 [guide.md](./guide.md)；想理解设计，去 [design.md](./design.md)。

## 0. 命名空间与头文件

- C++ 头文件：`#include "modeldeploy/video.h"`
- 命名空间：`modeldeploy::video`
- 统一帧类型来自视觉模块：`modeldeploy::vision::ImageData`

**语言暴露情况：**

| 语言 | 状态 |
|------|------|
| C++ | ✅ 完整（`VideoDecoder` / `VideoEncoder`） |
| C API | ✅ 完整（`md_video_*`，见 [../api/capi.md](../api/capi.md)） |
| C# | ✅ 完整（`VideoDecoder` / `VideoEncoder` / `VideoConfig` / `VideoCapabilities`） |
| Rust | ✅ 完整（`VideoDecoder` / `VideoEncoder` / `VideoConfig` / `VideoCapabilities`） |
| Python | ✅ 完整（`VideoDecoder` / `VideoEncoder` / 全量配置 / 能力探测） |

---

## 1. 公共枚举与类型（`video_common.h`）

### CodecBackend（后端）
```cpp
enum class CodecBackend { Auto, FFmpeg, GStreamer };
```
配置里默认 `FFmpeg`。`Auto` 自动探测可用后端（只启用 FFmpeg 时即选它）。

### HwAccel（硬件加速）
```cpp
enum class HwAccel { Auto, None, Cuda, Vaapi, Sophgo };
```
- `Auto`：优先硬解，硬解失败自动回退软解。
- `None`：强制软解。
- `Cuda`：NVIDIA CUVID/NVENC。
- `Vaapi`：VAAPI（Linux，Intel/AMD）。
- `Sophgo`：算能 TPU（BM1688 等）。编码暂不支持（fail-closed）。

### Backpressure（背压策略，`video_codec_config.h`）
```cpp
enum class Backpressure { Block, Drop, OverwriteOldest };
```

### State（会话状态）
```cpp
enum class State { Idle, Opening, Running, Reconnecting, Eof, Error, Closed };
```

### ErrorCode（错误码）
```cpp
enum class ErrorCode {
    Ok, OpenFailed, ReadFailed, EncodeFailed,
    BackendUnavailable, NotInitialized, InvalidArgument, PermanentFailure
};
```

### VideoStats（统计）
```cpp
struct VideoStats {
    uint64_t frames_in; uint64_t frames_out; uint64_t dropped;
    double avg_decode_ms; double avg_encode_ms;
    uint64_t reconnect_count; uint64_t error_count;
};
```

### VideoFrame（统一帧）
```cpp
struct VideoFrame {
    modeldeploy::vision::ImageData image;  // 解码输出为 NV12
    uint64_t pts_ms = 0;                   // 毫秒时间戳
};
```

---

## 2. 配置（`video_codec_config.h`）

### VideoCodecConfig（解码/编码共用）
```cpp
struct VideoCodecConfig {
    CodecBackend backend = CodecBackend::FFmpeg;
    HwAccel       hw_accel   = HwAccel::Auto;
    int           reconnect_delay_ms = 5000;   // 重连间隔
    int           max_reconnects     = 10;     // 最大重连次数
    int           timeout_us    = 10000000;    // 10s 超时
    std::string   rtsp_transport = "tcp";      // rtsp 传输协议
    bool          device_only   = false;       // 解码输出保持设备 NV12（GPU 直通）
    int           async_queue_size = 30;       // 异步队列容量（有界）
    bool          pooling       = true;        // 帧缓冲池开关
    Backpressure  backpressure  = Backpressure::Block;
};
```

### VideoDecoderConfig（= VideoCodecConfig + 校验）
```cpp
struct VideoDecoderConfig : VideoCodecConfig {
    bool validate(std::string* err) const;   // 校验配置，非法时返回 false 并写 err
};
```

### VideoEncoderConfig（编码专用）
```cpp
struct VideoEncoderConfig : VideoCodecConfig {
    int  fps = 0;              // 0=自动
    int  bitrate_kbps = 2500;
    int  gop = 12;
    std::string codec = "auto";   // auto/libx264/x264enc/h264_nvenc/nvh264enc/vaapih264enc
    std::string preset = "ultrafast";
    std::string format = "auto";  // auto/rtsp/rtmp/flv/mp4
    int  max_b_frames = 0;
    bool low_latency = true;
    bool gpu_direct_input = false;  // GPU 显存 NV12 直编（需 hw_accel=Cuda 且 nvenc/nvh264enc）

    VideoEncoderConfig& set_fps(int);          // 链式 setter
    VideoEncoderConfig& set_bitrate_kbps(int);
    VideoEncoderConfig& set_codec(const std::string&);
    VideoEncoderConfig& set_format(const std::string&);
    bool validate(std::string* err) const;
};
```

> 编码 `codec` 与后端要匹配：选 FFmpeg 后端时用 `libx264`(软)/`h264_nvenc`(硬)；选 GStreamer 时用 `x264enc`(软)/`nvh264enc`(硬)/`vaapih264enc`(硬)。可用 `query_video_capabilities()` 探测当前环境实际可用的编解码器名。

---

## 3. 能力探测

```cpp
struct VideoCodecCapabilities {
    bool ffmpeg_available;
    bool gstreamer_available;
    std::vector<std::string> hw_decoders;  // 如 "h264_cuvid"
    std::vector<std::string> hw_encoders;  // 如 "h264_nvenc"
};
VideoCodecCapabilities query_video_capabilities();   // 编译+运行环境实际可用能力
```

用于运行时判断「当前机器有没有 FFmpeg/某硬解硬编」，避免盲目创建。

---

## 4. 解码器 VideoDecoder（C++）

```cpp
class VideoDecoder {
public:
    // 工厂：按配置创建后端并返回门面；后端不可用时置 err 并返回 nullptr
    static std::shared_ptr<VideoDecoder>
        create(const VideoDecoderConfig& cfg, std::string* err = nullptr);

    ~VideoDecoder();                       // 析构自动 close()（幂等）

    bool open(const std::string& url, std::string* err = nullptr);

    // 同步抽下一帧到 *out（CPU NV12 + 毫秒时间戳）；失败/EOF 返回 false
    bool read_one_frame(VideoFrame* out, std::string* err = nullptr);

    // 异步：注册回调（帧以移动语义交付，消费后容器回收回池）
    using FrameCallback = std::function<void(VideoFrame&&)>;
    void set_callback(FrameCallback cb);
    bool start(std::string* err = nullptr);   // 启动后台解码+交付线程
    void stop();                              // 请求停止、排空队列、join 线程（幂等）

    void set_device_only(bool v);             // 切换是否 GPU 设备内存直通

    State             state() const;          // 会话状态（原子，跨线程可轮询）
    const VideoStats& stats() const;
    std::string       last_error() const;
    int               fps() const;            // 打开后可查
    int               width() const;
    int               height() const;
    void              close();                // 幂等：stop + 关后端 + Closed

    // 测试可观测：缓冲池命中/归还次数；dropped/reconnect_count 经 stats()
    uint64_t pool_hits() const;
    uint64_t pool_returns() const;
};
```

**生命周期约定：**
1. `create` 返回 `shared_ptr`，用 RAII 管理；`nullptr` = 创建失败（后端不可用）。
2. `open` 成功后可查 `width()/height()/fps()`；未 `open` 前它们无意义。
3. **同步 / 异步二选一**：
   - 同步：只调 `read_one_frame` 循环取帧（不经异步队列、不触发重连的会话级能力）。
   - 异步：`set_callback` + `start`，走背压队列 + 缓冲池 + 重连。
4. `close()`/析构幂等，随时可安全调用。

---

## 5. 编码器 VideoEncoder（C++）

```cpp
class VideoEncoder {
public:
    static std::shared_ptr<VideoEncoder>
        create(const VideoEncoderConfig& cfg, std::string* err = nullptr);
    ~VideoEncoder();

    bool open(const std::string& url, int w, int h, int src_fps, std::string* err = nullptr);

    // 编码一帧（CPU BGR / 设备 NV12 + 可选 pts_ms）；失败返回 false
    bool encode(const VideoFrame& frame, std::string* err = nullptr);

    // 异步编码
    bool encode_async(const modeldeploy::vision::ImageData& image);
    bool start_async(std::string* err = nullptr);
    void stop_async();
    bool has_permanently_failed() const;

    State             state() const;
    std::string       last_error() const;
    const VideoStats& stats() const;
    void              close();   // 幂等
};
```

**GPU 借用契约（重要）：**
- `encode(frame)` 时若 `frame.image.device() == Device::GPU`（设备 NV12），SDK **借用调用方显存平面直通编码、不拷贝**，且**不持有其生命周期** —— 调用方必须保证这些平面在 `close()` **之前始终有效**。
- CPU 输入由 SDK 拷贝处理，无此约束。
- `open` 必须传 `w,h,src_fps`（输出尺寸与源帧率）。
- **编码输出容器**（如 mp4）要正确，`close()` 前才会把尾部索引（moov）写盘 —— 别忘 `close()`。

---

## 6. Python 接口（`modeldeploy.video`）

Python 与 C++ / C / C# / Rust **全功能对齐**：解码+编码、全量配置、能力探测、异步、状态/统计。

```python
import modeldeploy as md
from modeldeploy import vision   # ImageData（NV12 构造用）

# 能力探测（等价 query_video_capabilities）
cap = md.video.query_video_capabilities()
print(cap.ffmpeg_available, cap.hw_decoders, cap.hw_encoders)

# ── 解码 ──
cfg = md.video.VideoDecoderConfig()          # 全字段，均有默认
cfg.backend   = md.video.CodecBackend.FFmpeg
cfg.hw_accel  = md.video.HwAccel.Cuda       # Auto 自动回退软解
cfg.device_only = False
cfg.async_queue_size = 30
cfg.backpressure = md.video.Backpressure.Block
cfg.pooling = True
cfg.reconnect_delay_ms = 5000
cfg.max_reconnects = 10

dec = md.video.VideoDecoder(cfg)
ok  = dec.open("demo.mp4")                  # 或 rtsp://...；失败抛异常
w, h, fps_ = dec.width, dec.height, dec.fps

ok, image, pts = dec.read_frame()           # 返回 (是否成功, NV12帧ImageData, 毫秒时间戳)
dec.close()                                 # 幂等

# ── 异步解码（回调推送，后台线程投递，帧为自有 ImageData）──
async_dec = md.video.VideoDecoder(cfg)
async_dec.set_callback(lambda image, pts_ms: print(pts_ms, image.width, image.height))
async_dec.open("demo.mp4")
async_dec.start()
# ... 同时做别的事 ...
async_dec.stop()

# ── 编码 ──
ecfg = md.video.VideoEncoderConfig()
ecfg.codec = "libx264"; ecfg.fps = 25; ecfg.bitrate_kbps = 2000; ecfg.format = "mp4"
enc = md.video.VideoEncoder(ecfg)
enc.open("out.mp4", 1280, 720, 25)
enc.encode(image, pts)                      # image 为 ImageData（CPU BGR 或设备 NV12）
enc.close()                                 # 必须 close，mp4 尾部索引(moov)在此写盘
```

**VideoDecoder 成员：**

| 成员 | 签名 | 说明 |
|------|------|------|
| `VideoDecoder(cfg=None)` | 构造函数 | `cfg` 为 `VideoDecoderConfig`，缺省用默认；创建失败抛异常 |
| `open(url)` | `-> bool` | 打开视频/流；失败抛异常（含底层错误） |
| `read_frame()` | `-> (bool, ImageData, int)` | 抽下一帧；EOF/失败时第一个元素为 False |
| `set_callback(cb)` | — | 异步交付回调 `cb(image, pts_ms)`（后台线程投递） |
| `start()` | `-> bool` | 启动后台解码线程（需先 `set_callback`） |
| `stop()` | — | 停止 |
| `set_device_only(enable)` | — | 切换 GPU 设备内存直通 |
| `close()` | — | 幂等关闭 |
| `width / height / fps` | 只读属性 | 打开后有效 |
| `state` | 只读属性 | 当前 `State` |
| `last_error()` | `-> str` | 最近一次错误 |
| `stats()` / `pool_hits()` / `pool_returns()` | — | 统计 / 缓冲池可观测 |

**VideoEncoder 成员：**

| 成员 | 签名 | 说明 |
|------|------|------|
| `VideoEncoder(cfg=None)` | 构造函数 | `cfg` 为 `VideoEncoderConfig`；创建失败抛异常 |
| `open(url, w, h, src_fps)` | `-> bool` | 打开输出容器/流 |
| `encode(image, pts_ms=0)` | `-> bool` | 编码一帧（CPU BGR / 设备 NV12） |
| `encode_async(image)` / `start_async()` / `stop_async()` | — | 异步编码 |
| `has_permanently_failed()` | `-> bool` | 是否永久失败 |
| `close()` | — | 幂等关闭；mp4 尾索引在此写盘 |
| `state` / `last_error()` / `stats()` | — | 状态 / 错误 / 统计 |

**Python 可用枚举：** `CodecBackend.{Auto,FFmpeg,GStreamer}`、`HwAccel.{Auto,None,Cuda,Vaapi,Sophgo}`、`Backpressure.{Block,Drop,OverwriteOldest}`、`State.{Idle,Opening,Running,Reconnecting,Eof,Error,Closed}`。

**`VideoDecoderConfig` 全量字段：** `backend`、`hw_accel`、`device_only`、`async_queue_size`、`backpressure`、`pooling`、`reconnect_delay_ms`、`max_reconnects`、`timeout_us`、`rtsp_transport`。

**`VideoEncoderConfig` 全量字段：** 上述共用字段 + `fps`、`bitrate_kbps`、`gop`、`codec`、`preset`、`format`、`max_b_frames`、`low_latency`、`gpu_direct_input`。

> ImageData 由 `modeldeploy.vision.ImageData` 提供（`from_nv12` / `from_device_nv12` 构造帧），与解码 `read_frame` 返回的类型一致。设备显存（`device_only` / `gpu_direct_input`）零拷贝直通。

---

## 7. 常见错误信息示例

| 报错字符串 | 含义 / 处理 |
|------------|------------|
| `backend unavailable` | 目标后端正交编译进来或探测不可用，`create` 返回 nullptr |
| `unsupported-codec` | 后端不支持该 codec 名（如给 GStreamer 传 `libx264`，或选 Sophgo 编码） |
| `no-nvh264enc` / `no-x264enc` / `no-vaapih264enc` | 指定编码器插件当前环境不存在 |
| `gpu-direct-needs-nvh264enc` | 开了 `gpu_direct_input` 但没选 `nvh264enc`/`h264_nvenc` |
| `nvcodec-*`（GStreamer 硬解） | nvcodec 相关能力不足/插件不可用/映射失败 |

---

继续看 [README.md](./README.md) 索引，或回到 [../README.md](../README.md)。
