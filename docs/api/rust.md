# Rust 绑定

Rust 通过 FFI 封装 C API。目录 `rust/modeldeploy/`：

```rust
use modeldeploy::runtime::RuntimeOption;

let mut option = RuntimeOption::new();
option.ort_backend();   // 或 sophgo_backend(0)/trt_backend()/mnn_backend()
```

## 主要模块文件

| 文件 | 说明 |
|------|------|
| `src/runtime.rs` | `RuntimeOption` 封装 |
| `src/ffi.rs` | FFI 声明（`MD_BACKEND_*` 常量等） |
| `src/model.rs` | 模型加载/推理 |
| `src/audio.rs` | 音频能力封装 |
| `src/barcode.rs` | 条码/二维码封装 |
| `src/nlp.rs` | NLP 能力封装 |
| `src/solution.rs` | 解决方案封装 |
| `src/tracker.rs` | 跟踪能力封装 |
| `src/video.rs` | 视频编解码封装（解码/编码全功能） |
| `src/image.rs` | 图像处理 |
| `src/types.rs` / `src/error.rs` | 类型与错误 |

## 示例（运行于 `rust/modeldeploy/examples/`）

内置 12 个示例：`classification` / `depth` / `detection` / `face_age` / `face_detection` / `face_gender` / `face_rec` / `obb` / `pose` / `seg` / `sem` / `video_decode`。

```bash
cd rust/modeldeploy
cargo run --example detection
cargo run --example video_decode -- demo.mp4 100
```

## 图像原始字节（`Image`）

`Image` 提供原生格式的宿主字节读取（`format()` / `width()` / `height()` / `plane_count()` 见 `Image`）：

```rust
let raw: Vec<u8> = img.to_native_bytes()?;   // 整幅原生连续字节；GPU 设备帧自动回读
let y:  Vec<u8>  = img.plane_bytes(0)?;      // 第 0 平面（NV12 的 Y）
let uv: Vec<u8>  = img.plane_bytes(1)?;      // 第 1 平面（NV12 的 UV）
```

- `to_native_bytes()` 布局随 `format()` 而定：BGR24=`w*h*3`；NV12=`w*h + w*h/2`；I420=`w*h + w*h/4 + w*h/4`。
- 平面越界 / 不支持类型返回 `Err`；方法立即把缓冲拷贝进自有 `Vec<u8>`。

## 视频编解码（`video`）

`video` 模块等价于 C++ `modeldeploy::video`（解码+编码全功能，经 C API `md_video_*` 承载）。主要公开类型：

| 类型 | 说明 |
|------|------|
| `VideoConfig` | 解码/编码配置（构建器，setter 返回 `&mut Self`） |
| `VideoCapabilities` | 能力探测 |
| `VideoDecoder` | 解码器：同步 `read_frame` + 异步回调 + 状态/统计 |
| `VideoEncoder` | 编码器：`encode` / `encode_async` / `start_async` |
| `VideoFrame { image, pts_ms }` | 统一帧（`Image` 本绑定自有） |
| `VideoStats` | 编解码统计 |

枚举：`CodecBackend{Auto,FFmpeg,GStreamer}`、`HwAccel{Auto,None,Cuda,Vaapi,Sophgo}`、`Backpressure{Block,Drop,OverwriteOldest}`、`VideoState{Idle,Opening,Running,Reconnecting,Eof,Error,Closed}`。

```rust
use modeldeploy::{Backpressure, CodecBackend, VideoConfig, VideoDecoder, VideoFrame};

let mut cfg = VideoConfig::new()?;
cfg.backend(CodecBackend::FFmpeg)?
    .backpressure(Backpressure::Block)?
    .pooling(true)?;

let dec = VideoDecoder::create(Some(&cfg))?;
dec.open("demo.mp4")?;                       // 或 rtsp://...
let (w, h, f) = dec.size()?;
println!("{}x{} @ {}fps", w, h, f);

while let Ok(VideoFrame { image, pts_ms }) = dec.read_frame() {
    // image 为本绑定自有 Image，Drop 自动释放底层 md_image_destroy
    println!("{}ms {}x{}", pts_ms, image.width(), image.height());
}
let st = dec.stats()?;                       // VideoStats
```

**异步解码（回调推送）**：`dec.set_callback(|frame| { ... })` 注册交付回调（帧为自有 `VideoFrame`，用后自动释放；回调自后台线程投递），再 `dec.start()` / `dec.stop()`。

**编码**：
```rust
let mut ecfg = VideoConfig::new()?;
ecfg.codec("h264")?.bitrate_kbps(2000)?.gop(30)?.fps(25)?;
let enc = modeldeploy::VideoEncoder::create(Some(&ecfg))?;
enc.open("out.mp4", 1280, 720, 25)?;
enc.encode(&img, pts_ms)?;                    // img 生命周期须覆盖本次调用
enc.close();
```

设备显存（`device_only` 解码 / `gpu_direct_input` 编码）零拷贝直通。详见 [视频接口总览](../video/api.md)。
