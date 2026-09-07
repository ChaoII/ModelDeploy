# Rust 绑定

Rust 通过 FFI 封装 C API。目录 `rust/modeldeploy/`：

## 1. 引入

```toml
[dependencies]
modeldeploy = { path = "path/to/ModelDeploy/rust/modeldeploy" }
```

常用类型在 crate 顶层 re-export：`RuntimeOption`、`Image`、各模型类（如 `UltralyticsDet`）与 `MdError`；设备枚举 `MDDevice` 位于 `ffi` 模块（`modeldeploy::ffi::MDDevice`）。

## 2. RuntimeOption（后端/设备/精度）

```rust
use modeldeploy::{Image, RuntimeOption, UltralyticsDet};
use modeldeploy::ffi::MDDevice;

fn main() -> Result<(), modeldeploy::MdError> {
    let mut opt = RuntimeOption::new()?;          // 返回 Result
    opt.use_ort().set_device(MDDevice::CPU, 0)?;  // 设备
    opt.set_cpu_threads(4)?;                      // CPU 线程数

    // 其它后端（&mut self -> &mut Self，可链式）：
    // opt.use_mnn() / opt.use_trt() / opt.use_sophgo() / opt.use_ncnn()
    // 精度：opt.set_fp16(true)?
    // GPU：opt.use_ort().set_device(MDDevice::GPU, 0)?
    // OPENCL/VULKAN 需显式 MNN 后端，否则 fail-closed：
    // opt.use_mnn().set_device(MDDevice::OPENCL, 0)?;

    let model = UltralyticsDet::new("yolo11n.onnx", &opt)?;
    let img = Image::read("test.jpg")?;
    let dets = model.predict(&img)?;
    Ok(())
}
```

- `RuntimeOption::new()` 返回 `Result`；setter 返回 `Result<&mut Self, MdError>`，用 `?` 链式调用。
- 后端方法：`use_ort` / `use_mnn` / `use_trt` / `use_sophgo` / `use_ncnn`。
- 设备：`set_device(MDDevice::CPU, 0)?`；线程数 `set_cpu_threads(n)?`；精度 `set_fp16(bool)?`。

## 3. 目标检测（`UltralyticsDet`）

检测模型 `UltralyticsDet::new(path, &opt)?` 加载。结果类型 `Detection`（字段 `rect: Rect{x, y, width, height}`、`label_id: i32`、`score: f32`）；`predict` 返回 `Vec<Detection>`，`predict_batch` 返回 `Vec<Vec<Detection>>`（按图分组）。

```rust
use modeldeploy::{DrawOptions, Image, RuntimeOption, UltralyticsDet};
use modeldeploy::ffi::MDDevice;

fn main() -> Result<(), modeldeploy::MdError> {
    // 1. 运行时选项（详见上节）
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(MDDevice::CPU, 0)?.set_cpu_threads(4)?;

    // 2. 构造模型
    let model = UltralyticsDet::new("yolo11n.onnx", &opt)?;

    // 3. 预处理/后处理参数（均为 Result<(), MdError>）
    model.set_input_size(640, 640)?;      // letterbox 输入尺寸
    model.set_conf_threshold(0.25)?;      // 置信度阈值（默认 0.25）
    model.set_nms_threshold(0.45)?;       // NMS IoU 阈值（默认 0.5）

    // 4. 单图推理：predict(&Image) -> Vec<Detection>
    let img = Image::read("test.jpg")?;
    let dets = model.predict(&img)?;
    for d in &dets {
        println!("label={} score={:.3} rect=({:.0},{:.0},{:.0},{:.0})",
                 d.label_id, d.score, d.rect.x, d.rect.y, d.rect.width, d.rect.height);
    }

    // 5. 批量推理：predict_batch(&[&Image]) -> Vec<Vec<Detection>>
    let img2 = Image::read("bus.jpg")?;
    let batch = model.predict_batch(&[&img, &img2])?;
    for (i, dets) in batch.iter().enumerate() {
        println!("image {}: {} objects", i, dets.len());
    }

    // 6. 可视化：predict_and_draw 句柄直达 C++ vis_det，把结果绘制到 canvas（并返回检测结果）
    let canvas = img.clone()?;
    model.predict_and_draw(
        &img,
        &canvas,
        &DrawOptions::new()
            .with_threshold(0.25)
            .with_label_map(vec![(0, "person".into()), (1, "bicycle".into()), (2, "car".into())])
            .with_alpha(0.3),
    )?;
    canvas.save("det_vis.png")?;

    // 7. 多线程：clone() 深拷贝独立实例（返回 Result<Self, MdError>，每线程持有一个）
    let model2 = model.clone()?;
    Ok(())
}
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
