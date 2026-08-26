# 视频编解码 · 快速上手指南（小白版）

> 这是给**第一次接触视频编解码、第一次用 ModelDeploy SDK 接口的人**写的教程。
> 如果你已经会用，想直接看接口签名，去 [api.md](./api.md)；想看实现原理，去 [design.md](./design.md)。

## 0. 先理解三件事（30 秒）

**① 什么是解码 / 编码？**

- **解码（decode）**：把一段“压缩好的视频文件或网络流”变成一帧一帧的画面。压缩流你不能直接拿来跑 AI，解码后每一帧才是能喂给检测/识别模型的图片。
- **编码（encode）**：反过来，把一帧帧画面压缩成视频文件或网络流（推送 RTSP/RTMP、存成 mp4 等）。

**② ModelDeploy 帮你做了什么？**

ModelDeploy 把「用什么库解、是否用 GPU 硬解、出流出错怎么办、帧快了怎么刹住」这些麻烦全封装了。你只需要：

1. 创建一个解码器/编码器对象；
2. 告诉它地址；
3. 一帧一帧地拿/送画面。

**③ 你能拿到什么？**

解码后每一帧是一个 `VideoFrame`，里面有：

- `image` —— 一个 `ImageData`（ModelDeploy 统一的图片类型，解码输出为 **NV12** 格式）；
- `pts_ms` —— 这一帧的时间戳（毫秒）。

### 支持矩阵速览

| 项 | 说明 |
|----|------|
| 后端 | **FFmpeg**（默认）/ **GStreamer**（需 `-DENABLE_GSTREAMER=ON`） |
| 解码 | 软解（CPU）+ 硬解（CUDA/NV 卡、VAAPI、Sophgo TPU） |
| 编码 | 软编 `libx264`/`x264enc` + 硬编 `h264_nvenc`/`nvh264enc`/`vaapih264enc` |
| 输出容器 | mp4 / flv / rtmp / rtsp |
| 语言 | **C++**、**C API**、**C#**、**Rust**、**Python**（全部解码+编码全功能） |

---

## 1. 需要准备什么

- 一份带视频能力的构建。视频模块是**可开关**的编译选项：

```bash
cmake -S . -B build -G Ninja \
  -DBUILD_VIDEO=ON -DBUILD_VISION=ON \
  -DENABLE_ORT=ON -DENABLE_MNN=OFF -DENABLE_TRT=OFF -DWITH_GPU=OFF \
  -DENABLE_FFMPEG=ON -DENABLE_GSTREAMER=OFF
cmake --build build --parallel
cmake --install build
```

> 关键点：
> - `BUILD_VIDEO=ON` 是开关，**必须同时 `BUILD_VISION=ON`**（解出来的帧是视觉模块的 `ImageData`）。
> - `BUILD_VIDEO` 要求 FFmpeg 或 GStreamer 至少有一个编译进去；找不到时 `BUILD_VIDEO` 会自动关掉（优雅禁用）。
> - `ENABLE_FFMPEG=ON`（默认）就会带 FFmpeg 后端；想用 GStreamer 后端再开 `ENABLE_GSTREAMER=ON`。
> - 想用 Python 接口，再加 `-DBUILD_PYTHON=ON`。

- 一段视频：本地 `demo.mp4`，或一个 RTSP 摄像头地址 `rtsp://127.0.0.1:8554/test`。

---

## 2. 第一个解码程序（C++）

把下面的代码保存成 `demo_decode.cpp`：

```cpp
#include "modeldeploy/video.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("Usage: demo_decode <video.mp4|rtsp://...>\n"); return 1; }

    // 1) 创建解码器（用默认配置）
    std::string err;
    auto dec = modeldeploy::video::VideoDecoder::create(
        modeldeploy::video::VideoDecoderConfig{}, &err);
    if (!dec) { std::printf("create failed: %s\n", err.c_str()); return 1; }

    // 2) 打开视频/流
    if (!dec->open(argv[1], &err)) { std::printf("open failed: %s\n", err.c_str()); return 1; }
    std::printf("input: %dx%d @ %d fps\n", dec->width(), dec->height(), dec->fps());

    // 3) 一帧一帧读（同步方式）
    modeldeploy::video::VideoFrame vf;
    int n = 0;
    while (n < 100 && dec->read_one_frame(&vf, &err)) {
        // vf.image 是 NV12 帧，直接喂给检测模型，或转成 BGR 用于显示/存图
        modeldeploy::vision::ImageData bgr = modeldeploy::vision::ImageData::cvt_color(
            vf.image, modeldeploy::vision::ColorConvertType::CVT_NV122PKG_BGR);
        std::printf("frame %d: %dx%d pts=%llums\n", n++, vf.image.width(), vf.image.height(),
                    (unsigned long long)vf.pts_ms);
    }

    // 4) 关闭（其实析构也会自动关，但显式关更规范）
    dec->close();
    return 0;
}
```

**逐行解释：**

- `VideoDecoder::create(cfg, &err)` 返回一个 `shared_ptr<VideoDecoder>`。**如果后端不可用，返回 `nullptr`**，错误写在 `err`。
- `dec->open(url)` 打开本地文件或网络流。打开后可以查 `width()/height()/fps()`。
- `read_one_frame(&vf)` 同步阻塞地取下一帧。返回 `false` 表示**读完了（EOF）或出错**——所以放在 while 条件里最稳。
- 解出来的 `vf.image` 是 **NV12** 双平面格式。上文用 `ImageData::cvt_color(..., CVT_NV122PKG_BGR)` 转成 BGR 便于后续处理。

> **为什么解出来是 NV12？** NV12 是视频领域最常用的 YUV 采样格式之一。ModelDeploy 的解码统一输出 NV12，方便你在「解码 → 预处理 → 推理」之间直接零拷贝流转，省去多余的颜色转换。视觉模块的推理管线大多能直接吃 NV12。

---

## 3. 用 Python 写第一个解码程序

```python
import modeldeploy as md

dec = md.video.VideoDecoder()          # 默认配置（FFmpeg + 自动硬件）
if not dec.open("demo.mp4"):           # 也可传 rtsp://...
    raise SystemExit("无法打开视频")

print(dec.width, "x", dec.height, "@", dec.fps, "fps")

for i in range(100):
    ok, image, pts = dec.read_frame()  # 返回 (是否成功, NV12帧ImageData, 时间戳毫秒)
    if not ok:
        break
    print(f"frame {i}: {image.width}x{image.height} pts={pts}ms")

dec.close()
```

**Python 与 C++ 的对应关系：**

| C++ | Python |
|-----|--------|
| `VideoDecoder::create(cfg)` | `VideoDecoder(cfg?)` |
| `dec->open(url)` | `dec.open(url)` |
| `read_one_frame(&vf)` | `ok, image, pts = dec.read_frame()` |
| `dec->close()` | `dec.close()` |
| `dec->fps()/width()/height()` | `dec.fps / dec.width / dec.height` |

Python 同样**全功能**：支持编码与异步读（与 C++ 对齐）——

```python
import modeldeploy as md

# 异步解码（回调推送）
dec = md.video.VideoDecoder()
dec.set_callback(lambda image, pts_ms: print(pts_ms, image.width, image.height))
dec.open("demo.mp4"); dec.start(); dec.stop()

# 编码成 mp4
ecfg = md.video.VideoEncoderConfig()
ecfg.codec = "libx264"; ecfg.fps = 25; ecfg.bitrate_kbps = 2000
enc = md.video.VideoEncoder(ecfg)
enc.open("out.mp4", 1280, 720, 25)
# enc.encode(image, pts)  # image 为 ImageData（可用 read_frame 的返回项）
enc.close()              # 必须 close，mp4 尾索引在此写盘
```
> 完整成员/字段表见 [api.md](./api.md#6-python-接口modeldeployvideo)。

---

## 4. 同步读 vs 异步读（重要）

解码有两种用法，别搞混：

**同步（默认、最常用来「卡点取帧」）**
```cpp
VideoFrame vf;
while (dec->read_one_frame(&vf)) { /* 处理这一帧 */ }
```
- 你调一次，返回一帧。适合「我慢条斯理地处理，处理完再要下一帧」。
- 网络流卡顿时，`read_one_frame` 会阻塞等待。

**异步（回调推送，适合高通量流水线）**
```cpp
dec->set_callback([](modeldeploy::video::VideoFrame&& vf) {
    // 后台线程把解好的帧"推"给你，你在此回调里处理
});
dec->start();      // 启动后台解码线程
// ... 同时做别的事 ...
dec->stop();       // 停止
```
- 后台有**专门的解码线程+交付线程**，解好的帧通过回调自动送过来，你不必自己管理线程。
- 适合「解码要持续跑，处理管线并行」的场景。
- 异步同时为你准备了**背压**和**缓冲池**两个能力（见第 5 节），防止队列无限膨胀。

---

## 5. 切到硬解（GPU 加速，一步）

一道配置就能从「CPU 软解」切到「NVIDIA 硬解」，配置放在 `VideoDecoderConfig` 里：

```cpp
modeldeploy::video::VideoDecoderConfig cfg;
cfg.backend   = modeldeploy::video::CodecBackend::FFmpeg;  // 或 GStreamer
cfg.hw_accel  = modeldeploy::video::HwAccel::Cuda;         // 关键：切到 CUDA 硬解
auto dec = modeldeploy::video::VideoDecoder::create(cfg);
```

- `HwAccel::Auto`：自动优先硬解，硬解不可用（显卡/驱动/格式不支持）时**自动回退软解**，不会挂会话。
- `HwAccel::Cuda`：NVIDIA CUVID/NVENC 硬解硬编。
- `HwAccel::Vaapi`：Intel/AMD 的 VAAPI（Linux）。
- `HwAccel::Sophgo`：算能 TPU（BM1688 等，`set_cls_batch_size(1)` 场景）。
- `HwAccel::None`：强制软解。

> 编码侧想用 GPU 直编，看 [api.md](./api.md#编码器-videoencoder) 里 `gpu_direct_input` 一节。

---

## 6. 第一个编码程序（把帧写成 mp4）

```cpp
#include "modeldeploy/video.h"

modeldeploy::video::VideoEncoderConfig cfg;
cfg.set_fps(25).set_bitrate_kbps(2000).set_codec("libx264");  // CPU 软编 H.264
auto enc = modeldeploy::video::VideoEncoder::create(cfg);
std::string err;
if (!enc->open("out.mp4", 1280, 720, 25, &err)) { /* 失败 */ }

// 一帧一帧送进去（frame.image 是 BGR 或 NV12 均可，内部会按需转换）
for (auto& frame : frames) {
    if (!enc->encode(frame, &err)) break;
}
enc->close();   // 必须 close，mp4 的尾部索引（moov）在这里写盘
```

- `codec`: `auto` / `libx264`(FFmpeg 软编) / `h264_nvenc`(FFmpeg 硬编) / `x264enc`(GStreamer 软编) / `nvh264enc`(GStreamer 硬编) / `vaapih264enc`(VAAPI)。
- `format`: `auto` / `mp4` / `flv` / `rtmp` / `rtsp`。
- 编码 H.264 时也支持 `gpu_direct_input` 直接从 GPU 显存 NV12 直编，省掉主机往返（见 api.md）。

---

## 7. 跨语言绑定（C API / C# / Rust）

除 C++ / Python 外，视频模块还通过 **C API（`md_video_*`）** 暴露给 **C#** 与 **Rust**，三者功能完全对齐 C++。以下给三端各一段「同步解码」最小示例（编码/异步/能力探测的姿态一致）。

**C（`capi/md_capi.h`）**
```c
#include "md_capi.h"
MDVideoConfigHandle cfg; md_video_config_create(&cfg);
md_video_config_set_backend(cfg, MD_CODEC_FFMPEG);
MDVideoDecoderHandle dec; md_video_decoder_create(cfg, &dec);
md_video_decoder_open(dec, "demo.mp4");

MDImageHandle frame; uint64_t pts;
while (md_video_decoder_read_frame(dec, &frame, &pts) == MD_OK) {
    /* frame 为自有 NV12 帧，用后必须 md_image_destroy(frame)（所有权转移） */
    md_image_destroy(frame);
}
md_video_decoder_destroy(dec);
md_video_config_destroy(cfg);
```
> C 最大的坑是**所有权**：`md_video_decoder_read_frame` / 异步回调交付的 `frame` 归你所有，用完必须 `md_image_destroy`（对齐 C++ 移动交付语义），否则泄漏。

**C#（`ModelDeploy` 命名空间）**
```csharp
using ModelDeploy;

var cfg = new VideoConfig { Backend = CodecBackend.FFmpeg };
using var dec = new VideoDecoder(cfg);        // IDisposable，析构自动释放
dec.Open("demo.mp4");

VideoFrame vf;
while (dec.ReadFrame(out vf)) {
    // vf.Image 是 NV12 帧（已归 C# 管理，用后自动释放）
    Console.WriteLine($"{vf.PtsMs}ms {vf.Image.Width}x{vf.Image.Height}");
}
```

**Rust（`modeldeploy` crate）**
```rust
use modeldeploy::{Backpressure, CodecBackend, VideoConfig, VideoDecoder, VideoFrame};

let mut cfg = VideoConfig::new()?;
cfg.backend(CodecBackend::FFmpeg)?.backpressure(Backpressure::Block)?;
let dec = VideoDecoder::create(Some(&cfg))?;
dec.open("demo.mp4")?;

while let Ok(VideoFrame { image, pts_ms }) = dec.read_frame() {
    // image 是本绑定自有的 Image，Drop 自动释放
    println!("{}ms {}x{}", pts_ms, image.width(), image.height());
}
```
> Rust 的 `read_frame` 返回的 `Image` 与异步回调里收到的帧同样是**自有所有权**，无需手动释放（`Drop` 自动 `md_image_destroy`）。
> 各语言接口签名对照，见 [api.md](./api.md) 语言暴露表与 [C API](../api/capi.md)、[C#](../api/csharp.md)、[Rust](../api/rust.md) 的视频章节。

---

## 8. 出错了怎么办（FAQ 简答）

| 现象 | 原因 / 处理 |
|------|------------|
| `create` 返回 `nullptr` | 没编译进视频模块（`BUILD_VIDEO=OFF` 或 FFmpeg/GStreamer 没找到），或指定后端不可用。 |
| `open` 返回 false | 地址打不开、协议不支持。可从 `err`/`last_error()` 里看具体信息。网络流不要忘了 `rtsp_transport="tcp"`（默认即 tcp）。 |
| 硬解 `Auto` 却在软解 | 显卡/驱动/格式不支持该硬解，自动回退了——这是正常兜底。 |
| Windows 下 mp4 没有 moov / 播放器打不开 | 编码完**必须 `close()`**，尾索引才会写盘。 |
| Python `import modeldeploy` 报 DLL 错误 | 运行环境缺 `ModelDeploySDK.dll` 或 FFmpeg 等依赖的 DLL 在 PATH/搜索路径里，把 `build/bin` 与 FFmpeg `bin` 加进 `PATH` 或 `os.add_dll_directory`。 |

---

## 9. 下一步

- 想完整看每个接口怎么用、每个参数什么意思 → [api.md](./api.md)
- 想看懂背后是怎么架构的、为什么这么设计、性能怎么调 → [design.md](./design.md)
- 项目文档总览 → [../README.md](../README.md)
