# 视频编解码（Video Codec）

ModelDeploy 的视频编解码模块，为「视频/流 → 帧」与「帧 → 视频/流」提供**后端无关、可硬解硬编、自带工程化能力**的统一接口。

## 它能做什么

- **解码**：本地视频文件 / RTSP / RTMP 等 → 一帧帧 `VideoFrame`（NV12），支持 CPU 软解与 CUDA/VAAPI/Sophgo 硬解（`Auto` 自动回退软解）。
- **编码**：一帧帧画面 → mp4 / flv / rtmp / rtsp，支持 `libx264`/`x264enc` 软编与 `h264_nvenc`/`nvh264enc`/`vaapih264enc` 硬编。
- **GPU 直通**：`device_only` 解码设备内存直通、`gpu_direct_input` 编码显存直编，免主机往返。
- **工程化**：有界背压异步队列、帧缓冲池（零拷贝复用）、断流自动重连状态机、状态/统计可观测。

## 文档

| 文档 | 内容 | 适合 |
|------|------|------|
| [使用教程 guide.md](./guide.md) | 小白向：从构建到第一个解码/编码程序，全语言（C++/C/C#/Rust/Python） | 新手首次使用 |
| [技术报告 design.md](./design.md) | 架构分层、抽象方案、后端/编解码器矩阵、背压/缓冲池/重连原理、性能方向 | 想理解设计与原理 |
| [接口参考 api.md](./api.md) | 逐签名接口、参数、生命周期与 GPU 借用约定 | 精确调用接口 |

## 快速开始（30 秒）

**C++：**
```cpp
auto dec = modeldeploy::video::VideoDecoder::create({});  // 默认 FFmpeg + Auto 硬件
dec->open("demo.mp4");
modeldeploy::video::VideoFrame vf;
while (dec->read_one_frame(&vf)) { /* vf.image 是 NV12 帧 */ }
```

**Python：**
```python
dec = md.video.VideoDecoder()          # FFmpeg + Auto
dec.open("demo.mp4")
ok, image, pts = dec.read_frame()      # (bool, ImageData, 时间戳毫秒)
```

> 各语言绑定均通过 C API（`md_video_*`）承载，见 [guide.md](./guide.md) 的「跨语言绑定」与 [api.md](./api.md)。

## 构建要点

```bash
cmake -S . -B build -G Ninja -DBUILD_VIDEO=ON -DBUILD_VISION=ON \
      -DENABLE_FFMPEG=ON -DENABLE_GSTREAMER=OFF ...
```

- `BUILD_VIDEO=ON` 必须配 `BUILD_VISION=ON`；FFmpeg 或 GStreamer 至少一个。
- 需要 FFmpeg/GStreamer 开发库，找不到时 `BUILD_VIDEO` 自动关闭。

## 支持矩阵

| 项 | 取值 |
|----|------|
| 后端 | FFmpeg（默认）、GStreamer、Auto |
| 硬件加速 | Auto / None / Cuda / Vaapi / Sophgo |
| 解码硬解 | `h264_cuvid` `hevc_cuvid` `av1_cuvid`、VAAPI、GStreamer `nvcodec` |
| 编码 | `libx264` `x264enc`（软）、`h264_nvenc` `nvh264enc` `vaapih264enc`（硬） |
| 容器 | mp4 / flv / rtmp / rtsp |
| 语言 | C++、C API、C#、Rust、Python（全部解码+编码全功能） |

## 设备侧 NV12 可视化（GPU 就地绘制）

支持「解码 → 设备 NV12 就地绘制 → GPU 直编」的全设备流水线，免 D2H/H2D 往返：

- `VisionProcessorBackend` 提供高层设备接口 `vis_*_nv12`（det/obb/pose/keypoints/hand/ocr/lpr/attr/cls/iseg/sem/depth），
  语义与 CPU `vis_*` 一致（框/骨架/四边形/文本/调色板/半透明/字体），就地写设备 NV12 帧。
- **CUDA 后端**已实现：自研 CUDA 内核（填充矩形/多边形、线段、CJK 位图文本、语义/深度/实例 mask 叠加），
  内置 CJK 位图字库（由 `tools/gen_cjk_font.py` 从 ttf 生成，无需运行时字体/FreeType）。
- C API `md_draw_result` 对设备 NV12 帧自动走此路径；设备后端不可用且回退 CPU 无法绘设备内存时返回 `MD_ERR_NOT_IMPLEMENTED`。
- 文本用内置 16px 位图（`font_size=1` 缩放），`node.font_path`/`save_result` 对设备路径忽略。

**Sophgo(TPU)**:设备接口已声明于 `SophgoProcessorBackend`,bmcv 实现与真机验证待 `.243` 构建容器可用后补（见 AGENTS.md 的 TPU 约定）。

## 构建要点

```bash
cmake -S . -B build -G Ninja -DBUILD_VIDEO=ON -DBUILD_VISION=ON \
      -DENABLE_FFMPEG=ON -DENABLE_GSTREAMER=OFF ...
```

- `BUILD_VIDEO=ON` 必须配 `BUILD_VISION=ON`；FFmpeg 或 GStreamer 至少一个。
- 需要 FFmpeg/GStreamer 开发库，找不到时 `BUILD_VIDEO` 自动关闭。

## 支持矩阵

| 项 | 取值 |
|----|------|
| 后端 | FFmpeg（默认）、GStreamer、Auto |
| 硬件加速 | Auto / None / Cuda / Vaapi / Sophgo |
| 解码硬解 | `h264_cuvid` `hevc_cuvid` `av1_cuvid`、VAAPI、GStreamer `nvcodec` |
| 编码 | `libx264` `x264enc`（软）、`h264_nvenc` `nvh264enc` `vaapih264enc`（硬） |
| 设备绘制 | CUDA 全部 `vis_*_nv12`；Sophgo 待补 |
| 容器 | mp4 / flv / rtmp / rtsp |
| 语言 | C++、C API、C#、Rust、Python（全部解码+编码全功能） |

> 返回 [文档中心](../README.md)
