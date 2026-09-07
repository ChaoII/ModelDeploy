# C# 绑定（.NET）

C# 绑定封装 C API，命名空间 `ModelDeploy`（模型类在 `ModelDeploy.Models`）。解决方案见 `csharp/ModelDeploy.sln`。

## 1. 引入

```csharp
using ModelDeploy;
using ModelDeploy.Models;
```

## 2. RuntimeOption 与检测示例

C# 没有 `UseCpu()`；改 CPU 用 `SetDevice(Device.CPU)`。后端用 `UseOrt()`/`UseMnn()`/`UseTrt()`/`UseSophgo()`/`UseNcnn()`，设备用 `SetDevice(Device dev, int deviceId = 0)`：

```csharp
using ModelDeploy;
using ModelDeploy.Models;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);
option.SetCpuThreads(4);

// 设备（OPENCL/VULKAN 需显式 MNN 后端，否则 fail-closed）
// option.UseMnn().SetDevice(Device.OPENCL, 0);
// option.SetDevice(Device.VULKAN, 0);
// GPU：option.UseOrt().SetDevice(Device.GPU, 0);

var model = new DetectionModel("yolo11n.onnx", option);
model.SetInputSize(640, 640);

using var img = VisionImage.Read("test.jpg");
using var pred = model.Predict(img);
foreach (var r in pred) {
    Console.WriteLine($"{r.LabelId} {r.Score} {r.Box}");
}
```

## 3. 目标检测（`DetectionModel`）

检测模型类 `ModelDeploy.Models.DetectionModel`（对应 `MD_MODEL_DETECTION`）。结果类型 `ModelDeploy.Results.DetectionResult`（属性 `Box: RectF{X,Y,Width,Height}`、`LabelId: int`、`Score: float`）；`Predict` 返回 `Prediction<DetectionResult>`（实现 `IReadOnlyList<T>`，`using`/`Dispose` 释放底层结果句柄）。

```csharp
using System;
using System.Collections.Generic;
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var det = new DetectionModel("yolo11n.onnx", option);

// 预处理/后处理参数（类型化 setter，编译期检查；通用 setter 为 SetParam(name, value)，见下）
det.SetInputSize(640, 640);      // letterbox 输入尺寸
det.SetConfThreshold(0.25);      // 置信度阈值（默认 0.25）
det.SetNmsThreshold(0.45);       // NMS IoU 阈值（默认 0.5）
// 自省：本 kind 支持的参数名与类型（'I'/'D'/'B'/'S'）
Console.WriteLine(string.Join(", ", det.ParamNames()));

using var img = VisionImage.Read("test.jpg");

// 单图推理：Predict 返回 Prediction<DetectionResult>（可 foreach / 索引 / .Count）
using var pred = det.Predict(img);
foreach (var r in pred) {
    Console.WriteLine($"{r.LabelId} {r.Score:F3} ({r.Box.X},{r.Box.Y},{r.Box.Width},{r.Box.Height})");
}

// 可视化：pred.Draw 直达 C++ vis_det（句柄直达原始结果；绘制到任意 VisionImage 画布）
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions {
    Threshold = 0.25,
    LabelMap = new Dictionary<int, string> { { 0, "person" }, { 1, "bicycle" }, { 2, "car" } },
    FontSize = 14,
    Alpha = 0.3,
});
canvas.Save("det_vis.jpg");

// 批量推理：PredictBatch 按图返回 IReadOnlyList<DetectionResult[]>
using var img2 = VisionImage.Read("bus.jpg");
var batch = det.PredictBatch(new[] { img, img2 });
for (int g = 0; g < batch.Count; g++) {
    Console.WriteLine($"image {g}: {batch[g].Length} objects");
}

// 多线程：Clone() 深拷贝独立实例（每线程持有一个，互不干扰）
using var det2 = det.Clone();
```

## 运行示例

```bash
cd csharp
dotnet build ModelDeployExample/ModelDeployExample.csproj -c Debug
cd ModelDeployExample/bin/Debug/net9.0
./ModelDeployExample.exe
```

`Program.cs` 的 `Main` 依次运行：检测 → 图像工具 → 分类 → 姿态 → OCR → InsightFace → ASR(SenseVoice) → TTS(Kokoro)，各结果打印到控制台，检测/图像落盘，TTS 落盘 `output.wav`。

## 主要项目

- `ModelDeploy` — C# 绑定库
- `ModelDeployExample` — 示例
- `ModelDeployUnitTest` — 单元测试

## TTS（Kokoro）

命名空间 `ModelDeploy.Models`，`KokoroModel` 返回 `TtsResult { Audio (float[]), SampleRate }`，另有 `SaveWav(result, path)` 落盘。

```csharp
using ModelDeploy;
using ModelDeploy.Models;

var opt = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

// Kokoro：24kHz。modelPath 格式: model.onnx|tokens.txt|lex_en.txt|lex_zh.txt|voices.bin|jieba_dir|norm_dir
var kokoro = new KokoroModel("kokoro.onnx|tokens.txt|...", opt);
kokoro.SaveWav(kokoro.Predict("你好，世界。", "zf_001"), "kokoro.wav");

// 统一流式：PredictStream 逐块回调 onChunk(samples, progress)，同时返回整段音频
// chunkFrames <= 0 等价一次性合成（单次回调整段）
var r1 = kokoro.PredictStream("你好，世界。", "zf_001", 1.0f, 120,
    (samples, progress) => Console.WriteLine($"progress={progress:P0} chunk={samples.Length}"));
```

`KokoroModel` 提供 `Predict(text, voice, speed=1.0f)`、`PredictStream(text, voice, speed, chunkFrames, onChunk)` 与 `Clone()`（深拷贝实例，多线程用）。

## 图像原始字节（`VisionImage`）

`VisionImage` 提供原生格式的宿主字节读取（`Type`/`Width`/`Height` 见 `VisionImage` 属性）：

```csharp
byte[] raw = img.ToNativeBytes();     // 整幅原生连续字节；GPU 设备帧自动回读主机
byte[] y   = img.GetPlaneBytes(0);    // 第 0 平面（NV12 的 Y）
byte[] uv  = img.GetPlaneBytes(1);    // 第 1 平面（NV12 的 UV）
```

- `ToNativeBytes()` 布局随 `Type` 而定：BGR24=`Width*Height*3`；NV12=`w*h + w*h/2`；I420=`w*h + w*h/4 + w*h/4`。
- 平面越界抛 `InvalidOperationException`；不支持的类型/设备（如 TPU）抛 `NotSupportedException`。
- 两方法立即将原生缓冲拷贝为托管 `byte[]`（指针在下次调用即失效）。
- 与既有 `ToByteArray()`（编码为 **BMP** 文件字节，含 54 字节头）不同，上面拿到的是**原始像素**，可直接交给其它编解码接口。

## 视频编解码（`Video`）

`ModelDeploy` 命名空间下的视频封装等价于 C++ `modeldeploy::video`（解码+编码全功能，经 C API 承载），类均实现 `IDisposable`，析构自动释放底层句柄。

| 类型 | 说明 |
|------|------|
| `VideoConfig` | 解码/编码配置（`set` 属性，可用对象初始化器） |
| `VideoCapabilities` | 能力探测（FFmpeg/GStreamer/硬解硬编列表） |
| `VideoDecoder` | 解码器：同步 `ReadFrame` + 异步回调 + 状态/统计 |
| `VideoEncoder` | 编码器：`Encode` / `EncodeAsync` / `StartAsync` |
| `VideoStats` | 编解码统计 |

枚举：`VideoCodecBackend{Auto,FFmpeg,GStreamer}`、`VideoHwAccel{Auto,None,Cuda,Vaapi,Sophgo}`、`VideoBackpressure{Block,Drop,OverwriteOldest}`、`VideoState{Idle,Opening,Running,Reconnecting,Eof,Error,Closed}`。

```csharp
using ModelDeploy;

// 能力探测
var cap = new VideoCapabilities();
Console.WriteLine($"ffmpeg={cap.FfmpegAvailable} hw_encode={string.Join(",", cap.HwEncoders)}");

// 配置 + 解码
using var dec = new VideoDecoder(new VideoConfig {
    Backend = VideoCodecBackend.FFmpeg,
    HwAccel = VideoHwAccel.Cuda,
    DeviceOnly = false,               // true = GPU 设备内存直通
});
dec.Open("demo.mp4");                 // 或 rtsp://...
var (image, pts) = dec.ReadFrame();   // 返回 (VisionImage NV12帧, 时间戳毫秒)；空帧表示 EOF
while (image != null) {
    Console.WriteLine($"{pts}ms {image.Width}x{image.Height}");
    (image, pts) = dec.ReadFrame();
}
var st = dec.Stats;                   // VideoStats
```

**异步解码（回调推送）**：`dec.SetCallback((image, pts) => { ... })` 注册交付回调（帧为自有 `VisionImage`，用后自动释放），再 `dec.Start()` / `dec.Stop()`。

**编码**：
```csharp
using var enc = new VideoEncoder(new VideoConfig {
    Codec = "libx264", BitrateKbps = 2000, Gop = 30, Fps = 25,
});
enc.Open("out.mp4", 1280, 720, 25);
enc.Encode(img, ptsMs);               // img 生命周期须覆盖本次调用
enc.Close();                          // 必须 Close，mp4 尾部索引(moov)在此写盘
```

设备显存（`DeviceOnly` 解码 / `GpuDirectInput` 编码）零拷贝直通。详见 [视频接口总览](../video/api.md)。
