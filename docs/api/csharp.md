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

## 4. 实例分割（`InstanceSegModel`）

实例分割模型类 `ModelDeploy.Models.InstanceSegModel`（对应 `MD_MODEL_INSTANCE_SEG`）。结果类型 `ModelDeploy.Results.InstanceSegResult`（属性 `Box: RectF`、`LabelId: int`、`Score: float`、`Mask: byte[]` + `MaskWidth`/`MaskHeight`；`Mask` 为 uint8 0/1，行主序 `MaskHeight * MaskWidth`）。

```csharp
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var seg = new InstanceSegModel("yolo11n-seg.onnx", option);
seg.SetInputSize(640, 640);      // letterbox 输入尺寸
seg.SetConfThreshold(0.25);      // 置信度阈值（默认 0.25）
seg.SetNmsThreshold(0.45);       // NMS IoU 阈值（默认 0.5）
seg.SetMaskThreshold(0.5);       // 掩码二值化阈值（默认 0.5）

using var img = VisionImage.Read("test.jpg");

// 单图推理：Prediction<InstanceSegResult>（foreach / 索引 / .Count）
using var pred = seg.Predict(img);
foreach (var r in pred) {
    Console.WriteLine($"{r.LabelId} {r.Score:F3} ({r.Box.X},{r.Box.Y},{r.Box.Width},{r.Box.Height})");
    // 掩码逐像元读取：r.Mask[y * r.MaskWidth + x]
    Console.WriteLine($"mask {r.MaskWidth}x{r.MaskHeight}, bytes={r.Mask.Length}");
}

// 批量推理：按图返回 IReadOnlyList<InstanceSegResult[]>
using var img2 = VisionImage.Read("bus.jpg");
var batch = seg.PredictBatch(new[] { img, img2 });
for (int g = 0; g < batch.Count; g++) {
    Console.WriteLine($"image {g}: {batch[g].Length} instances");
}

// 可视化：pred.Draw 直达 C++ vis_iseg
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions { Threshold = 0.5 });
canvas.Save("iseg_vis.jpg");

// 多线程：Clone() 深拷贝独立实例
using var seg2 = seg.Clone();
```

## 5. FastSAM（`FastSamModel`）

FastSAM 结果与实例分割同构（`InstanceSegResult`）。`PredictWithPrompts` 在全量结果上按提示过滤实例，**不重跑网络**；提示为空等价全图 `Predict`。

```csharp
using ModelDeploy;
using ModelDeploy.Models;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var sam = new FastSamModel("fastsam-s.onnx", option);
sam.SetInputSize(1024, 1024);    // 默认 640x640；官方 FastSAM-s 常配 1024x1024
sam.SetConfThreshold(0.30);      // 默认 conf 0.30 / nms 0.5 / mask 0.5
sam.SetNmsThreshold(0.40);
sam.SetMaskThreshold(0.5);

using var img = VisionImage.Read("test.jpg");

// 全图（Everything）分割：Prediction<InstanceSegResult>，用法同 InstanceSegModel
using var pred = sam.Predict(img);

// 提示过滤：bboxes 为 [x,y,w,h,...]（原图像素）、points 为 [x,y,...]、
// labels 逐点（1=前景保留, 0=背景剔除）。返回 IReadOnlyList<InstanceSegResult>
var prompted = sam.PredictWithPrompts(img,
    new float[] { 100, 80, 220, 180 },
    new float[] { 150, 130 },
    new int[] { 1 });
Console.WriteLine($"prompted: {prompted.Count}");
```

## 6. 语义分割（`SemSegModel`）

语义分割模型（`yolo26n-sem` 等，cityscapes 19 类）。结果类型 `ModelDeploy.Results.SemSegResult`（属性 `Labels: byte[]`、`Width`/`Height`、`NumClasses: int`；`Labels` 为每像素类别索引 `[0, NumClasses)`，行主序）。该模型无阈值 setter，也无 `PredictBatch`（C API 对 `MD_MODEL_SEM_SEG` 无参数、结果为整图单值）。

```csharp
using ModelDeploy;
using ModelDeploy.Models;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var sem = new SemSegModel("yolo26n-sem.onnx", option);
sem.SetInputSize(640, 640);      // 输入尺寸可调；无其它参数（后处理 argmax）

using var img = VisionImage.Read("test.jpg");
using var pred = sem.Predict(img);
var r = pred[0];
// 掩码/标签逐像元读取：r.Labels[y * r.Width + x]
Console.WriteLine($"sem {r.Width}x{r.Height} classes={r.NumClasses} labels={r.Labels.Length}");

// 可视化：pred.Draw 直达 C++ vis_sem（cityscapes 调色板叠加）
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions { Alpha = 0.5 });
canvas.Save("sem_vis.jpg");
```

## 7. 深度估计（`DepthModel`）

深度估计模型（`yolo26n-depth` 等）。结果类型 `ModelDeploy.Results.DepthResult`（属性 `Depth: float[]`、`Width`/`Height`；每像素深度单位**米**，log 输出已 `exp` 还原，行主序）。该模型无阈值 setter，也无 `PredictBatch`（同上，整图单值结果）。

```csharp
using System;
using ModelDeploy;
using ModelDeploy.Models;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var dep = new DepthModel("yolo26n-depth.onnx", option);
dep.SetInputSize(640, 640);

using var img = VisionImage.Read("test.jpg");
using var pred = dep.Predict(img);
var r = pred[0];
// 深度逐像元读取：r.Depth[y * r.Width + x]（米）
float near = float.MaxValue, far = float.MinValue;
foreach (var d in r.Depth) { near = Math.Min(near, d); far = Math.Max(far, d); }
Console.WriteLine($"depth {r.Width}x{r.Height} range=[{near:F2}, {far:F2}] m");

// 可视化：pred.Draw 直达 C++ vis_depth（JET 伪彩）
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions());
canvas.Save("depth_vis.jpg");
```

## 8. 姿态与关键点族（`PoseModel` / `HandModel` / `VehicleKeypointModel` / `FaceLandmarkModel`）

四个模型类（`ModelDeploy.Models`，分别对应 `MD_MODEL_POSE` / `MD_MODEL_HAND` / `MD_MODEL_VEHICLE_KEYPOINT` / `MD_MODEL_FACE_LANDMARK`）结果类型统一为 `ModelDeploy.Results.PoseResult`（属性 `Box: RectF`、`Score: float`、`KeyPoints: Point3F[]`，`Point3F` 含 `X/Y/Z`，`Z` 为关键点置信度；无 `LabelId`）。`PoseModel`（COCO 17 点人体骨架）、`HandModel`（21 点手部）、`VehicleKeypointModel`（4 车轮关键点）支持 `SetConfThreshold` / `SetNmsThreshold` / `SetKeypointsNum`；`FaceLandmarkModel`（InsightFace 2d106 面部 106 点，`Z` 恒为 0）无阈值 setter（输入须为人脸裁剪图）。

```csharp
using System;
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var pose = new PoseModel("yolo11n-pose.onnx", option);
using var hand = new HandModel("hand.onnx", option);
using var vehicle = new VehicleKeypointModel("vehicle.onnx", option);
using var face = new FaceLandmarkModel("face_landmark.onnx", option);

// 预处理/后处理参数（FaceLandmarkModel 无参数 setter；HandModel 构造默认 21 点、
// VehicleKeypointModel 构造默认 4 点，均可 SetKeypointsNum 覆盖）
pose.SetInputSize(640, 640);      // letterbox 输入尺寸
pose.SetConfThreshold(0.30);      // 置信度阈值（默认 0.30）
pose.SetNmsThreshold(0.45);       // NMS IoU 阈值（默认 0.5）
pose.SetKeypointsNum(17);         // 关键点数（默认 17，须与模型输出一致）
hand.SetKeypointsNum(21);
vehicle.SetKeypointsNum(4);

using var img = VisionImage.Read("test.jpg");

// 单图推理：Predict 返回 Prediction<PoseResult>（foreach / 索引 / .Count）
using var pred = pose.Predict(img);
foreach (var r in pred) {
    Console.WriteLine($"{r.Score:F3} ({r.Box.X},{r.Box.Y},{r.Box.Width},{r.Box.Height}) kps={r.KeyPoints.Length}");
    foreach (var kp in r.KeyPoints) {
        Console.WriteLine($"  ({kp.X},{kp.Y},{kp.Z})");
    }
}
// 面部 Landmark：输入人脸裁剪图 -> 单元素结果（106 点 Z=0，Box 为整图、Score=1.0）
using var crop = VisionImage.Read("face_crop.jpg");
using var fpred = face.Predict(crop);

// 批量推理：PredictBatch 按图返回 IReadOnlyList<PoseResult[]>
using var img2 = VisionImage.Read("bus.jpg");
var batch = pose.PredictBatch(new[] { img, img2 });
for (int g = 0; g < batch.Count; g++) {
    Console.WriteLine($"image {g}: {batch[g].Length} persons");
}

// 可视化：pred.Draw 直达 C++ vis_pose（COCO 骨架连线）
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions { Threshold = 0.30, Alpha = 0.3 });
canvas.Save("pose_vis.jpg");

// 多线程：Clone() 深拷贝独立实例
using var pose2 = pose.Clone();
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
