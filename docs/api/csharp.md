# C# 绑定（.NET）

C# 绑定封装 C API，命名空间 `ModelDeploy`。解决方案见 `csharp/ModelDeploy.sln`。

```csharp
using ModelDeploy;

var option = new MDRuntimeOption();
option.UseOrtBackend();
option.UseCpu();

// 设备（OPENCL/VULKAN 需显式 MNN 后端，否则 fail-closed）
option.UseMnn().SetDevice(Device.OPENCL, 0);
option.SetDevice(Device.VULKAN, 0);

var model = new MDDetectionModel("yolo11n.onnx", option);
model.SetInputSize(640, 640);

var results = model.Predict(img);
foreach (var r in results) {
    Console.WriteLine($"{r.LabelId} {r.Score} {r.Box}");
}
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
