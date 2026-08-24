# C# 绑定（.NET）

C# 绑定封装 C API，命名空间 `ModelDeploy`。解决方案见 `csharp/ModelDeploy.sln`。

```csharp
using ModelDeploy;

var option = new MDRuntimeOption();
option.UseOrtBackend();
option.UseCpu();

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
