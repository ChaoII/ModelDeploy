# ModelDeploy 多语言 API

ModelDeploy 核心逻辑全部在 C++ SDK，提供 **C++ / Python / C / C# / Rust** 五种绑定，其余语言是对 C++/C 的薄封装，行为一致。

| 语言 | 文档 | 定位 |
|------|------|------|
| C++ | [cpp](./cpp.md) | 首选，完整功能全部模型/后端 |
| Python | [python](./python.md) | pybind11，科学计算场景 |
| C | [capi](./capi.md) | 嵌入式 / FFI 桥接（`md_*` 前缀） |
| C# | [csharp](./csharp.md) | .NET（`ModelDeploy` 命名空间） |
| Rust | [rust](./rust.md) | FFI 封装 C API |

后端/设备/精度配置在所有语言中保持一致，见 [RuntimeOption 配置](../runtime_option.md)。

## 跨语言模型/类名对照表

五种绑定的模型族类名/构造方式对照。每类模型在各语言下的写法如下（C API 为**统一分发点**，经 `md_model_create` 按 `MDModelKind` 创建）：

| 模型族 | C++ | Python | C API | C# | Rust |
|--------|-----|--------|-------|----|------|
| 目标检测 | `vision::detection::UltralyticsDet` | `modeldeploy.vision.UltralyticsDet` | `MD_MODEL_DETECTION`（经 `md_model_create`） | `Models.DetectionModel` | `UltralyticsDet` |
| 实例分割 / FastSAM | `vision::detection::UltralyticsSeg` / `vision::seg::FastSam` | `modeldeploy.vision.UltralyticsSeg` / `.FastSam` | `MD_MODEL_INSTANCE_SEG` / `MD_MODEL_FASTSAM`（经 `md_model_create`） | `Models.InstanceSegModel` / `Models.FastSamModel` | `UltralyticsSeg` / `FastSam` |
| 语义分割 | `vision::detection::UltralyticsSem` | `modeldeploy.vision.UltralyticsSem` | `MD_MODEL_SEM_SEG`（经 `md_model_create`） | `Models.SemSegModel` | `UltralyticsSem` |
| 深度估计 | `vision::detection::UltralyticsDepth` | `modeldeploy.vision.UltralyticsDepth` | `MD_MODEL_DEPTH`（经 `md_model_create`） | `Models.DepthModel` | `UltralyticsDepth` |
| 姿态 / 关键点 | `vision::detection::UltralyticsPose` / `vision::hand::HandKeypoint` / `vision::landmark::VehicleKeypoint` / `vision::landmark::FaceLandmark` | `modeldeploy.vision.UltralyticsPose` / `.HandKeypoint` / `.landmark.VehicleKeypoint` / `.landmark.FaceLandmark` | `MD_MODEL_POSE` / `MD_MODEL_HAND` / `MD_MODEL_VEHICLE_KEYPOINT` / `MD_MODEL_FACE_LANDMARK`（经 `md_model_create`） | `Models.PoseModel` / `Models.HandModel` / `Models.VehicleKeypointModel` / `Models.FaceLandmarkModel` | `UltralyticsPose` / `HandKeypoint` / `VehicleKeypoint` / `FaceLandmark` |

其余模型族在后续小节逐个补全。
