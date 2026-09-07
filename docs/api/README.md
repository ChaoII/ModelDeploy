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

其余模型族在后续小节逐个补全。
