# C++ 绑定

首选，完整功能，支持全部模型与后端。核心逻辑全在 C++ SDK。

```cpp
#include "modeldeploy/vision.h"

modeldeploy::RuntimeOption option;
option.use_ort_backend();
option.use_cpu();

auto det = modeldeploy::vision::detection::UltralyticsDet("yolo11n.onnx", option);
auto img = modeldeploy::ImageData::imread("test.jpg");
std::vector<modeldeploy::vision::DetectionResult> result;
det.predict(img, &result);
```

编译链接：见 [快速开始](../quickstart.md#3-编写第一个检测程序)。
