# ModelDeploy

多后端推理 SDK（OnnxRuntime / TensorRT / MNN / Sophgo TPU），支持检测/分割/姿态/OBB/分类/人脸/OCR/车牌/行人属性/Re-ID/条码二维码/语音(ASR/TTS/VAD)/声纹/文档理解(→Markdown)/跟踪/视频动作识别/NLP 等模型与能力，并提供 C++ / Python / C / C# / Rust 五种绑定。一套代码统一调用四种后端。

> 完整文档见 **文档中心 [docs/README.md](./docs/README.md)**。

## 功能亮点

- **多后端统一 API**：`RuntimeOption` 一键切换 OnnxRuntime / TensorRT / MNN / Sophgo(算能 TPU)
- **AI 视觉**：检测/分割/姿态/OBB/分类/深度/语义分割/人脸/车牌/OCR/行人属性/Re-ID/手势/条码二维码
- **AI 音频**：ASR(SenseVoice)/TTS(Kokoro、Audio8、Qwen3-TTS)/VAD/声纹验证/说话人分段
- **文档理解**：版面分析 + 公式/OCR/表格 → Markdown
- **视频**：解码 + 动作识别(TSN/ST-GCN) + 多目标跟踪
- **NLP**：jieba 分词/分句/关键词/统计 + BERT 文本分类
- **解决方案层**：对象计数/热力图/测速/车位管理/说话人检索/流式 STT/TTS 批处理
- **模型加密**：AES-256-CBC 防模型权重泄露
- **多语言绑定**：C++ / Python / C / C# / Rust

## 快速开始

```bash
git clone https://github.com/ChaoII/ModelDeploy.git && cd ModelDeploy
# Windows 用 "x64 Native Tools Command Prompt for VS 2022"; 推荐 Ninja
cmake -S . -B build -G Ninja -DBUILD_AUDIO=ON -DBUILD_VISION=ON \
      -DBUILD_CAPI=OFF -DBUILD_PYTHON=OFF -DENABLE_MNN=OFF \
      -DENABLE_ORT=ON -DENABLE_TRT=OFF -DWITH_GPU=OFF
cmake --build build --config Release --parallel
cmake --install build
```

安装后生成 `install/`（`include/` + `lib/`）。详细编译选项与第一个程序见 [快速开始](./docs/quickstart.md)。

### 拉取测试数据

```bash
# Windows
powershell -ExecutionPolicy Bypass -File tools/fetch_test_data.ps1
# Linux/macOS
bash tools/fetch_test_data.sh
```

### 最小检测示例

```cpp
#include "modeldeploy/vision.h"
int main() {
    modeldeploy::RuntimeOption option;
    option.use_ort_backend(); option.use_cpu();
    auto det = modeldeploy::vision::detection::UltralyticsDet("yolo11n.onnx", option);
    det.get_preprocessor().set_size({640, 640});
    auto img = modeldeploy::ImageData::imread("test.jpg");
    std::vector<modeldeploy::vision::DetectionResult> result;
    det.predict(img, &result);
    return 0;
}
```

## 支持矩阵

| 后端 | 格式 | CPU | CUDA | OpenCL | TPU |
|------|------|-----|------|--------|-----|
| OnnxRuntime | `.onnx` | ✅ | ✅ | ✅ | — |
| TensorRT | `.engine`/`.onnx` | — | ✅ | — | — |
| MNN | `.mnn` | ✅ | ✅ | ✅ | — |
| Sophgo | `.bmodel` | — | — | — | ✅ (BM1688/CV186X) |

## 路线图

- [x] 重构 `Tensor` 支持 CUDA
- [x] Python / C# 绑定
- [x] Pipeline DAG 编排、视频解码、多目标跟踪、动作识别、文档理解、Re-ID、声纹、解决方案层
- [x] 多后端（ORT/TRT/MNN/Sophgo）统一 API + 模型加密
- [ ] 更多 CUDA 预处理函数

## 更多文档

| 主题 | 文档 |
|------|------|
| 文档中心(总导航) | [docs/README.md](./docs/README.md) |
| 快速开始(构建/首个程序) | [docs/quickstart.md](./docs/quickstart.md) |
| 架构 | [docs/architecture.md](./docs/architecture.md) |
| 后端详解 | [docs/backends.md](./docs/backends.md) |
| 模型转换/量化 | [docs/conversion.md](./docs/conversion.md) |
| 模型详解 | [docs/models.md](./docs/models.md) |
| 预处理 | [docs/preprocess.md](./docs/preprocess.md) |
| 性能优化 | [docs/performance.md](./docs/performance.md) |
| 多语言 API | [docs/api/README.md](./docs/api/README.md) |
| 模型加密 | [docs/encryption.md](./docs/encryption.md) |
| 多线程 | [docs/multi_thread.md](./docs/multi_thread.md) |
| 示例 | [examples/EXAMPLES.md](./examples/EXAMPLES.md) |
