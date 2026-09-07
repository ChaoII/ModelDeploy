# ModelDeploy

多后端推理 SDK（OnnxRuntime / TensorRT / MNN / **ncnn** / Sophgo TPU），支持检测/分割/姿态/OBB/分类/人脸/OCR/车牌/行人属性/Re-ID/条码二维码/语音(ASR/TTS/VAD)/声纹/文档理解(→Markdown)/跟踪/视频动作识别/NLP 等模型与能力，并提供 **C++ / Python / C / C# / Rust** 五种绑定。一套代码统一调用五种后端。

> 完整文档见 **文档中心 [docs/README.md](./docs/README.md)**。

## 功能亮点（能力）

**推理后端**
- **多后端统一 API**：`RuntimeOption` 一键切换 OnnxRuntime / TensorRT / MNN / ncnn / Sophgo(算能 TPU)
- **ncnn 后端（新增）**：CPU + **Vulkan**，支持 ultralytics YOLO 全系（det/cls/obb/pose/seg/sem/depth），配置时按平台从模型库自动下载接入
- **模型加密**：AES-256-GCM 防模型权重泄露

**AI 视觉**
- 检测 / 实例分割 / 语义分割 / 深度估计 / 姿态估计 / OBB 旋转框 / 分类 / 人脸(检测/识别/分析/年龄性别/防伪) / 车牌 / OCR / 文档理解(版面+公式+表格→Markdown) / 行人属性 / Re-ID / 手势 / 手部关键点 / 条码二维码

**AI 音频**
- ASR(SenseVoice/AAsr/流式) / TTS(Kokoro) / VAD / 声纹验证 / 说话人分段与检索

**视频编解码**
- **解码**：本地文件 / RTSP / RTMP → `VideoFrame`(NV12)，CPU 软解 + **CUDA / VAAPI / QSV / Sophgo** 硬解（`Auto` 自动回退）
- **编码**：帧 → mp4 / flv / rtmp / rtsp，软编（libx264/x264enc）+ 硬编（h264_nvenc/nvh264enc/vaapih264enc）
- **GPU 直通**：`device_only` 解码设备内存直通、`gpu_direct_input` 编码显存直编、设备 NV12 就地可视化（免主机往返）
- **工程化**：有界背压异步队列、帧缓冲池（零拷贝复用）、断流自动重连、状态/统计可观测
- 后端：FFmpeg（默认）/ GStreamer / Auto

**其他**
- **视频动作识别**(TSN/ST-GCN) + **多目标跟踪**(Byte/BotSort/StrongSort) + **行人 Re-ID**
- **NLP**：jieba 分词/分句/关键词/统计 + BERT 文本分类
- **解决方案层**：对象计数/热力图/测速/车位管理/距离估计/打码/裁剪/健身/说话人/流式 STT/TTS 批处理
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

### 后端 × 设备

| 后端 | 模型格式 | CPU | CUDA | OpenCL | Vulkan | TPU |
|------|---------|-----|------|--------|--------|-----|
| OnnxRuntime | `.onnx` | ✅ | ✅ | ✅ | — | — |
| TensorRT | `.engine`/`.onnx` | — | ✅ | — | — | — |
| MNN | `.mnn` | ✅ | — | ✅ | ✅ | — |
| **ncnn** | `.param`/`.bin` | ✅ | — | — | ✅ | — |
| Sophgo | `.bmodel` | — | — | — | — | ✅ (BM1688/CV186X) |

### 视频编解码

| 项 | 取值 |
|----|------|
| 后端 | FFmpeg（默认）、GStreamer、Auto |
| 硬件加速 | Auto / None / CUDA / VAAPI / **QSV** / Sophgo |
| 解码硬解 | `h264/hevc/av1_cuvid`、VAAPI、GStreamer `nvcodec` |
| 编码 | 软：`libx264`/`x264enc`；硬：`h264_nvenc`/`nvh264enc`/`vaapih264enc` |
| 容器 | mp4 / flv / rtmp / rtsp |
| 语言 | C++ / C API / C# / Rust / Python（解码+编码全功能） |

## 路线图

> 完整版本历史与发布流程见 [docs/release.md](./docs/release.md)。

### 已完成里程碑

**核心推理底座**
- [x] `Tensor` 重构：CUDA 支持、零拷贝外部内存、多线程 `clone()` 并发
- [x] 预处理流水线：CPU / CUDA(自研核) / BMCV(TPU) / Vulkan 硬件加速、`ImageData` 统一抽象
- [x] Pipeline DAG 编排 + 前处理 → 推理 → 后处理解耦

**推理后端（统一 API）**
- [x] 五后端统一：OnnxRuntime / TensorRT / MNN / **ncnn** / Sophgo(算能 TPU)，一套 `RuntimeOption` 切换
- [x] **ncnn（CPU + Vulkan，YOLO 全系）**、MNN（CPU/OpenCL/Vulkan）
- [x] **ncnn 覆盖非 YOLO 模型**：人脸（InsightFace/Scrfd/SeetaFace）、OCR（DBDetector/Recognizer/Classifier/PaddleOCR）、分类、车牌 LPR（CPU + Vulkan）
- [x] TensorRT engine 在线构建/缓存 + 动态 shape；ORT 内嵌 TRT EP
- [x] Sophgo：`.bmodel` 转换(F16/INT8)、BMCV 设备端零拷贝、混合量化(qtable)
- [x] **模型加密**：AES-256-GCM 权重防泄露

**AI 视觉**
- [x] 检测 / 实例分割 / 语义分割 / 深度估计 / 姿态估计 / OBB / 分类 / FastSAM 提示词
- [x] 人脸全套（检测/识别/分析/年龄性别/防伪/Re-ID）、车牌、OCR、文档理解(→Markdown)
- [x] 行人属性、手部关键点、手势、条码/二维码

**AI 音频 / 语音**
- [x] ASR（SenseVoice/AAsr/流式）、TTS（Kokoro）、VAD、声纹验证、说话人分段/检索

**视频编解码**
- [x] FFmpeg / GStreamer 双后端解码 + 编码（mp4/flv/rtmp/rtsp）
- [x] 硬解/硬编：CUDA / VAAPI / **QSV(Intel)** / Sophgo，`Auto` 自动回退
- [x] GPU 直通：`device_only` 解码、`gpu_direct_input` 编码、设备 NV12 就地可视化（免主机往返）
- [x] 工程化：有界背压、帧缓冲池(零拷贝复用)、断流重连、状态/统计可观测
- [x] 视频动作识别（TSN/ST-GCN）+ 多目标跟踪（Byte/BotSort/StrongSort）

**NLP**
- [x] jieba 分词/分句/关键词/统计、BERT 文本分类、**WeText 数字归一化 ITN（v2.6.0）**

**解决方案 / 工具 / 服务化**
- [x] 解决方案层：计数/热力/测速/车位/距离/打码/裁剪/健身/说话人/流式 STT/TTS 批处理
- [x] 后处理工具底座：标注/检测容器/指标/切片/平滑/区域判断
- [x] Triton 推理服务部署（preprocess → pipeline → postprocess）

**多语言绑定**
- [x] C++ / Python / C / C# / Rust 五语言绑定（视频编解码全功能覆盖）

**构建 / CI**
- [x] **aarch64（Linux ARM64）CI**：x86_64 runner 交叉编译 SDK + Python wheel（cibuildwheel）

### 未来路线

按「中期 → 远期」分层，越靠前优先落地（近期事项已全部落地/移出，暂无近期条目）。

**中期（Mid）**
- [ ] **更多 NPU 后端**：Rockchip RKNN（RK3588）、Qualcomm QNN/Snapdragon、华为昇腾 CANN（国产化）——沿用下载式接入范式
- [ ] **流式/异步推理**：一期**仅 C++**——`AsyncModel` 异步投递壳（有界背压队列 + worker 线程 + `std::future`/回调，复用解码头基建的范式），跨帧 stateful 算子；**其它语言绑定后续再议**（文档注明）
- [ ] **服务化增强（嵌入式推理网关）**：内置 HTTP / gRPC 推理端点、模型仓库热更新、多 worker 调度、健康/指标/TLS——**按 llama.cpp server 级工程化完备落地，覆盖各类部署场景**（非精简版）
- [ ] **视频编解码深化**：AV1 硬编、HEVC 遍历、GPU 多路编码、更高吞吐基准

**远期（Long）**
- [ ] **WASM / Web + iOS / Android** 端到端部署模板（边缘全平台覆盖）
- [ ] **模型管理平台**：模型仓库 + 版本 + 分片 + 注册，一键下发设备
- [ ] **边缘–云协同调度**：分布式推理、带宽感知分流
- [ ] **算子层收敛**：自研统一算子抽象 + **设备端算子统一注册表**，屏蔽各后端差异，降低新后端接入成本
- [ ] **生成式大模型（LLM/VLM）**：接入 **llama.cpp**（GGUF/int4/CMake 集成），跑通 Qwen / DeepSeek 等小模型 chat + VLM（如 Qwen2.5-VL）；标的**可选、按需拉前**；ASR/TTS 沿用现有 ORT 专用后端，不纳入 llama.cpp 范围
- [ ] **视觉大模型（VLM）融合**：Qwen2-VL 等视觉语言模型**接入统一推理 API**（走现有 ONNX 视觉模型路线，区别于上面的 llama.cpp 生成式路线）
- [ ] **低代码可视化编排**：拖拽式流程构建，降低集成门槛

## 已知问题 / 当前 SDK 痛点

以下为当前 SDK 存在的已知问题与限制，均在文档中标记、多数为环境/平台固有，已明确处置方案。

1. **`baseline_compare` 跨平台差异（GCC CI 需重生成基线）**：GCC `-O3 -mavx2 -mfma` 的 FMA 收缩使 `cal_letter_box_param` 的 `pad_h` 比精确值高 1 ULP（80.000007629），fused 预处理核的 `src_yf<0`/`src_xf<0` 严格越界判断把首行/左缘列误写为 pad，导致输入张量逐像元差最大 0.557、翻转临界检测。**与 onnxruntime 打包/重构无关**（同机双平台同走 AVX2 核），属预存在缺陷、生产影响趋近于零，判定**不修**；`-ffp-contract=off` 可修复，CI 在 GCC 平台需重新生成基线。

2. **GStreamer–CUDA 进程级限制**：GStreamer 一旦在进程内创建 CUDA context，其 nvcodec 会对本进程余下所有 `nvh264enc` 管道**全局注册 CUDA 缓冲**，致后续 mp4 缺 moov、软解打开失败——该进程级状态不可逆。故带 `[gst-cuda]` 标签的用例**必须单独进程**运行（见 AGENTS.md）。属 GStreamer 行为，非 SDK 逻辑缺陷。

3. **ncnn 输出加载时常为 `[-1]`**：导出模型的 `.param` 常未声明输入 shape（blob hint 空），dummy probe 无输入形状可跑，输出列只能保持 `[-1]`。属模型导出固有局限，需导出时声明输入 shape 才能显示真实值。

4. **MNN/ncnn 动态输出形状走 probe**：这两类后端加载时无原生输出 shape，由加载期零数据 probe 填充（用户已定保持方案，不切换原生形状展示）。

5. **`build_py`（`BUILD_CAPI=OFF`）的 `demo_image_from_base64` 链接失败**：该 demo 使用 C API 符号，而 build_py 关闭了 C API，属预存在问题（与本次重构无关）；需开启 `BUILD_CAPI=ON` 或移除该 demo。

6. **Sophgo 后端无法本地编译验证**：仅能在 `172.168.100.243` 的 `tpuc_dev` 容器交叉编译、`.70`（BM1688）真机验证；改动需走该远程链路。

7. **Sophgo INT8 量化**：det/pose/seg 需**解码头 qtable**（INT8 backbone + 检测头输出链 F16）才可 INT8（cmodel cos≈0.99）；OBB 的 INT8 仍失效（坐标失真、cos≈0.6），**建议用 F16**；cls/sem/depth 可直接 INT8。详见 [docs/backends.md](./docs/backends.md)。

8. **Windows 构建环境**：MSVC 编译需在 "x64 Native Tools Command Prompt"（含 `vcvars64` include/lib 路径）下进行；模型加密基于 mbedTLS（git submodule），构建前需 `git submodule update --init --recursive`，不再需要安装 OpenSSL。

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
| **视频编解码** | [docs/video/README.md](./docs/video/README.md) |
| 多语言 API | [docs/api/README.md](./docs/api/README.md) |
| 模型加密 | [docs/encryption.md](./docs/encryption.md) |
| 多线程 | [docs/multi_thread.md](./docs/multi_thread.md) |
| 示例 | [examples/EXAMPLES.md](./examples/EXAMPLES.md) |
