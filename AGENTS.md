# ModelDeploy — Agent 指南

## 沟通语言

- **始终用中文回答**，除非用户明确要求其它语言。

## 快速开始

```bash
# CPU 构建（推荐 Ninja，MSVC 需要 x64 Native Tools 命令提示符）
cmake -S . -B build -G Ninja -DBUILD_AUDIO=ON -DBUILD_VISION=ON -DBUILD_CAPI=OFF -DBUILD_PYTHON=OFF -DENABLE_MNN=OFF -DENABLE_ORT=ON -DENABLE_TRT=OFF -DWITH_GPU=OFF
cmake --build build --config Release --parallel
cmake --install build
```

## Python wheel

```bash
pip install build
python -m build
# 构建后生成 .pyi 存根：
pybind11-stubgen modeldeploy
```

## 测试（Catch2，单二进制）

```bash
cmake -S . -B build -G Ninja -DBUILD_TESTS=ON ...
cmake --build build
cd build && ctest -C Release --output-on-failure
# 或直接用 Catch2 标签运行二进制：
./test_modeldeploy          # 全部
./test_modeldeploy [core]   # 仅核心
./test_modeldeploy ~[gpu]   # 排除 GPU
```

设备相关用例标签分组（沿用 `~[gpu]` 约定）：
- `[gpu]`/`[gst-cuda]`：需真实 GPU/CUDA 设备，CPU-only CI 默认排除。
- `[opencl]`/`[vulkan]`：需真实 OpenCL/Vulkan 设备，与 `[gpu]` 同等待遇，CPU-only CI 默认排除（`ctest ... -E "opencl|vulkan"`），仅在带相应设备的 GPU 任务/独进程运行。
- 用例示例见 `tests/test_mnn_device.cpp`（MNN OpenCL/Vulkan smoke）。

视频编解码测试分组（`[video]` 套件，沿用 `~[gpu]` 约定）：
- 常规（软编/软解 + 无 CUDA 互操的硬编）：`test_modeldeploy "[video]~[gpu]"`。
- **GStreamer CUDA 互操隔离**：GStreamer 一旦在进程内创建 CUDA context（`gst_cuda_context_new` / CUDA memory），其 nvcodec 会对本进程**余下所有** `nvh264enc` 管道全局注册 CUDA 缓冲，致后续 mp4 缺 moov、软解打开失败——该进程级状态**不可逆**。故带 `[gst-cuda]` 标签的用例（GStreamer 解码 device_only GPU + CUDA memory 直编，共 3 用例）**必须单独进程**运行：
  - `test_modeldeploy "[video][gpu]~[gst-cuda]"`（GPU 非 CUDA 互操部分）
  - `test_modeldeploy "[video][gst-cuda]"`（GStreamer CUDA 互操，独进程）

这是 GStreamer–CUDA 的进程级限制，非 SDK 逻辑缺陷。

编码输入统一为 `encode(const VideoFrame&)`：按 `ImageData::device()` 路由（CPU 软编 /
GPU CUDA 直通 / TPU 占位 fail-closed），`pts_ms` 可注入外部时间戳（0=内部按 fps 自增）。

测试数据需单独下载：`curl -L -o test_data.zip https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/test_data.zip`

## 架构

```
csrc/           — C++ SDK 源码
├── core/       — Tensor、日志、平台声明
├── runtime/    — 推理运行时 + 后端抽象（ort/mnn/trt）
├── vision/     — 视觉模型（检测、分类、OCR、人脸、姿态等）
├── audio/      — ASR/TTS/VAD/SR
├── pybind/     — pybind11 绑定
└── encryption/ — XOR 模型加密
python/         — Python 包（封装 pybind 模块）
capi/           — C API
csharp/         — C# 绑定
cmake/          — 查找 onnxruntime、mnn、opencv、trt 的模块
```

## 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_ORT` | ON | OnnxRuntime 后端 |
| `ENABLE_MNN` | ON | MNN 后端 |
| `ENABLE_TRT` | OFF | 需要 `WITH_GPU=ON`，不支持 Apple |
| `WITH_GPU` | ON | 启用 CUDA（默认 SM 8.6） |
| `BUILD_AUDIO` | ON | 启用音频模块（samplerate、kaldi-native-fbank、cppjieba） |
| `BUILD_VISION` | ON | 启用视觉模块（OpenCV） |
| `BUILD_CAPI` | ON | C API |
| `BUILD_PYTHON` | ON | pybind11 模块 |
| `BUILD_TESTS` | OFF | Catch2 测试二进制 |
| `BUILD_ENCRYPTION` | ON | 需要 OpenSSL；未找到时静默禁用 |
| `ENABLE_WETEXT` | OFF | 可选 WeTextProcessing ITN 后端（覆盖最全，需 OpenFst + `-DWETEXT_INCLUDE_DIR`）；未找到时 ITN 退化到内置轻量实现 |

## 注意事项

- **MSVC**：必须添加 `/utf-8` 编译选项（根 CMakeLists.txt 已为 SDK 自动设置）。设置 `CMAKE_CXX_STANDARD=17`。
- **scikit-build-core**：使用多配置生成器，必须在 `pyproject.toml` 中显式设置 `CMAKE_BUILD_TYPE="Release"`，环境变量不会被转发。
- **Python `__init__.py`**：由 `python/__init__.py.in` 生成 —— CMake 在配置时替换 `@WITH_GPU@`、`@ENABLE_ORT@` 等变量。生成文件位于 `python/modeldeploy/__init__.py`。
- **OpenSSL**：Windows 下从 slproweb.com 安装，设置 `-DOPENSSL_ROOT_DIR="C:/Program Files/OpenSSL-Win64"`。未找到 OpenSSL 时加密功能静默禁用。
- **GPU 构建**：默认 CUDA 架构为 86（RTX 40 系列）。测试数据来自 modelscope，不在仓库内。
- **TRT 后端**：需要预先通过 `trtexec` 生成 `.engine` 文件。从 ONNX 在线构建 engine 速度较慢。
- **Linux rpath**：`$ORIGIN`；macOS：`@loader_path` —— SDK 运行时无需设置 `LD_LIBRARY_PATH`。
- **NVIDIA Jetson**：通过 `/etc/nv_tegra_release` 自动检测；设置架构标志并强制 `WITH_GPU=ON`、`ENABLE_TRT=ON`，需要 TBB。
- **C++17 必需**；第三方依赖（pybind11、Catch2）已捆绑在 `third_party/` 中。
- **Sophgo TPU 测试（触发词“在 sophgo 上测试”）**：用户要求算能 TPU 交叉编译 + 部署测试时，纯 Sophgo 构建用 `ENABLE_SOPHGO=ON` + **`ENABLE_ORT=OFF`**（不依赖动态 onnxruntime）；在 `172.168.100.243` 的 `tpuc_dev` 容器（`/workspace`）内构建，产物经 `.243` 直传 `172.168.100.70` 的 `/data/ModelDeploy/build_sophgo/bin` 运行；Sophgo int8 bmodel 多属 batch=1 静态形状，pipeline 内须 `set_cls_batch_size(1)`。

## CI 工作流

两个 GitHub Actions 工作流：
- `build_wheel.yml` — 在 ubuntu/windows 上运行 `python -m build`，Python 3.12–3.13，无测试
- `build_release.yml` — 完整 cmake 构建 + `ctest`（CPU）；GPU 任务仅编译，无测试。标签 `v*` 触发 GitHub Release 上传

## 开发过程文档（superpowers）

- **覆盖技能默认路径**：superpowers 的 design spec / 实施计划默认写 `docs/superpowers/`，但本仓库约定改到仓库根 **`.superpowers/`**：spec 存 `.superpowers/specs/`，plan 存 `.superpowers/plans/`（`docs/` 只放面向使用者的公开文档）。
- 这些是开发者过程产物，不进入 `docs/` 公开导航，`tools/check_docs_links.ps1` 不扫描。