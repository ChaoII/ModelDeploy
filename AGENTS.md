# ModelDeploy — Agent 指南

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
| `WITH_STATIC_CRT` | OFF | MSVC MT 运行时替代 MD |

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

## CI 工作流

两个 GitHub Actions 工作流：
- `build_wheel.yml` — 在 ubuntu/windows 上运行 `python -m build`，Python 3.12–3.13，无测试
- `build_release.yml` — 完整 cmake 构建 + `ctest`（CPU）；GPU 任务仅编译，无测试。标签 `v*` 触发 GitHub Release 上传