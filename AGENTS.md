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

**模型/权重（重要）**：`test_data/test_models/` 下已按后端预置大量模型——`onnx/`、`mnn/`、
`ncnn/`、`trt/`（含现成 `.engine`，如 `trt/yolo11n/yolo11n.engine`、`yolo11n_without_nms.engine`、
`yolo11n_nms_b8.engine`（batch=8）、OCR/人脸/车牌等）、`sophgo/`。**在转换/生成/下载任何模型之前，
先查看 `test_data/test_models` 是否已有对应后端的现成模型**，避免重复转换或在线构建（如 TRT 首次
从 ONNX 构建 engine 需数十秒）。

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
| `ENABLE_MNN` | OFF | MNN 后端 |
| `ENABLE_TRT` | OFF | 需要 `WITH_GPU=ON`，不支持 Apple |
| `ENABLE_NCNN` | OFF | ncnn 后端（CPU+Vulkan）；依赖 `cmake/ncnn.cmake`（配置时按平台从 modelscope 自动下载，Win/Linux-x64/aarch64）。Vision 模型经 ncnn 支持 ultralytics YOLO 全系（det/cls/obb/pose/seg/sem/depth），用 `ultralytics .pt export format=ncnn` 转出的 param/bin 加载 |
| `WITH_GPU` | ON | 启用 CUDA（默认 SM 8.6） |
| `BUILD_AUDIO` | OFF | 启用音频模块（samplerate、kaldi-native-fbank、cppjieba） |
| `BUILD_VISION` | ON | 启用视觉模块（OpenCV） |
| `BUILD_CAPI` | ON | C API |
| `BUILD_PYTHON` | OFF | pybind11 模块 |
| `BUILD_TESTS` | OFF | Catch2 测试二进制 |
| `BUILD_ENCRYPTION` | ON | 需要 mbedTLS git submodule（`git submodule update --init --recursive`） |
| `ENABLE_WETEXT` | OFF | 可选 WeTextProcessing ITN 后端（覆盖最全）；依赖已 vendor 于 `third_party/`（openfst + WeTextProcessing + glog stub），开启即由 CMake 现场构建，无需外部路径；模型缺失时 ITN 退化到内置轻量实现 |

## 注意事项

- **MSVC**：必须添加 `/utf-8` 编译选项（根 CMakeLists.txt 已为 SDK 自动设置）。设置 `CMAKE_CXX_STANDARD=17`。
- **scikit-build-core**：使用多配置生成器，必须在 `pyproject.toml` 中显式设置 `CMAKE_BUILD_TYPE="Release"`，环境变量不会被转发。
- **Python `__init__.py`**：由 `python/__init__.py.in` 生成 —— CMake 在配置时替换 `@WITH_GPU@`、`@ENABLE_ORT@` 等变量。生成文件位于 `python/modeldeploy/__init__.py`。
- **BUILD_ENCRYPTION**：基于 mbedTLS（git submodule，`third_party/mbedtls`，构建前 `git submodule update --init --recursive`）。无系统 OpenSSL 依赖。
- **GPU 构建**：默认 CUDA 架构为 86（RTX 40 系列）。测试数据来自 modelscope，不在仓库内。
- **TRT 后端**：需要预先通过 `trtexec` 生成 `.engine` 文件。从 ONNX 在线构建 engine 速度较慢。
- **ncnn 后端**：`cmake/ncnn.cmake` 按平台（Win-x64 / Linux-x64 / Linux-aarch64）从 modelscope 自动下载预编译静态包（含 ncnn + glslang 与各自 CMake config，`find_package(ncnn)`/`find_package(glslang)` 直接可用）；不支持平台配置时 fail-fast。MNN 同理（`cmake/mnn.cmake`，单库静态包，无独立 CUDA 库）。
- **Linux rpath**：`$ORIGIN`；macOS：`@loader_path` —— SDK 运行时无需设置 `LD_LIBRARY_PATH`。
- **NVIDIA Jetson**：通过 `/etc/nv_tegra_release` 自动检测；设置架构标志并强制 `WITH_GPU=ON`、`ENABLE_TRT=ON`，需要 TBB。
- **C++20 必需**（根 `CMakeLists.txt` 设 `CMAKE_CXX_STANDARD 20`，但未用 `std::format`/`std::ranges`/`std::span`，GCC 11+/MSVC 19.29+ 理论上可编，CI 用 GCC 13/14 与 MSVC 14.44/14.51）；第三方依赖（pybind11、Catch2）已捆绑在 `third_party/` 中。
- **baseline_compare 已知平台差异（不修）**：`tests/baseline_compare.cpp` 的 ORT 端到端/预处理护栏基线按 MSVC 生成。GCC/Linux 构建因 `-O3 -mavx2 -mfma` 的 FMA 收缩，使 `utils::cal_letter_box_param` 的 `pad_h` 比 80.0 高 1 ULP（80.000007629）、`pad_w` 得 7.6e-6，随后 fused 预处理核（`src_yf<0`/`src_xf<0` 严格判断）把首行内容/左缘列误写为 pad → 输入张量逐像元差最大 0.557，翻转 logit≈阈值的临界检测（如 yolo26n instance[3] score 0.303 vs 0.564）。已确认与 onnxruntime 打包/MNN/ncnn 下载式接入**无关**（同机 i7-12700 双平台同走 AVX2 核），属预存在缺陷、生产影响趋近于零，故不修。CI 若在 GCC 跑 baseline_compare 会持续失败，需在 CI 平台重新生成基线。

## CI 工作流
- **Sophgo TPU 测试（触发词“在 sophgo 上测试”）**：用户要求算能 TPU 交叉编译 + 部署测试时，纯 Sophgo 构建用 `ENABLE_SOPHGO=ON` + **`ENABLE_ORT=OFF`**（不依赖动态 onnxruntime）；在 `172.168.100.243` 的 `tpuc_dev` 容器（`/workspace`）内构建，产物经 `.243` 直传 `172.168.100.70` 的 `/data/ModelDeploy/build_sophgo/bin` 运行；Sophgo int8 bmodel 多属 batch=1 静态形状，pipeline 内须 `set_cls_batch_size(1)`。

GitHub Actions 工作流（job 名与产物名都带 OS + 工具链标记，便于区分产物最低运行环境）：

- `build_release.yml` — SDK 产物 + GitHub Release（标签 `v*`）。CPU 三档：
  `ubuntu24.04-gcc13`（glibc 2.39）、`windows2025-vs2026`（MSVC 14.51）、
  `windows2022-vs2022`（MSVC 14.44）；另有 `cpu_glibc228` 在
  `manylinux_2_28_x86_64` 容器里出 glibc 2.28 兼容产物（Ubuntu 20.04+/Debian 10+/RHEL 8+），
  容器内编译测试并校验 `objdump -T` 的 glibc 符号上限。GPU 三档同 CPU 的工具链矩阵，CUDA **12.8**
  （与本地开发机、`cmake/onnxruntime.cmake` 固定的 ORT 1.29 GPU 包同大版本），仅编译 + 无设备 smoke，
  并带 `ldd`/`dumpbin` 探针核对 ORT provider 实际依赖的 `cudart` 大版本。
- `build_aarch64.yml` — **原生 arm64 runner**（`ubuntu-24.04-arm`）构建 + 跑测试，两档：
  `ubuntu24.04-gcc13`（glibc 2.39）与 `glibc2.28`（`manylinux_2_28_aarch64` 容器，原生执行非 QEMU）。
  原先的 x86→aarch64 交叉编译作业已移除。
- `build_wheel.yml` — `python -m build` 快速回归（ubuntu24.04 / windows2025，py3.12-3.13）；
  另有 `manylinux` 作业经 cibuildwheel 出 `manylinux_2_28_x86_64` 可移植轮子。
- `build_wheel_aarch64.yml` — 在 `ubuntu-24.04-arm` 上原生经 cibuildwheel 出
  `manylinux_2_28_aarch64` 轮子（此前跑 QEMU，每次撞 6 小时上限被取消，从未产出）。
- `rust.yml` — Rust 绑定构建 + 测试。

Linux 兼容性基线：**glibc 2.28**（`manylinux_2_28` 容器，GCC 14）与 **glibc 2.39**
（Ubuntu 24.04，GCC 13）并存；Windows 基线为 **MSVC 14.44（VS2022 / v143）** 与
**14.51（VS2026）** 并存。`ilammy/msvc-dev-cmd` 的 `toolset` 是**前缀匹配**：写 `14.40`
会命中镜像里并存的 `14.44`（日志可验 `VCToolsVersion`），务必写实际存在的精确版本。
CPU 发布作业 `BUILD_TESTS=OFF`，测试覆盖由 `cpu_glibc228` / GPU / aarch64 作业承担；
这些测试跑 `~[opencl]~[vulkan]~[gpu]~[regression]`，排除 GCC 上已知失败的 baseline_compare。

## GitHub 凭据

- GitHub 访问令牌（供 `gh` / REST API / Chrono II 调试 Actions 用）以纯文本存放在 git 凭据文件：
  `C:\Users\aichao\.git-credentials`
  每行格式 `https://<github用户名>:<token>@github.com`（本机用户 `ChaoII`，仓库 `ChaoII/ModelDeploy`）。
- 读取方式（不打印明文本体，token 存入变量复用）：
  ```powershell
  $line = Get-Content "$env:USERPROFILE\.git-credentials" |
          Where-Object { $_ -like '*@github.com*' } | Select-Object -First 1
  $token = (($line.Split('@'))[0] -split ':')[2]   # 提取 https://user:<token>@... 中的 token
  ```
- 用法示例（拉 CI 失败日志需仓库 admin 权限，本目录 token 具备）：
  ```powershell
  curl -s -H "Authorization: Bearer $token" `
       "https://api.github.com/repos/ChaoII/ModelDeploy/actions/jobs/<job_id>/logs"
  ```
- **安全红线**：该 token 属机密。**禁止**把 token 明文写进 AGENTS.md / 仓库文件 / 提交 / 日志回显；调用 GitHub 一律经环境变量或动态读取上述文件取得，切勿硬编码或打印其明文。

## 开发过程文档（superpowers）

- **覆盖技能默认路径**：superpowers 的 design spec / 实施计划默认写 `docs/superpowers/`，但本仓库约定改到仓库根 **`.superpowers/`**：spec 存 `.superpowers/specs/`，plan 存 `.superpowers/plans/`（`docs/` 只放面向使用者的公开文档）。
- 这些是开发者过程产物，不进入 `docs/` 公开导航，`tools/check_docs_links.ps1` 不扫描。