# Demo 后端×平台矩阵重构 — 设计

日期：2026-08-19
状态：已确认（含 ENABLE_TRT 语义更正）

## 背景与目标

当前 `examples/demo_<model>/` 下是通用命名的 demo 源文件（如 `demo_detection_cxx.cpp`），每个文件内硬编码后端配置与相对路径，CMakeLists 无条件 `add_executable`。目标：

1. 每个模型目录内置一套“后端×平台”命名的 demo 源文件，名字即语义，一目了然。
2. CMakeLists 用**布尔开关（仅复用现有开关）**控制哪些文件编译。
3. 二进制**纯硬编码、无参数**，直接运行即可用内置默认模型/图片路径。

## 范围

- `demo_det`、`demo_cls`、`demo_pose(demo_kps)`、`demo_obb`、`demo_iseg`、`demo_sem`、`demo_depth`、`demo_face`、`demo_lpr`、`demo_ocr`、`demo_pipeline` 一起改。
- `demo_audio`、`demo_image` 不含模型矩阵，保持不变。
- 现有特殊 demo（capi / batch / multi_thread / multi_thread_trt / multi_thread_compare / benchmark / profile / 旧 `*_sophgo`）**保留**，用现有开关包好并归类。

## 命名约定

每个模型目录内生成后端×平台文件（`<model>` 为模型名，如 `detection`、`classification`）：

```
demo_<model>_ort_cpu.cpp
demo_<model>_ort_gpu_cuda_ep.cpp
demo_<model>_ort_gpu_trt_ep.cpp
demo_<model>_mnn_cpu.cpp
demo_<model>_mnn_cuda.cpp
demo_<model>_mnn_opencl.cpp
demo_<model>_mnn_vulkan.cpp
demo_<model>_sophgo_tpu_f16.cpp
demo_<model>_sophgo_tpu_int8.cpp
```

## 共享运行助手（减少重复）

新增 `examples/common/demo_runner.h`（及各模型共享实现），职责：

- 后端枚举 + `make_runtime_option(DemoBackend)`：按后端构造 `RuntimeOption`
  - ort_cpu → `use_ort_backend()`
  - ort_gpu_cuda_ep → `use_ort_backend() + use_gpu()`
  - ort_gpu_trt_ep → `use_ort_backend() + use_gpu()` + ORT TRT EP 选项
  - mnn_cpu → `use_mnn_backend()`
  - mnn_cuda → `use_mnn_backend() + use_gpu()`
  - mnn_opencl / mnn_vulkan → `use_mnn_backend()` + 对应设备后端
  - sophgo_tpu_f16 / int8 → `use_sophgo_backend()` + fp16/int8（bmodel）
- 模板运行函数（按模型）`run_<model>(DemoBackend, 默认模型路径, 默认图片路径)`：无参数，加载内置默认路径，推理，绘制，保存 `result_<model>_<backend>.jpg` 到当前目录。

于是每个矩阵文件仅 ~15 行：`int main(){ return run_detection<UltralyticsDet>(DemoBackend::OrtCpu); }`。9 个二进制都在、CMake 都可见，共享逻辑集中在 helper，不重复。

## CMake：只用现有开关推导（不加新开关）

helper 函数 `md_add_demo_matrix(<name> <cppdir>)` 内部按以下条件 `add_executable`：

| 文件后缀 | 编译条件 |
|---------|---------|
| `_ort_cpu` | `ENABLE_ORT` |
| `_ort_gpu_cuda_ep` | `ENABLE_ORT AND WITH_GPU` |
| `_ort_gpu_trt_ep` | `ENABLE_ORT AND WITH_GPU`（**不含** `ENABLE_TRT`，那是原生 TRT 后端） |
| `_mnn_cpu` | `ENABLE_MNN` |
| `_mnn_cuda` | `ENABLE_MNN AND WITH_GPU` |
| `_mnn_opencl` / `_mnn_vulkan` | `ENABLE_MNN` |
| `_sophgo_tpu_f16` / `_sophgo_tpu_int8` | `ENABLE_SOPHGO` |

保留的特殊 demo：
- 原生 TRT（如 `demo_detection_multi_thread_trt`）→ `ENABLE_TRT`
- capi / batch / benchmark / profile / multi_thread 等 → 按其实际依赖的开关包好（capi 依赖 BUILD_CAPI；一般模型 demo 依赖 ENABLE_ORT 或 ENABLE_MNN 等，按源码实际后端定）

未启用的后端文件不入 build，但**源文件保留在磁盘**（“一律生成”）。

## 默认路径（纯硬编码）

每个模型矩阵文件内置该模型 `test_data/` 下的默认模型+图片路径：
- ort 变体默认 `.onnx` 模型
- mnn 变体默认 `.mnn` 模型
- sophgo 变体默认 `.bmodel`
无命令行参数；直接运行二进制即用默认。

## 测试

当前环境仅能构建 ORT-CPU（`build_tdc`）与 ORT-GPU（`build_tdc_gpu`）。实测：
- 重新配置 + 编译 `_ort_cpu`（可能含 `_ort_gpu_cuda_ep`）demo。
- 直接运行二进制，确认出图/输出正常。
- MNN / Sophgo / TRT 因未构建，只保证 CMake 开关正确、源码可编译性由配置时校验，无法实跑（符合预期）。
