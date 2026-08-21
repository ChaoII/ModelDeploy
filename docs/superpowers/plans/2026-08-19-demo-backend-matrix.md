# Demo 后端×平台矩阵重构 · 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `examples/demo_<model>/` 重构为“后端×平台”命名的一次性二进制矩阵（9 种：ort_cpu / ort_gpu_cuda_ep / ort_gpu_trt_ep / mnn_cpu / mnn_cuda / mnn_opencl / mnn_vulkan / sophgo_tpu_f16 / sophgo_tpu_int8），纯硬编码直接运行，CMake 只用现有开关推导编译。

**Architecture:** 新增共享 `examples/common/demo_runner.{h,cpp}` 集中每种主模型的“选后端 + 加载 + 推理 + 绘制 + 保存”逻辑；每个矩阵文件仅 ~2 行选择后端。新增 CMake 函数 `md_add_demo_matrix(STEM SOURCE_DIR)` 按现有开关条件 `add_executable`（每个目标同时编译该 `.cpp` 与 `../common/demo_runner.cpp`）。矩阵按**目录主模型**一套（demo_face 只给 face_det 主矩阵，其余人脸子模型作保留特殊 demo）。

**Tech Stack:** CMake / C++17 / ModelDeploySDK + OpenCV（仅 `BuildExamples`）。

## Global Constraints
- 命名：`demo_<stem>_{ort_cpu|ort_gpu_cuda_ep|ort_gpu_trt_ep|mnn_cpu|mnn_cuda|mnn_opencl|mnn_vulkan|sophgo_tpu_f16|sophgo_tpu_int8}.cpp`
- **只用现有 CMake 开关**（ENABLE_ORT / ENABLE_MNN / ENABLE_SOPHGO / WITH_GPU / BUILD_CAPI），不新增。
- `_ort_gpu_trt_ep` 仅 gated 于 `ENABLE_ORT AND WITH_GPU`；**不含 `ENABLE_TRT`**（那是原生 TRT 后端）。
- `Device::VULKAN` 在 MNN `build_option` 未处理 → mnn_vulkan 必须显式 `opt.mnn_option.forward_type = modeldeploy::mnn::MNN_FORWARD_VULKAN;`。
- Sophgo fp16/int8 = **文件名区分**（`*_F16.bmodel` / `*_INT8.bmodel`），无运行时开关；经 `use_sophgo_backend(0)` 加载。
- 矩阵文件**纯硬编码、无命令行参数**；默认模型/图片路径内置，输出 `result_<model>_<backend>.jpg`。
- 保留现有特殊 demo（capi/batch/multi_thread/benchmark/profile/子模型/pipeline）源文件，仅用开关包好并归类，**不展开 ×9**。
- 每个矩阵目标同时编译 `${SOURCE_DIR}/demo_<stem>_<backend>.cpp` 与 `${SOURCE_DIR}/../common/demo_runner.cpp`。
- include 统一用 `csrc/...` 前缀（根与 csrc 均已在 include_directories）。
- 默认模型/图片路径以 `../../test_data/...` 相对（examples 构建工作目录）。
- 当前环境只能实跑 ORT-CPU（`build_tdc`）与 ORT-GPU（`build_tdc_gpu`）；MNN/Sophgo/TRT 仅保证 CMake 开关与源码可编译校验，不实跑。

---
## 模型 → 主模型映射（本计划范围）
| 目录 | stem | 主模型类 | include |
|---|---|---|---|
| demo_det | detection | `detection::UltralyticsDet` | `csrc/vision/detection/ultralytics_det.h` |
| demo_cls | classification | `classification::Classification` | `csrc/vision/classification/classification.h` |
| demo_kps | pose | `detection::UltralyticsPose` | `csrc/vision/pose/ultralytics_pose.h` |
| demo_obb | obb | `detection::UltralyticsObb` | `csrc/vision/obb/ultralytics_obb.h` |
| demo_iseg | instance_seg | `detection::UltralyticsSeg` | `csrc/vision/iseg/ultralytics_seg.h` |
| demo_sem | sem | `detection::UltralyticsSem` | `csrc/vision/sem/ultralytics_sem.h` |
| demo_depth | depth | `detection::UltralyticsDepth` | `csrc/vision/depth/ultralytics_depth.h` |
| demo_face | face_det | `face::Scrfd` | `csrc/vision/face/face_det/scrfd.h` |
| demo_lpr | lpr_pipeline | `lpr::LprPipeline` | `csrc/vision/lpr/lpr_pipeline/lpr_pipeline.h` |
| demo_ocr | ocr_pipeline | `ocr::PaddleOCR` | `csrc/vision/ocr/ppocr.h` |
| demo_pipeline | pedestrian_attribute | `pipeline::PedestrianAttribute` | `csrc/vision/pipeline/pedestrian_attribute.h` |

保留特殊 demo：demo_det 的 capi/batch/multi_thread(_trt)/multi_thread_compare/benchmark/profile、demo_face 的 age/gender/rec/as*/rec_pipeline/insightface、demo_lpr 的 det/rec、demo_ocr 的 det/rec/structure/capi/recognition、demo_cls/demo_obb/demo_kps/demo_iseg/demo_sem/demo_depth/demo_pipeline 的 capi 与各 `*_sophgo`、`demo_sophgo_clone.cpp`。

---

### Task 1: 共享运行助手 demo_runner + CMake 矩阵函数 + demo_det 全链

**Files:**
- Create: `examples/common/demo_runner.h`
- Create: `examples/common/demo_runner.cpp`
- Modify: `examples/CMakeLists.txt`（定义 `md_add_demo_matrix` + 添加 `add_subdirectory(demo_det)` 已存在）
- Create: `examples/demo_det/demo_detection_{ort_cpu,ort_gpu_cuda_ep,ort_gpu_trt_ep,mnn_cpu,mnn_cuda,mnn_opencl,mnn_vulkan,sophgo_tpu_f16,sophgo_tpu_int8}.cpp`
- Modify: `examples/demo_det/CMakeLists.txt`
- Test: 构建 + 运行 `demo_detection_ort_cpu`

**Interfaces:**
- Produces: `demo::Backend` 枚举；`demo::run_detection(Backend)->int`；CMake `md_add_demo_matrix(STEM SOURCE_DIR)`。后续任务复用同一 `Backend` 与 `md_add_demo_matrix`，把 `run_detection` 换成各自 `run_<model>`。

- [ ] **Step 1: 创建 `examples/common/demo_runner.h`**

```cpp
#pragma once
namespace demo {
enum class Backend {
    OrtCpu, OrtGpuCudaEp, OrtGpuTrtEp,
    MnnCpu, MnnCuda, MnnOpencl, MnnVulkan,
    SophgoF16, SophgoInt8
};
int run_detection(Backend);
int run_classification(Backend);
int run_pose(Backend);
int run_obb(Backend);
int run_instance_seg(Backend);
int run_sem(Backend);
int run_depth(Backend);
int run_face_det(Backend);
int run_lpr_pipeline(Backend);
int run_ocr_pipeline(Backend);
int run_pedestrian_attribute(Backend);
}
```

- [ ] **Step 2: 创建 `examples/common/demo_runner.cpp`（含 `run_detection` 参考实现；其余 `run_<model>` 由后续任务填充，先用 `return 1;` 占位以保证本任务可链接）**

```cpp
#include "demo_runner.h"

#include <cstdio>
#include <string>
#include <vector>

#include "csrc/runtime/runtime_option.h"
#include "csrc/vision/detection/ultralytics_det.h"
#include "csrc/vision/common/visualize/visualize.h"

namespace demo {
namespace {
constexpr const char* kFont = "../../test_data/msyh.ttc";

modeldeploy::RuntimeOption make_option(Backend b) {
    modeldeploy::RuntimeOption opt;
    switch (b) {
        case Backend::OrtCpu:
            opt.use_ort_backend(); opt.use_cpu(); opt.set_cpu_thread_num(4); break;
        case Backend::OrtGpuCudaEp:
            opt.use_ort_backend(); opt.use_gpu(0); break;
        case Backend::OrtGpuTrtEp:
            opt.use_ort_backend(); opt.use_gpu(0); opt.enable_trt = true;
            opt.enable_fp16 = true; opt.ort_option.trt_engine_cache_path = "./trt_engine"; break;
        case Backend::MnnCpu:
            opt.use_mnn_backend(); opt.use_cpu(); break;
        case Backend::MnnCuda:
            opt.use_mnn_backend(); opt.use_gpu(0); break;
        case Backend::MnnOpencl:
            opt.use_mnn_backend(); opt.use_opencl(0); break;
        case Backend::MnnVulkan:
            opt.use_mnn_backend(); opt.mnn_option.forward_type = modeldeploy::mnn::MNN_FORWARD_VULKAN; break;
        case Backend::SophgoF16:
        case Backend::SophgoInt8:
            opt.use_sophgo_backend(0); break;
    }
    return opt;
}

const char* backend_tag(Backend b) {
    switch (b) {
        case Backend::OrtCpu: return "ort_cpu";
        case Backend::OrtGpuCudaEp: return "ort_gpu_cuda_ep";
        case Backend::OrtGpuTrtEp: return "ort_gpu_trt_ep";
        case Backend::MnnCpu: return "mnn_cpu";
        case Backend::MnnCuda: return "mnn_cuda";
        case Backend::MnnOpencl: return "mnn_opencl";
        case Backend::MnnVulkan: return "mnn_vulkan";
        case Backend::SophgoF16: return "sophgo_tpu_f16";
        case Backend::SophgoInt8: return "sophgo_tpu_int8";
    }
    return "?";
}

// MNN 模型目前多数未提供 .mnn（仅 OCR 有）；未提供时返回 onnx 同名 .mnn 推断路径（运行时会报文件缺失）。
std::string model_path(const std::string& base, const std::string& onnx, const std::string& bmodel, Backend b) {
    if (b == Backend::SophgoF16 || b == Backend::SophgoInt8)
        return "../../test_data/test_models/sophgo/" + bmodel;
    return "../../test_data/test_models/onnx/" + onnx;
}
}  // namespace

int run_detection(Backend b) {
    std::string model = model_path("detection", "yolo26n/yolo26n.onnx", "yolo11n_det1280_int8.bmodel", b);
    const char* img = "../../test_data/test_images/test_pedestrian_attribute_scale.png";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto det = std::make_unique<modeldeploy::vision::detection::UltralyticsDet>(model, opt);
    if (!det->is_initialized()) {
        std::fprintf(stderr, "[%s] init failed: %s\n", backend_tag(b), model.c_str());
        return 1;
    }
    det->get_preprocessor().set_size({640, 640});
    const auto label_map = det->get_label_map("names");
    auto img_im = modeldeploy::vision::ImageData::imread(img);
    if (img_im.empty()) { std::fprintf(stderr, "cannot read image: %s\n", img); return 1; }
    std::vector<modeldeploy::vision::DetectionResult> result;
    constexpr int warmup = 10, loops = 50;
    for (int i = 0; i < warmup; ++i) det->predict(img_im, &result);
    modeldeploy::TimerArray timers;
    for (int i = 0; i < loops; ++i) det->predict(img_im, &result, &timers);
    timers.print_benchmark();
    modeldeploy::vision::dis_det(result);
    auto vis = modeldeploy::vision::vis_det(img_im, result, 0.5, label_map, kFont, 14, 0.15, false);
    (void)vis.imwrite("result_detection_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu objects\n", backend_tag(b), result.size());
    return 0;
}

// 后续任务填充（本任务先占位，保证全家目标可链接）
int run_classification(Backend) { return 1; }
int run_pose(Backend) { return 1; }
int run_obb(Backend) { return 1; }
int run_instance_seg(Backend) { return 1; }
int run_sem(Backend) { return 1; }
int run_depth(Backend) { return 1; }
int run_face_det(Backend) { return 1; }
int run_lpr_pipeline(Backend) { return 1; }
int run_ocr_pipeline(Backend) { return 1; }
int run_pedestrian_attribute(Backend) { return 1; }
}  // namespace demo
```

> 需在 demo_runner.cpp 顶部确认 include 无误：`UltralyticsDet` 所在头、`TimerArray`（`csrc/core/timer.h` 或经 vision.h）、`ImageData::imread`/`empty`（`csrc/vision/common/image_data.h`）、`dis_det`/`vis_det`（`visualize.h`）。若 `TimerArray`/`dis_det` 不在当前 include 下，补 `#include "csrc/core/timer.h"` 与 `#include "csrc/vision/common/visualize/visualize.h"`（后者已含）。请以实际编译为准补 include。

- [ ] **Step 3: 修改 `examples/CMakeLists.txt` —— 顶部定义 `md_add_demo_matrix`**

在 `add_subdirectory(demo_image)` 之前插入：

```cmake
# 后端×平台 demo 矩阵：只用现有开关推导；每个目标编译 demo_<stem>_<backend>.cpp + common/demo_runner.cpp
function(md_add_demo_matrix STEM SOURCE_DIR)
  set(_runner "${SOURCE_DIR}/../common/demo_runner.cpp")
  set(_base "${SOURCE_DIR}/demo_${STEM}")
  if(ENABLE_ORT)
    add_executable(demo_${STEM}_ort_cpu "${_base}_ort_cpu.cpp" "${_runner}")
    target_link_libraries(demo_${STEM}_ort_cpu PUBLIC ${LIBRARY_NAME})
  endif()
  if(ENABLE_ORT AND WITH_GPU)
    add_executable(demo_${STEM}_ort_gpu_cuda_ep "${_base}_ort_gpu_cuda_ep.cpp" "${_runner}")
    target_link_libraries(demo_${STEM}_ort_gpu_cuda_ep PUBLIC ${LIBRARY_NAME})
    # ORT 的 TRT EP 不需要原生 ENABLE_TRT
    add_executable(demo_${STEM}_ort_gpu_trt_ep "${_base}_ort_gpu_trt_ep.cpp" "${_runner}")
    target_link_libraries(demo_${STEM}_ort_gpu_trt_ep PUBLIC ${LIBRARY_NAME})
  endif()
  if(ENABLE_MNN)
    foreach(_s mnn_cpu mnn_opencl mnn_vulkan)
      add_executable(demo_${STEM}_${_s} "${_base}_${_s}.cpp" "${_runner}")
      target_link_libraries(demo_${STEM}_${_s} PUBLIC ${LIBRARY_NAME})
    endforeach()
  endif()
  if(ENABLE_MNN AND WITH_GPU)
    add_executable(demo_${STEM}_mnn_cuda "${_base}_mnn_cuda.cpp" "${_runner}")
    target_link_libraries(demo_${STEM}_mnn_cuda PUBLIC ${LIBRARY_NAME})
  endif()
  if(ENABLE_SOPHGO)
    add_executable(demo_${STEM}_sophgo_tpu_f16 "${_base}_sophgo_tpu_f16.cpp" "${_runner}")
    target_link_libraries(demo_${STEM}_sophgo_tpu_f16 PUBLIC ${LIBRARY_NAME})
    add_executable(demo_${STEM}_sophgo_tpu_int8 "${_base}_sophgo_tpu_int8.cpp" "${_runner}")
    target_link_libraries(demo_${STEM}_sophgo_tpu_int8 PUBLIC ${LIBRARY_NAME})
  endif()
endfunction()
```

- [ ] **Step 4: 创建 9 个 `examples/demo_det/demo_detection_<backend>.cpp`**（各自内容仅 main；后端枚举对应如下）

每个文件固定 2 行：
```cpp
#include "../common/demo_runner.h"
int main() { return demo::run_detection(demo::Backend::OrtCpu); }
```
对应 9 个文件与后端枚举：
| 文件 | 枚举 |
|---|---|
| `demo_detection_ort_cpu.cpp` | `OrtCpu` |
| `demo_detection_ort_gpu_cuda_ep.cpp` | `OrtGpuCudaEp` |
| `demo_detection_ort_gpu_trt_ep.cpp` | `OrtGpuTrtEp` |
| `demo_detection_mnn_cpu.cpp` | `MnnCpu` |
| `demo_detection_mnn_cuda.cpp` | `MnnCuda` |
| `demo_detection_mnn_opencl.cpp` | `MnnOpencl` |
| `demo_detection_mnn_vulkan.cpp` | `MnnVulkan` |
| `demo_detection_sophgo_tpu_f16.cpp` | `SophgoF16` |
| `demo_detection_sophgo_tpu_int8.cpp` | `SophgoInt8` |

- [ ] **Step 5: 重写 `examples/demo_det/CMakeLists.txt`**

```cmake
# 主模型矩阵
md_add_demo_matrix(detection ${CMAKE_CURRENT_LIST_DIR})

# 保留特殊 demo（按真实依赖开关 gating）
if(ENABLE_ORT)
  add_executable(demo_detection_batch demo_detection_batch.cpp)
  target_link_libraries(demo_detection_batch PUBLIC ${LIBRARY_NAME})
  add_executable(demo_detection_multi_thread demo_detection_multi_thread.cpp)
  target_link_libraries(demo_detection_multi_thread PUBLIC ${LIBRARY_NAME})
  add_executable(demo_detection_multi_thread_trt demo_detection_multi_thread_trt.cpp)  # 源码实为 ORT(use_ort_backend)
  target_link_libraries(demo_detection_multi_thread_trt PUBLIC ${LIBRARY_NAME})
  add_executable(demo_multi_thread_compare demo_multi_thread_compare.cpp)
  target_link_libraries(demo_multi_thread_compare PUBLIC ${LIBRARY_NAME})
  add_executable(demo_benchmark demo_benchmark.cpp)
  target_link_libraries(demo_benchmark PUBLIC ${LIBRARY_NAME})
  add_executable(demo_profile demo_profile.cpp)
  target_link_libraries(demo_profile PUBLIC ${LIBRARY_NAME})
endif()
if(BUILD_CAPI)
  add_executable(demo_detection_capi demo_detection_capi.cpp)
  target_link_libraries(demo_detection_capi PUBLIC ${LIBRARY_NAME})
endif()
if(ENABLE_SOPHGO)
  add_executable(demo_detection_sophgo demo_detection_sophgo.cpp)
  target_link_libraries(demo_detection_sophgo PUBLIC ${LIBRARY_NAME})
endif()
```
> 说明：原 `demo_detection_cxx.cpp` 被 9 个矩阵文件取代，从 build 移除（源文件可保留或删除，本计划删除以避免同义）。`demo_detection_multi_thread_trt.cpp` 名字含 trt 但源码是 ORT-CPU，故归 ENABLE_ORT（与 spec 一致）。

- [ ] **Step 6: 配置 + 构建 ORT-CPU**

```bash
cd E:\CLionProjects\ModelDeploy
# 复用 build_tdc（现有 Ninja/CPU/ORT 配置）
cmake -S . -B build_tdc -G Ninja -DENABLE_ORT=ON -DENABLE_MNN=OFF -DENABLE_SOPHGO=OFF -DWITH_GPU=OFF -DBUILD_CAPI=ON -DBUILD_PYTHON=OFF -DBUILD_AUDIO=ON -DBUILD_VISION=ON -DBUILD_TESTS=OFF
```
在 MSVC 环境（`vcvars64`）内：
```bash
cmake --build build_tdc --target demo_detection_ort_cpu demo_detection_batch demo_detection_capi --parallel
```
Expected: 三个目标编译/链接成功；9 个矩阵目标中仅 ort_cpu 生成（其余因 ENABLE_MNN/OFF、WITH_GPU/OFF、ENABLE_SOPHGO/OFF 不生成）。

- [ ] **Step 7: 直接运行二进制**

```bash
# 工作目录 examples/demo_det（相对路径 "../../test_data/..." 生效）
& build_tdc\bin\demo_detection_ort_cpu.exe
```
Expected: 打印 benchmark 与 `[ort_cpu] done, N objects`，当前目录生成 `result_detection_ort_cpu.jpg`（exit 0）。

- [ ] **Step 8: 提交**

```bash
git add examples/common/demo_runner.h examples/common/demo_runner.cpp examples/CMakeLists.txt examples/demo_det/demo_detection_*.cpp examples/demo_det/CMakeLists.txt
git commit -m "feat(examples): demo backend×platform matrix + shared runner (detection)"
```

---

### Task 2: demo_cls 主模型矩阵

**Files:** Modify `examples/common/demo_runner.cpp`（把 `run_classification` 占位替换为完整实现）；Create 9× `examples/demo_cls/demo_classification_<backend>.cpp`；Modify `examples/demo_cls/CMakeLists.txt`。

**Interfaces:** Consumes `md_add_demo_matrix` / `demo::Backend`；Produces `demo::run_classification(Backend)`。

- [ ] **Step 1: 替换 demo_runner.cpp 中 `run_classification` 占位**

```cpp
int run_classification(Backend b) {
    std::string model = model_path("classification", "yolo11n/yolo11n-cls.onnx", "yolo11n-cls.bmodel", b);
    const char* img = "../../test_data/test_images/test_face.jpg";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::classification::Classification>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    auto im = modeldeploy::vision::ImageData::imread(img);
    if (im.empty()) { std::fprintf(stderr, "cannot read image: %s\n", img); return 1; }
    modeldeploy::vision::ClassifyResult res;
    constexpr int warmup = 10, loops = 50;
    for (int i = 0; i < warmup; ++i) m->predict(im, &res);
    modeldeploy::TimerArray timers;
    for (int i = 0; i < loops; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();
    modeldeploy::vision::dis_cls(res);
    auto vis = modeldeploy::vision::vis_cls(im, res, 5, 0.0, kFont, 14, 0.15, false);
    (void)vis.imwrite("result_classification_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] label=%d score=%.4f\n", backend_tag(b), res.label_ids.empty() ? -1 : res.label_ids[0],
                res.scores.empty() ? -1.f : res.scores[0]);
    return 0;
}
```
> 需在 demo_runner.cpp 顶部 `#include "csrc/vision/classification/classification.h"`（仅用到时）。

- [ ] **Step 2: 创建 9 个 `examples/demo_cls/demo_classification_<backend>.cpp`**（2 行模式，`run_classification` + 各自 `Backend`，同 Task 1 表）。
- [ ] **Step 3: 重写 `examples/demo_cls/CMakeLists.txt`**：`md_add_demo_matrix(classification ${CMAKE_CURRENT_LIST_DIR})` + 保留 `demo_classification_capi.cpp`（gated `if(BUILD_CAPI)`）、`demo_classification_sophgo.cpp`（gated `if(ENABLE_SOPHGO)`）。原 `demo_classification_cxx.cpp` 移除（被矩阵取代）。
- [ ] **Step 4: 构建**：`cmake --build build_tdc --target demo_classification_ort_cpu demo_classification_capi --parallel` Expected: 成功。
- [ ] **Step 5: 运行** `& build_tdc\bin\demo_classification_ort_cpu.exe` Expected: 打印 `[ort_cpu] label=... score=...`，生成结果图。
- [ ] **Step 6: 提交**（demo_runner.cpp + demo_cls/* 9 文件 + demo_cls/CMakeLists.txt）`git commit -m "feat(examples): classification demo matrix"`

---

### Task 3: demo_kps 主模型矩阵（pose）

**Files:** Modify `examples/common/demo_runner.cpp`（替换 `run_pose`）；Create 9× `examples/demo_kps/demo_pose_<backend>.cpp`；Modify `examples/demo_kps/CMakeLists.txt`。

- [ ] **Step 1: 替换 `run_pose` 占位**

```cpp
int run_pose(Backend b) {
    std::string model = model_path("pose", "yolo11n/yolo11n-pose.onnx", "yolo26n/yolo26n-pose_int8.bmodel", b);
    const char* img = "../../test_data/test_images/test_person.jpg";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsPose>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    m->get_preprocessor().set_size({640, 640});
    auto im = modeldeploy::vision::ImageData::imread(img);
    if (im.empty()) { std::fprintf(stderr, "cannot read image: %s\n", img); return 1; }
    std::vector<modeldeploy::vision::KeyPointsResult> res;
    constexpr int warmup = 10, loops = 50;
    for (int i = 0; i < warmup; ++i) m->predict(im, &res);
    modeldeploy::TimerArray timers;
    for (int i = 0; i < loops; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();
    auto vis = modeldeploy::vision::vis_pose(im, res, kFont, 12, 3, 0.15, false);
    (void)vis.imwrite("result_pose_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu persons\n", backend_tag(b), res.size());
    return 0;
}
```
> include: `csrc/vision/pose/ultralytics_pose.h`。
- [ ] **Step 2:** 9 个 `demo_pose_<backend>.cpp`（`run_pose`+Backend）。
- [ ] **Step 3: 重写 `demo_kps/CMakeLists.txt`**：`md_add_demo_matrix(pose ${CMAKE_CURRENT_LIST_DIR})` + 保留 `demo_pose_capi.cpp`（BUILD_CAPI）、`demo_pose_sophgo.cpp`（ENABLE_SOPHGO）、`demo_keypoint_cxx.cpp`（ENABLE_ORT）。移除原 `demo_pose_cxx.cpp`（被矩阵取代）。
- [ ] **Step 4-5:** 构建 `tm demo_pose_ort_cpu` + 运行，Expected 成功/N 人。
- [ ] **Step 6: 提交** `feat(examples): pose demo matrix`

---

### Task 4: demo_obb 主模型矩阵

- [ ] **Step 1: 替换 `run_obb`**（include `csrc/vision/obb/ultralytics_obb.h`）
```cpp
int run_obb(Backend b) {
    std::string model = model_path("obb", "yolo11n/yolo11n-obb_nms.onnx", "yolo26n/yolo26n-obb_int8.bmodel", b);
    const char* img = "../../test_data/test_images/test_obb1.jpg";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsObb>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    m->get_preprocessor().set_size({640, 640});
    auto im = modeldeploy::vision::ImageData::imread(img);
    if (im.empty()) { std::fprintf(stderr, "cannot read image: %s\n", img); return 1; }
    std::vector<modeldeploy::vision::ObbResult> res;
    constexpr int warmup = 10, loops = 50;
    for (int i = 0; i < warmup; ++i) m->predict(im, &res);
    modeldeploy::TimerArray timers;
    for (int i = 0; i < loops; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();
    auto vis = modeldeploy::vision::vis_obb(im, res, 0.5, kFont, 14, 0.15, false);
    (void)vis.imwrite("result_obb_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu obbs\n", backend_tag(b), res.size());
    return 0;
}
```
- [ ] **Step 2:** 9 个 `demo_obb_<backend>.cpp`（`run_obb`）。
- [ ] **Step 3: 重写 `demo_obb/CMakeLists.txt`**：`md_add_demo_matrix(obb ...)` + 保留 `demo_obb_capi.cpp`（BUILD_CAPI）、`demo_obb_sophgo.cpp`（ENABLE_SOPHGO）。移除 `demo_obb_cxx.cpp`。
- [ ] **Step 4-5:** 构建 `tm demo_obb_ort_cpu` + 运行。
- [ ] **Step 6: 提交** `feat(examples): obb demo matrix`

---

### Task 5: demo_iseg 主模型矩阵

- [ ] **Step 1: 替换 `run_instance_seg`**（include `csrc/vision/iseg/ultralytics_seg.h`）
```cpp
int run_instance_seg(Backend b) {
    std::string model = model_path("instance_seg", "yolo11n/yolo11n-seg_nms.onnx", "yolo26n/yolo26n-seg_int8.bmodel", b);
    const char* img = "../../test_data/test_images/test_person.jpg";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsSeg>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    m->get_preprocessor().set_size({640, 640});
    auto im = modeldeploy::vision::ImageData::imread(img);
    if (im.empty()) { std::fprintf(stderr, "cannot read image: %s\n", img); return 1; }
    std::vector<modeldeploy::vision::InstanceSegResult> res;
    constexpr int warmup = 10, loops = 50;
    for (int i = 0; i < warmup; ++i) m->predict(im, &res);
    modeldeploy::TimerArray timers;
    for (int i = 0; i < loops; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();
    auto vis = modeldeploy::vision::vis_iseg(im, res, 0.5, kFont, 14, 0.15, false);
    (void)vis.imwrite("result_instance_seg_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu masks\n", backend_tag(b), res.size());
    return 0;
}
```
- [ ] **Step 2:** 9 个 `demo_instance_seg_<backend>.cpp`。
- [ ] **Step 3: 重写 `demo_iseg/CMakeLists.txt`**：`md_add_demo_matrix(instance_seg ...)` + 保留 `demo_instance_seg_capi.cpp`（BUILD_CAPI）、`demo_iseg_sophgo.cpp`（ENABLE_SOPHGO）。移除 `demo_instance_seg_cxx.cpp`。
- [ ] **Step 4-5:** 构建 + 运行。
- [ ] **Step 6: 提交** `feat(examples): instance_seg demo matrix`

---

### Task 6: demo_sem 主模型矩阵

- [ ] **Step 1: 替换 `run_sem`**（include `csrc/vision/sem/ultralytics_sem.h`）
```cpp
int run_sem(Backend b) {
    std::string model = model_path("sem", "yolo26n/yolo26n-sem.onnx", "yolo26n/yolo26n-sem_int8.bmodel", b);
    const char* img = "../../test_data/test_images/test_sem_540.jpg";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsSem>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    const auto label_map = m->get_label_map("names");
    auto im = modeldeploy::vision::ImageData::imread(img);
    if (im.empty()) { std::fprintf(stderr, "cannot read image: %s\n", img); return 1; }
    modeldeploy::vision::SemSegResult res;
    constexpr int warmup = 10, loops = 50;
    for (int i = 0; i < warmup; ++i) m->predict(im, &res);
    modeldeploy::TimerArray timers;
    for (int i = 0; i < loops; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();
    auto vis = modeldeploy::vision::vis_sem(im, res, label_map, 0.5, false);
    (void)vis.imwrite("result_sem_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done %zux%zu\n", backend_tag(b), res.shape.empty() ? 0 : (size_t)res.shape[0],
                res.shape.size() < 2 ? 0 : (size_t)res.shape[1]);
    return 0;
}
```
- [ ] **Step 2:** 9 个 `demo_sem_<backend>.cpp`。
- [ ] **Step 3: 重写 `demo_sem/CMakeLists.txt`**：`md_add_demo_matrix(sem ...)` + 保留 `demo_sem_sophgo.cpp`（ENABLE_SOPHGO）。移除 `demo_sem_cxx.cpp`。
- [ ] **Step 4-5:** 构建 + 运行。
- [ ] **Step 6: 提交** `feat(examples): sem demo matrix`

---

### Task 7: demo_depth 主模型矩阵

- [ ] **Step 1: 替换 `run_depth`**（include `csrc/vision/depth/ultralytics_depth.h`）
```cpp
int run_depth(Backend b) {
    std::string model = model_path("depth", "yolo26n/yolo26n-depth.onnx", "yolo26n/yolo26n-depth_int8.bmodel", b);
    const char* img = "../../test_data/test_images/test_depth_540.jpg";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsDepth>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    auto im = modeldeploy::vision::ImageData::imread(img);
    if (im.empty()) { std::fprintf(stderr, "cannot read image: %s\n", img); return 1; }
    modeldeploy::vision::DepthResult res;
    constexpr int warmup = 10, loops = 50;
    for (int i = 0; i < warmup; ++i) m->predict(im, &res);
    modeldeploy::TimerArray timers;
    for (int i = 0; i < loops; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();
    auto vis = modeldeploy::vision::vis_depth(im, res, true, false);
    (void)vis.imwrite("result_depth_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done %zux%zu\n", backend_tag(b), res.shape.empty() ? 0 : (size_t)res.shape[0],
                res.shape.size() < 2 ? 0 : (size_t)res.shape[1]);
    return 0;
}
```
- [ ] **Step 2:** 9 个 `demo_depth_<backend>.cpp`。
- [ ] **Step 3: 重写 `demo_depth/CMakeLists.txt`**：`md_add_demo_matrix(depth ...)` + 保留 `demo_depth_sophgo.cpp`（ENABLE_SOPHGO）。移除 `demo_depth_cxx.cpp`。
- [ ] **Step 4-5:** 构建 + 运行。
- [ ] **Step 6: 提交** `feat(examples): depth demo matrix`

---

### Task 8: demo_face 主模型矩阵（face_det）

**Files:** Modify `demo_runner.cpp`（替换 `run_face_det`）；Create 9× `examples/demo_face/demo_face_det_<backend>.cpp`；Modify `examples/demo_face/CMakeLists.txt`（新增矩阵 + 保留全部人脸子模型 demo，仅 gating）。

- [ ] **Step 1: 替换 `run_face_det`**（include `csrc/vision/face/face_det/scrfd.h`）
```cpp
int run_face_det(Backend b) {
    std::string model = model_path("face_det", "face/scrfd_2.5g_bnkps_shape640x640.onnx", "face/scrfd_2.5g_int8.bmodel", b);
    const char* img = "../../test_data/test_images/test_face_detection4.jpg";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::face::Scrfd>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    m->get_preprocessor().set_size({640, 640});
    auto im = modeldeploy::vision::ImageData::imread(img);
    if (im.empty()) { std::fprintf(stderr, "cannot read image: %s\n", img); return 1; }
    std::vector<modeldeploy::vision::KeyPointsResult> res;
    constexpr int warmup = 10, loops = 50;
    for (int i = 0; i < warmup; ++i) m->predict(im, &res);
    modeldeploy::TimerArray timers;
    for (int i = 0; i < loops; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();
    auto vis = modeldeploy::vision::vis_keypoints(im, res, kFont, 12, 3, 0.15, false, true);
    (void)vis.imwrite("result_face_det_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu faces\n", backend_tag(b), res.size());
    return 0;
}
```
- [ ] **Step 2:** 9 个 `demo_face_det_<backend>.cpp`（`run_face_det`）。
- [ ] **Step 3: 重写 `demo_face/CMakeLists.txt`**：`md_add_demo_matrix(face_det ...)` + 保留 demo_face 内全部子模型 demo（`demo_face_age_cxx/capi`、`demo_face_gender_*`、`demo_face_as_*_cxx/capi`、`demo_face_as_pipeline_*`、`demo_face_rec_*`、`demo_face_rec_pipeline_*`、`demo_face_det_cxx/capi`、`demo_insightface_cxx`），逐一 gated：`_capi` → BUILD_CAPI；`_cxx` → ENABLE_ORT；`demo_insightface_cxx.cpp` 额外链接 `${OpenCV_LIBS}`（若原 CMake 如此）。移除或保留 `demo_face_det_cxx.cpp`——它被矩阵 fact 取代，删除；其余保留。
  > 提示：请先读原 `examples/demo_face/CMakeLists.txt` 全文，按原文逐项保留并加 gate（这是 demo 目录中最复杂的一个）。
- [ ] **Step 4-5:** 构建 `tm demo_face_det_ort_cpu`（+ 人脸各特殊 demo）并运行。
- [ ] **Step 6: 提交** `feat(examples): face_det demo matrix; gate face sub-demos`

---

### Task 9: demo_lpr 主模型矩阵（lpr_pipeline）

- [ ] **Step 1: 替换 `run_lpr_pipeline`**（include `csrc/vision/lpr/lpr_pipeline/lpr_pipeline.h`）
```cpp
int run_lpr_pipeline(Backend b) {
    std::string det = "../../test_data/test_models/onnx/yolov5plate.onnx";
    std::string rec = "../../test_data/test_models/onnx/plate_recognition_color.onnx";
    const char* img = "../../test_data/test_images/test_lpr_pipeline.jpg";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::lpr::LprPipeline>(det, rec, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    auto im = modeldeploy::vision::ImageData::imread(img);
    if (im.empty()) { std::fprintf(stderr, "cannot read image: %s\n", img); return 1; }
    std::vector<modeldeploy::vision::LprResult> res;
    constexpr int warmup = 10, loops = 50;
    for (int i = 0; i < warmup; ++i) m->predict(im, &res);
    modeldeploy::TimerArray timers;
    for (int i = 0; i < loops; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();
    auto vis = modeldeploy::vision::vis_lpr(im, res, kFont, 14, 4, 0.15, false);
    (void)vis.imwrite("result_lpr_pipeline_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu plates\n", backend_tag(b), res.size());
    return 0;
}
```
- [ ] **Step 2:** 9 个 `demo_lpr_pipeline_<backend>.cpp`。
- [ ] **Step 3: 重写 `demo_lpr/CMakeLists.txt`**：`md_add_demo_matrix(lpr_pipeline ...)` + 保留 `demo_lpr_detection_*` `demo_lpr_recognizer_*` `demo_lpr_pipeline_capi/cxx`（各自 gated：_capi→BUILD_CAPI、_cxx→ENABLE_ORT）。移除原 `demo_lpr_pipeline_cxx.cpp`（被矩阵取代；若名冲突需归并——本计划将矩阵命名为 `demo_lpr_pipeline_*`，与已有 `demo_lpr_pipeline_cxx.cpp`/`capi` 不冲突，故直接移除 `demo_lpr_pipeline_cxx.cpp` 源文件避免同名）。
- [ ] **Step 4-5:** 构建 `tm demo_lpr_pipeline_ort_cpu` + 运行。
- [ ] **Step 6: 提交** `feat(examples): lpr_pipeline demo matrix`

---

### Task 10: demo_ocr 主模型矩阵（ocr_pipeline）

- [ ] **Step 1: 替换 `run_ocr_pipeline`**（include `csrc/vision/ocr/ppocr.h`）
```cpp
int run_ocr_pipeline(Backend b) {
    const char* det = "../../test_data/test_models/onnx/ocr/ppocrv5_mobile/det_infer2.onnx";
    const char* cls = "../../test_data/test_models/onnx/ocr/ppocrv4_mobile/cls_infer.onnx";
    const char* rec = "../../test_data/test_models/onnx/ocr/ppocrv5_mobile/rec_infer1.onnx";
    const char* dict = "../../test_data/dict.txt";
    const char* img = "../../test_data/test_images/ocr2.jpg";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::ocr::PaddleOCR>(det, cls, rec, dict, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    m->set_rec_batch_size(8);
    auto im = modeldeploy::vision::ImageData::imread(img);
    if (im.empty()) { std::fprintf(stderr, "cannot read image: %s\n", img); return 1; }
    modeldeploy::vision::OCRResult res;
    constexpr int warmup = 5, loops = 20;
    for (int i = 0; i < warmup; ++i) m->predict(im, &res);
    modeldeploy::TimerArray timers;
    for (int i = 0; i < loops; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();
    modeldeploy::vision::dis_ocr(res);
    auto vis = modeldeploy::vision::vis_ocr(im, res, kFont, 14, 0.15, false);
    (void)vis.imwrite("result_ocr_pipeline_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu boxes\n", backend_tag(b), res.boxes.size());
    return 0;
}
```
- [ ] **Step 2:** 9 个 `demo_ocr_pipeline_<backend>.cpp`。
- [ ] **Step 3: 重写 `demo_ocr/CMakeLists.txt`**：`md_add_demo_matrix(ocr_pipeline ...)` + 保留 `demo_ocr_cxx.cpp`/`demo_ocr_det_cxx.cpp`/`demo_ocr_rec_cxx.cpp`/`demo_ocr_recognition_capi(_batch).cpp`/`demo_pp_structure_table_*`/`demo_structure_table_cxx.cpp`（gated：_cxx→ENABLE_ORT、_capi→BUILD_CAPI）。**若 `demo_ocr_cxx.cpp` 与矩阵的 `demo_ocr_pipeline_*` 重复**，移除 `demo_ocr_cxx.cpp`；若希望保留也可，用 ENABLE_ORT 包住，名字不与矩阵冲突即可。
- [ ] **Step 4-5:** 构建 `tm demo_ocr_pipeline_ort_cpu` + 运行（需 OCR 模型，若存在则出文字框）。
- [ ] **Step 6: 提交** `feat(examples): ocr_pipeline demo matrix`

---

### Task 11: demo_pipeline 主模型矩阵（pedestrian_attribute）

- [ ] **Step 1: 替换 `run_pedestrian_attribute`**（include `csrc/vision/pipeline/pedestrian_attribute.h`）
```cpp
int run_pedestrian_attribute(Backend b) {
    std::string det = "../../test_data/test_models/onnx/zhgd_det.onnx";
    std::string ml = "../../test_data/test_models/onnx/zhgd_ml.onnx";
    const char* img = "../../test_data/test_images/test_pedestrian_attribute_scale.png";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::pipeline::PedestrianAttribute>(det, ml, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    const auto label_map = m->get_label_map("names");
    auto im = modeldeploy::vision::ImageData::imread(img);
    if (im.empty()) { std::fprintf(stderr, "cannot read image: %s\n", img); return 1; }
    std::vector<modeldeploy::vision::AttributeResult> res;
    constexpr int warmup = 10, loops = 50;
    for (int i = 0; i < warmup; ++i) m->predict(im, &res);
    modeldeploy::TimerArray timers;
    for (int i = 0; i < loops; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();
    auto vis = modeldeploy::vision::vis_attr(im, res, 0.5, label_map, kFont, 8, 0.15, false);
    (void)vis.imwrite("result_pedestrian_attribute_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu persons\n", backend_tag(b), res.size());
    return 0;
}
```
- [ ] **Step 2:** 9 个 `demo_pedestrian_attribute_<backend>.cpp`。
- [ ] **Step 3: 重写 `demo_pipeline/CMakeLists.txt`**：`md_add_demo_matrix(pedestrian_attribute ...)` + 保留 `demo_pedestrian_attribute_capi.cpp`（BUILD_CAPI；win32 输出名 `demo_pedestrian_attribute_capi`，按原 CMake）、`demo_pedestrian_attribute_sophgo.cpp`（ENABLE_SOPHGO，链接 `${SOPHGO_LIBS}` 按原）、`demo_sophgo_clone.cpp`（ENABLE_SOPHGO）。移除原 `demo_pedestrian_attribute_cxx.cpp`（被矩阵取代）。
  > 请先读原 `demo_pipeline/CMakeLists.txt`，按其保留/特殊链接规则逐项 gating。
- [ ] **Step 4-5:** 构建 `tm demo_pedestrian_attribute_ort_cpu` + 运行。
- [ ] **Step 6: 提交** `feat(examples): pedestrian_attribute demo matrix`

---

### Task 12: 终验 + 收尾

**Files:** `examples/CMakeLists.txt`（确认无需改）、各 dir CMakeLists（确认）、README（如 `examples/README.md` 或根 README 有 examples 说明则同步矩阵用法）。

- [ ] **Step 1: 全量 ORT-CPU 重配置 + 构建矩阵目标**
```bash
cmake -S . -B build_tdc -G Ninja -DENABLE_ORT=ON -DENABLE_MNN=OFF -DENABLE_SOPHGO=OFF -DWITH_GPU=OFF -DBUILD_CAPI=ON -DBUILD_PYTHON=OFF -DBUILD_AUDIO=ON -DBUILD_VISION=ON -DBUILD_TESTS=OFF
cmake --build build_tdc --target demo_detection_ort_cpu demo_classification_ort_cpu demo_pose_ort_cpu demo_obb_ort_cpu demo_instance_seg_ort_cpu demo_sem_ort_cpu demo_depth_ort_cpu demo_face_det_ort_cpu demo_lpr_pipeline_ort_cpu demo_ocr_pipeline_ort_cpu demo_pedestrian_attribute_ort_cpu --parallel
```
Expected: 11 个 ort_cpu 目标全部编译链接成功。
- [ ] **Step 2: 运行检测**（examples/demo_det 工作目录）`& build_tdc\bin\demo_detection_ort_cpu.exe`（出图，exit 0）。
- [ ] **Step 3: 清理**：确认旧 `demo_<model>_cxx.cpp` 已从各 CMakeLists 移除、矩阵取代；保留特殊文件均在且 gated。`git status` 干净。
- [ ] **Step 4: 提交**（若 README/示例说明有改动）`docs: document demo backend×platform matrix`.

---
## Self-Review 备注
- Spec 覆盖：命名✓(Task1 表)、CMake开关推导✓(helper)、ort_gpu_trt_ep 无 ENABLE_TRT✓、mnn_vulkan 显式 forward_type✓、sophgo 文件名区分✓、纯硬编码无参数✓、目录主模型一套矩阵✓（face 只 face_det）、保留特殊 demo✓（各 Task Step3）、当前环境只实测 ORT✓（Task1/12）。
- 无占位符：所有 run_<model> 均给出完整代码；9 个矩阵文件为 2 行固定模式已枚举。
- 类型一致：backend 枚举名与文件后缀一一对应；`run_<model>` 名与 `demo_runner.h` 声明一致；CMake `md_add_demo_matrix(STEM)` 与文件名 `demo_<stem>_<backend>.cpp` 一致。
- 需实现的实现者在 Task8/Task11 Step3 前先读对应目录原 CMakeLists 以精确保留特殊 demo 与链接规则。
