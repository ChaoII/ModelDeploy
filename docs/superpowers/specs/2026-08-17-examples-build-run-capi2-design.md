# 设计：examples 全面可编译可运行（C++/capi/C#）

日期：2026-08-17 · 分支：capi-v2

## 背景与目标

ModelDeploy 仓库中有四大类 examples：
- **C++（`.cxx`/非 capi `.cpp`）**：基于 C++ SDK（`csrc/vision.h` 等）
- **capi（`*_capi.cpp`，22 个）**：使用**已删除的 v1 C API**（引用 `capi/utils/md_*.h`，该目录已不存在）
- **C#（`csharp/ModelDeployExample`）**：1 个，基于 capi2 的 C# 绑定，编译运行正常但多数测试被注释、缺新 API 演示
- **平台特定**：`*_sophgo`、`*_trt`、`multi_thread` 等

目标：让**所有 examples 能编译并实际推理运行出结果**（CPU/ORT 本机验证），覆盖 C++、capi、C# 三种绑定；并演示本次新增的模型参数 setter API（`md_model_set_param_*`）。

## 现状问题（已勘查确认）

| # | 问题 | 范围 | 影响 |
|---|------|------|------|
| P1 | 旧 v1 C API 已删除 | 22 个 `*_capi.cpp` | 全部编译失败（`capi/utils/md_*.h` 不存在） |
| P2 | sophgo examples 无 CMake 保护 | `demo_depth_sophgo`、`demo_sem_sophgo`（各自 CMakeLists 缺 `if(ENABLE_SOPHGO)`） | CPU 构建尝试编译 SOPHGO 专用目标而失败 |
| P3 | `imshow`/`waitKey` 阻塞 | ~29 个 C++ `.cxx`/capi | 交互窗口阻塞运行，无头环境挂起 |
| P4 | GPU/TRT 硬编码选项 | `use_gpu(0)`/`enable_trt`/`trt_engine` | 本机 CPU 后端不适用，可能导致 init 失败 |
| P5 | 注释死代码/实验代码 | 多处（如 demo_detection_cxx 大量 `//` 块） | 干扰阅读、误导 |
| P6 | C# 多数测试注释 + 缺新 API | ModelDeployExample `Program.cs` | 仅跑 Detection+Image，未覆盖其他模型与参数 API |

## 方案（已获用户认可）

### A. capi → capi2 重写（22 个 `*_capi.cpp`）

统一模板，基于 `capi2/md_capi.h` 的通用分发 API：

```c
#include "capi2/md_capi.h"
#include <iostream>

static void die(MDStatus s, const char* what) {
    if (s != MD_OK) { std::cerr << what << " failed: " << md_get_last_error() << "\n"; std::exit(1); }
}

int main() {
    MDOptionHandle opt; md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model;
    die(md_model_create(&model, MD_MODEL_<KIND>, "<模型路径聚合串>", opt), "create");

    /* 演示新 API（按 kind 对应的参数） */
    /* die(md_model_set_param_d(model, "conf_threshold", 0.4), "set_param"); */

    MDImageHandle img;
    die(md_image_from_file(&img, "../../test_data/test_images/xxx.jpg"), "read image");

    MDResultHandle res;
    die(md_model_predict(model, img, &res), "predict");

    /* md_result_* 读取并打印，示例见下 */
    size_t n = 0; /* 按 result kind 用对应 getter */

    md_draw_result(img, res, &(MDDrawOptions){ .threshold = 0.4, .font_path = "../../test_data/msyh.ttc" });
    md_image_save(img, "capi2_out.jpg");

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    return 0;
}
```

**关键点：**
- 不用阻塞的 `md_image_show`，统一 `md_image_save` 落盘
- 每个 example 按 `MDModelKind` 对应，model_path 用 `|` 聚合串（OCR/LPR/insightface/audio 同 capi2 规范）
- 每类演示 1-2 个 `md_model_set_param_*`（对应本次新 API）；无参数 kind 演示 `md_model_param_names` 自省
- 结果用 `md_result_*` 数组式 getter 打印
- 保留与 C++ `.cxx` 等价的功能覆盖（det/pose/obb/iseg/cls/face/ocr/lpr/sem/depth/ped/audio）

### B. 修 CMake 平台保护

- `examples/demo_depth/CMakeLists.txt`：`demo_depth_sophgo` 包进 `if(ENABLE_SOPHGO)`，`demo_depth_cxx` 保持无条件（需可 CPU 跑）
- `examples/demo_sem/CMakeLists.txt`：`demo_sem_sophgo` 包进 `if(ENABLE_SOPHGO)`
- 全量检查所有 `*_sophgo`/专用目标均受保护，保证 CPU/ORT 构建全绿

### C. C# example 全面启用 + 新 API 演示

`csharp/ModelDeployExample/Program.cs`：
- 启用 Classification / Pose / OCR / InsightFace（若模型可用）等测试
- 新增 `SetParam` 演示（如 det 设 `conf_threshold`/`nms_threshold`）
- 保留 Detection+Image 现有测试
- 音频（SenseVoice/Kokoro）按模型可用性启用（本机有 sense_voice/kokoro 模型则启用）

### D. C++ `.cxx` examples 修复 + 验证

- 把 GPU/TRT 硬编码选项改为 CPU/ORT 可跑（`use_ort_backend()` + CPU），或通过环境/平台判断
- `imshow`/`waitKey` 改为 `imwrite`/`save`（或 `#ifdef` 有头模式），避免无头挂起
- 清理影响编译/运行的注释死代码（轻微，不重写逻辑）
- 平台特定（sophgo/trt/multi_thread）仅保证受 CMake 保护、CPU 构建不参与，不再本机适配
- 逐一编译 + 实际推理验证（用 test_data 既有模型）

## 模型文件映射（capi2 聚合串）

| kind | 模型文件（test_data 下） |
|------|------|
| DETECTION | `test_models/onnx/yolo11n/yolo11n.onnx` |
| CLASSIFICATION | `test_models/onnx/yolo11n/yolo11n-cls.onnx` |
| POSE | `test_models/onnx/yolo11n/yolo11n-pose.onnx` |
| OBB | `test_models/onnx/yolo11n/yolo11n-obb.onnx` |
| INSTANCE_SEG | `test_models/onnx/yolo11n/yolo11n-seg.onnx` |
| SEM_SEG | `test_models/onnx/yolo26n/yolo26n-sem.onnx` |
| DEPTH | `test_models/onnx/yolo26n/yolo26n-depth.onnx` |
| FACE_DET | `test_models/onnx/face/scrfd_2.5g_bnkps_shape640x640.onnx` |
| FACE_REC | `test_models/onnx/face/face_recognizer.onnx` |
| FACE_AGE | `test_models/onnx/face/age_predictor.onnx` |
| FACE_GENDER | `test_models/onnx/face/gender_predictor.onnx` |
| OCR | `test_models/onnx/ocr/ppocrv4_mobile/det_infer.onnx\|cls_infer.onnx\|rec_infer.onnx\|dict.txt` |
| LPR_DET | `test_models/onnx/yolov5plate.onnx` |
| LPR_REC | `test_models/onnx/plate_recognition_color.onnx` |
| LPR_PIPELINE | `test_models/onnx/yolov5plate.onnx\|plate_recognition_color.onnx` |
| INSIGHTFACE | `test_models/onnx/insightface/buffalo_l/det_10g.onnx\|w600k_r50.onnx\|2d106det.onnx\|1k3d68.onnx\|genderage.onnx` |
| PED_ATTR | `test_models/onnx/zhgd_det.onnx\|test_models/onnx/zhgd_ml.onnx`（det+attr 双模型，CPU 可用；cxx 示例经命令行传参） |
| ASR | `test_models/onnx/sense_voice/model.int8.onnx\|tokens.txt` |
| TTS | `test_models/onnx/kokoro_v1_1/model.onnx\|tokens.txt\|lexicon-gb-en.txt\|lexicon-zh.txt\|voices.bin\|dict\|dir` |

> 具体路径以实施时对 test_data 实际文件的核认为准；缺模型的 example 标记为"需下载"并在 spec 中注明，仍保证编译 + 运行不崩溃（找不到文件时报清晰错误退出）。

## 验收标准

1. **编译**：CPU/ORT 构建（`build_tdc`）全量 examples 目标编译通过，0 error（sophgo/trt 专用目标经 CMake 保护不参与本机）。
2. **运行（实际推理出结果）**：对每个非平台特定 example，用 test_data 既有模型实际推理成功，输出/落盘结果文件（如 `*_out.jpg`）。
3. **C#**：`dotnet build` 0/0，运行产出实际推理结果，含 SetParam 演示。
4. **新 API 覆盖**：capi 与 C# 的示例演示 `md_model_set_param_*`。
5. CMake 保护修复后 CPU 构建不再因 sophgo 默认目标失败。

## 范围外（YAGNI）

- 不重新实现/适配 sophgo、TRT、GPU 专用 examples 的本机推理（无对应后端），仅保证 CMake 保护隔离。
- 不重写 C++ `.cxx` 的算法逻辑，只做"可编译 + 可 CPU 运行 + 结果落盘"的最小改造。
- 不新增 serving（Triton）目录的 C++ 示例。
- 不修改 SDK 本身（除非暴露关键编译/运行阻塞且修复最小）。
