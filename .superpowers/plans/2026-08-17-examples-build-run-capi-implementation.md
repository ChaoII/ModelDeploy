# Examples 全面可编译可运行 实施计划（capi 重写 + C++ + C#）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让所有 examples（C++/capi/C#）能编译并实际推理运行出结果（CPU/ORT 本机），演示本次新增的 `md_model_set_param_*` 参数 API。

**Architecture:** 22 个 `*_capi.cpp` 从已删除的 v1 C API（`capi/utils/md_*.h`）重写为 capi（`capi/md_capi.h`）统一模板（`md_model_create(kind,path,opt)` → `md_model_predict`/`md_audio_*` → `md_result_*` → `md_draw_result` → `md_image_save`/`md_wav_save`）。修 CMake 保护平台特定目标。修 C++ `.cxx` 的 GPU/imshow 问题。C# example 全面启用测试 + 演示 SetParam。

**Tech Stack:** C++17 / C99 capi / CMake / C# (.NET 9)。

## Global Constraints

- capi 头文件 `capi/md_capi.h` 是唯一 C API 权威；不再引用 `capi/`（已删除目录）。
- 全部用 CPU/ORT 后端（`md_option_set_backend(opt, MD_BK_ORT)` + `md_option_set_device(opt, MD_DEV_CPU)`），避免 GPU/TRT。
- 不调用阻塞 API：视觉用 `md_image_save` 代替 `md_image_show`；不调用 `waitKey`/`imshow`/`use_gpu`/`enable_trt`（除非平台条件编译保护）。
- 模型路径相对 `../../test_data/`（examples 的可执行文件运行在 `build_tdc/bin`，cxx 源里用相对路径从 CMake 运行的 cwd 解析——统一沿用既有约定的 `../../test_data/...`）。
- 平台特定目标（`*_sophgo`、`*_trt`、`multi_thread*`）保持 CMake 条件保护，不要求本机 CPU 推理；`demo_depth_sophgo`、`demo_sem_sophgo` 需补 `ENABLE_SOPHGO` 保护（当前缺）。
- `${LIBRARY_NAME}` 是全项目 SDK 库变量（CMake 侧），capi examples 链接它即可（与既有 cxx 相同，无需新链接依赖）。
- MDStatus 错误处理：每个 capi 调用检查返回码，失败打印 `md_get_last_error()` 并非零退出——这是"运行不崩溃"的核心。
- 实际推理为最终验收：每个非平台特定 example 必须用 test_data 既有模型跑出结果并落盘。
- 不改 SDK 本体，除非编译/运行阻塞且修复最小。
- 设计 spec：`docs/superpowers/specs/2026-08-17-examples-build-run-capi-design.md`。

---

### Task 1: capi 基础辅助头 + 模型路径常量

**Files:**
- Create: `examples/capi_common.h`（可被各 `*_capi.cpp` 复用：die() + 常见 MDDrawOptions 辅助）

**Interfaces:**
- Produces: `die(MDStatus, const char*)`（供所有 Task 2-9 的 capi example 复用）

- [ ] **Step 1: 写 examples/capi_common.h**

```cpp
//
// 供 capi examples 复用的最小错误处理辅助
//
#ifndef MODELDEPLOY_EXAMPLES_CAPI_COMMON_H
#define MODELDEPLOY_EXAMPLES_CAPI_COMMON_H

#include "capi/md_capi.h"
#include <cstdio>
#include <cstdlib>

inline void die(MDStatus s, const char* what) {
    if (s != MD_OK) {
        std::fprintf(stderr, "[capi] %s failed: %s\n", what, md_get_last_error());
        std::exit(1);
    }
}

#endif
```

- [ ] **Step 2: 提交**

```bash
git add examples/capi_common.h
git commit -m "feat(examples): capi common error-checking helper"
```

---

### Task 2: CMake 平台保护修复 + 构建基线

**Files:**
- Modify: `examples/demo_depth/CMakeLists.txt`
- Modify: `examples/demo_sem/CMakeLists.txt`
- Test: `demo_depth` / `demo_sem` 逐目标编译

**Interfaces:**
- Consumes: 无
- Produces: CPU/ORT 构建全绿的 CMake 配置（后续所有 Task 的编译前提）

- [ ] **Step 1: demo_depth 加 SOPHGO 保护**

当前 `examples/demo_depth/CMakeLists.txt`（无条件加两个目标）。改为：

```cmake
add_executable(demo_depth_cxx demo_depth_cxx.cpp)
target_link_libraries(demo_depth_cxx PUBLIC ${LIBRARY_NAME})

if(ENABLE_SOPHGO)
add_executable(demo_depth_sophgo demo_depth_sophgo.cpp)
target_link_libraries(demo_depth_sophgo PUBLIC ${LIBRARY_NAME})
endif()
```

- [ ] **Step 2: demo_sem 加 SOPHGO 保护**

当前 `examples/demo_sem/CMakeLists.txt`。改为：

```cmake
add_executable(demo_sem_cxx demo_sem_cxx.cpp)
target_link_libraries(demo_sem_cxx PUBLIC ${LIBRARY_NAME})

if(ENABLE_SOPHGO)
add_executable(demo_sem_sophgo demo_sem_sophgo.cpp)
target_link_libraries(demo_sem_sophgo PUBLIC ${LIBRARY_NAME})
endif()
```

> 实施前请先读两个目录下当前 CMakeLists.txt，按其现有变量/风格贴合（保留既有 `LIBRARY_NAME` 链接方式与目标名）。

- [ ] **Step 3: 编译验证（CPU build）**

```
cd E:\CLionProjects\ModelDeploy
"<VS>\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build build_tdc --target demo_depth_cxx demo_sem_cxx
```
Expected: 0 error（demo_depth_sophgo/demo_sem_sophgo 不再因缺保护而出现在 CPU 构建失败集）。

- [ ] **Step 4: 提交**

```bash
git add examples/demo_depth/CMakeLists.txt examples/demo_sem/CMakeLists.txt
git commit -m "fix(examples): guard sophgo-only targets with ENABLE_SOPHGO"
```

---

### Task 3: capi 重写——视觉单模型 examples（det/pose/obb/iseg/cls/face_det）

**Files:**
- Modify: `examples/demo_det/demo_detection_capi.cpp`
- Modify: `examples/demo_kps/demo_pose_capi.cpp`
- Modify: `examples/demo_obb/demo_obb_capi.cpp`
- Modify: `examples/demo_iseg/demo_instance_seg_capi.cpp`
- Modify: `examples/demo_cls/demo_classification_capi.cpp`
- Modify: `examples/demo_face/demo_face_det_capi.cpp`

**Interfaces:**
- Consumes: Task 1 的 `die()`；capi `md_model_create/predict/draw_result`；`md_result_{detection,pose,obb,instance_seg,classification,face}`
- Produces: 可编译可运行的六个单模型 capi example（含参数 API 演示）

- [ ] **Step 1: 重写 demo_detection_capi.cpp（DETECTION，模板基准）**

完整内容（替换原文件）：

```cpp
//
// capi 检测示例：演示 md_model_set_param_d 设置 conf/nms threshold
//
#include "capi_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_DETECTION,
                        "../../test_data/test_models/onnx/yolo11n/yolo11n.onnx", opt), "create detection");

    // 演示新参数 API
    die(md_model_set_param_d(model, "conf_threshold", 0.4), "set conf_threshold");
    die(md_model_set_param_d(model, "nms_threshold", 0.45), "set nms_threshold");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_detection0.jpg"), "read image");

    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    const MDDetectionItem* items = nullptr;
    size_t n = 0;
    die(md_result_detection(res, &items, &n), "get detection result");
    std::printf("detected %zu objects\n", n);
    for (size_t i = 0; i < n; ++i)
        std::printf("  [%d] score=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
                    items[i].label_id, items[i].score,
                    items[i].x, items[i].y, items[i].w, items[i].h);

    MDDrawOptions draw{};
    draw.threshold = 0.4;
    draw.font_path = "../../test_data/msyh.ttc";
    draw.save_result = 1;
    die(md_draw_result(img, res, &draw), "draw");
    die(md_image_save(img, "capi_detection_out.jpg"), "save");

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    std::puts("OK -> capi_detection_out.jpg");
    return 0;
}
```

- [ ] **Step 2: 重写 demo_pose_capi.cpp（POSE）**

同样模板，关键差异：
- kind: `MD_MODEL_POSE`；路径 `../../test_data/test_models/onnx/yolo11n/yolo11n-pose.onnx`
- 参数：`md_model_set_param_d(model,"conf_threshold",0.4)` + `md_model_set_param_i(model,"keypoints_num",17)`
- 结果：`md_result_pose`（MDPoseItem 数组）+ 打印；绘制 `md_draw_result` 支持 pose
- 落盘 `capi_pose_out.jpg`

- [ ] **Step 3: 重写 demo_obb_capi.cpp（OBB）**
- kind `MD_MODEL_OBB`；路径 `../../test_data/test_models/onnx/yolo11n/yolo11n-obb.onnx`
- 参数：conf/nms；结果 `md_result_obb`；落盘 `capi_obb_out.jpg`

- [ ] **Step 4: 重写 demo_instance_seg_capi.cpp（INSTANCE_SEG）**
- kind `MD_MODEL_INSTANCE_SEG`；路径 `../../test_data/test_models/onnx/yolo11n/yolo11n-seg.onnx`
- 参数：conf/nms/mask_threshold（`md_model_set_param_d(model,"mask_threshold",0.5)`）
- 结果 `md_result_instance_seg`；绘制 mdraw_result；落盘 `capi_iseg_out.jpg`

- [ ] **Step 5: 重写 demo_classification_capi.cpp（CLASSIFICATION）**
- kind `MD_MODEL_CLASSIFICATION`；路径 `../../test_data/test_models/onnx/yolo11n/yolo11n-cls.onnx`
- 参数：`md_model_set_param_i(model,"top_k",5)` + `md_model_set_param_b(model,"multi_label",0)`
- 结果 `md_result_classification`（MDClassifyItem）；绘制 md_draw_result 支持 classification；落盘 `capi_cls_out.jpg`
- 输入图 `../../test_data/test_images/bus.jpg`

- [ ] **Step 6: 重写 demo_face_det_capi.cpp（FACE_DET）**
- kind `MD_MODEL_FACE_DET`；路径 `../../test_data/test_models/onnx/face/scrfd_2.5g_bnkps_shape640x640.onnx`
- 参数：conf/nms/landmarks_per_face（`md_model_set_param_i(model,"landmarks_per_face",5)`）
- 结果 `md_result_face`；落盘 `capi_face_det_out.jpg`
- 输入 `../../test_data/test_images/test_face1.jpg`

- [ ] **Step 7: 编译并逐个运行验证**

```
cd E:\CLionProjects\ModelDeploy
"<VS>\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build build_tdc --target demo_detection_capi demo_pose_capi demo_obb_capi demo_instance_seg_capi demo_classification_capi demo_face_det_capi
```
然后逐个在 `build_tdc/bin` 运行（`cmd /c 'demo_detection_capi.exe'` 等），Expected：打印检测/分类结果 + `OK -> capi_*_out.jpg`（实际推理出结果）。

> 若某 example 因模型/图像缺失无法推理，报告 NEEDS_CONTEXT，不臆测；能跑的必须跑出结果。

- [ ] **Step 8: 提交**

```bash
git add examples/demo_det/demo_detection_capi.cpp examples/demo_kps/demo_pose_capi.cpp examples/demo_obb/demo_obb_capi.cpp examples/demo_iseg/demo_instance_seg_capi.cpp examples/demo_cls/demo_classification_capi.cpp examples/demo_face/demo_face_det_capi.cpp
git commit -m "feat(examples): rewrite capi demos to capi (det/pose/obb/iseg/cls/face_det) + param API"
```

---

### Task 4: capi 重写——OCR / LPR / pedestrian 多模型 examples

**Files:**
- Modify: `examples/demo_ocr/demo_ocr_capi.cpp`
- Modify: `examples/demo_ocr/demo_ocr_recognition_capi.cpp`（存在则合并/替换为整链 OCR）
- Modify: `examples/demo_lpr/demo_lpr_pipeline_capi.cpp`
- Modify: `examples/demo_pipeline/demo_pedestrian_attribute_capi.cpp`
- （`demo_ocr_recognition_capi_batch.cpp`、`demo_pp_structure_table_capi.cpp`、`demo_lpr_detection_capi.cpp`、`demo_lpr_recognizer_capi.cpp` 视需要收编——若超出 capi 能力则报告）

**Interfaces:**
- Consumes: capi `md_model_create`（多子模型 `|` 聚合）、`md_result_ocr`/`md_result_lpr`/`md_result_attribute`
- Produces: OCR/LPR/PED_ATTR capi example（含参数 API 演示）

- [ ] **Step 1: 重写 demo_ocr_capi.cpp（OCR 整链路）**

```cpp
#include "capi_common.h"
int main() {
    MDOptionHandle opt; md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT); md_option_set_device(opt, MD_DEV_CPU); md_option_set_cpu_threads(opt, 4);
    MDModelHandle model;
    die(md_model_create(&model, MD_MODEL_OCR,
        "../../test_data/test_models/onnx/ocr/ppocrv4_mobile/det_infer.onnx|"
        "../../test_data/test_models/onnx/ocr/ppocrv4_mobile/cls_infer.onnx|"
        "../../test_data/test_models/onnx/ocr/ppocrv4_mobile/rec_infer.onnx|"
        "../../test_data/ppocrv4_dict.txt", opt), "create ocr");
    // 演示新参数 API（OCR 整链路参数）
    die(md_model_set_param_d(model, "det_db_box_thresh", 0.6), "set box_thresh");
    die(md_model_set_param_d(model, "cls_thresh", 0.9), "set cls_thresh");

    MDImageHandle img; die(md_image_from_file(&img, "../../test_data/test_images/test_ocr.png"), "read");
    MDResultHandle res; die(md_model_predict(model, img, &res), "predict");
    size_t n=0; die(md_result_count(res, &n), "count");
    for (size_t i=0;i<n;++i){ const int* quad=nullptr; const char* text=nullptr; float score=0;
        die(md_result_ocr(res,i,&quad,&text,&score),"ocr");
        if (text) std::printf("  [%zu] score=%.3f %s\n", i, score, text); }
    MDDrawOptions draw{}; draw.font_path="../../test_data/msyh.ttc"; draw.save_result=1;
    die(md_draw_result(img,res,&draw),"draw"); die(md_image_save(img,"capi_ocr_out.jpg"),"save");
    md_result_destroy(res); md_image_destroy(img); md_model_destroy(model); md_option_destroy(opt);
    std::puts("OK -> capi_ocr_out.jpg"); return 0;
}
```

- [ ] **Step 2: 重写 demo_lpr_pipeline_capi.cpp（LPR pipeline）**
- kind `MD_MODEL_LPR_PIPELINE`；路径 `../../test_data/test_models/onnx/yolov5plate.onnx|../../test_data/test_models/onnx/plate_recognition_color.onnx`
- 结果 `md_result_lpr` + `md_result_plate(i,&plate,&color)`；绘制；落盘 `capi_lpr_out.jpg`
- 输入 `../../test_data/test_images/` 下任一有车牌的图（若存在；按 test_data 实际图片命名）

- [ ] **Step 3: 重写 demo_pedestrian_attribute_capi.cpp（PED_ATTR）**
- kind `MD_MODEL_PED_ATTR`；路径聚合 det+attr：`../../test_data/test_models/onnx/zhgd_det.onnx|../../test_data/test_models/onnx/zhgd_ml.onnx`
- 参数：`md_model_set_param_d(model,"det_threshold",0.5)`（PED_ATTR 支持 det_threshold）
- 调用 `md_model_set_cls_input_size(model, 192, 256)` 与 `md_model_set_input_size(model, 1280, 1280)`（对齐 cxx 示例）
- 结果 `md_result_attribute` + `md_result_attr_scores(i,&scores,&n)`；绘制；落盘 `capi_attr_out.jpg`
- 输入 `../../test_data/test_images/test_pedestrian_attribute1.jpg`

- [ ] **Step 4: 编译 + 逐个运行验证**

构建并运行三个目标（`demo_ocr_capi`、`demo_lpr_pipeline_capi`、`demo_pedestrian_attribute_capi`），Expected 实际推理输出 + 落盘。

> `demo_ocr_recognition_capi.cpp`/`demo_ocr_recognition_capi_batch.cpp`/`demo_pp_structure_table_capi.cpp`/`demo_lpr_detection_capi.cpp`/`demo_lpr_recognizer_capi.cpp`：若这些 v1 示例在 capi 中有等价（batch 用 `md_model_predict_batch`，structure table 可能无 capi 等价），重写或收编；无法等价时从 CMake 移除目标并**在报告里说明**（不臆测保留带错误引用的文件）。实施者按每个文件实际情况决定，并在报告列出处置。

- [ ] **Step 5: 提交**

```bash
git add examples/demo_ocr examples/demo_lpr examples/demo_pipeline
git commit -m "feat(examples): rewrite OCR/LPR/pedestrian attr capi demos to capi"
```

---

### Task 5: capi 重写——音频（ASR/TTS）examples

**Files:**
- Modify: `examples/demo_audio/demo_kokoro_capi.cpp`（TTS）
- Create/Modify: `examples/demo_audio/` 若需 ASR capi example（对应 v1 sense_voice capi，如存在）

**Interfaces:**
- Consumes: capi `md_model_create(kind=MD_MODEL_TTS/ASR, ...)`、`md_audio_tts`、`md_audio_asr_wav`、`md_wav_save`
- Produces: 可运行的音频 capi example

- [ ] **Step 1: 重写 demo_kokoro_capi.cpp（TTS）**

```cpp
#include "capi_common.h"
int main() {
    MDOptionHandle opt; md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT); md_option_set_device(opt, MD_DEV_CPU);
    MDModelHandle model;
    die(md_model_create(&model, MD_MODEL_TTS,
        "../../test_data/test_models/onnx/kokoro_v1_1/model.onnx|"
        "../../test_data/test_models/onnx/kokoro_v1_1/tokens.txt|"
        "../../test_data/test_models/onnx/kokoro_v1_1/lexicon-gb-en.txt|"
        "../../test_data/test_models/onnx/kokoro_v1_1/lexicon-zh.txt|"
        "../../test_data/test_models/onnx/kokoro_v1_1/voices.bin|"
        "../../test_data/test_models/onnx/kokoro_v1_1/dict|"
        "../../test_data", opt), "create tts");
    int sample_rate=0; const float* audio=nullptr; size_t n=0;
    die(md_audio_tts(model, "你好世界今天天气不错 hello world", "zf_001", 1.0f,
                     &sample_rate, &audio, &n), "tts");
    std::printf("synthesized %zu samples @ %d Hz\n", n, sample_rate);
    die(md_wav_save(audio, n, sample_rate, "capi_tts_out.wav"), "save wav");
    md_model_destroy(model); md_option_destroy(opt);
    std::puts("OK -> capi_tts_out.wav"); return 0;
}
```

- [ ] **Step 2: ASR capi example（若 v1 有 sense_voice capi）**
用 kind `MD_MODEL_ASR`，路径 `../../test_data/test_models/onnx/sense_voice/model.int8.onnx|tokens.txt`，`md_audio_asr_wav(model, "../../test_data/test_models/onnx/sense_voice/test_wavs/zh.wav", &text)` 打印文本。若 v1 无 sense_voice capi 示例则跳过并说明。

- [ ] **Step 3: 编译 + 运行验证**
构建并运行 `demo_kokoro_capi`（+ASR 若有），Expected 输出采样数 + 保存 wav（实际合成）。注意 TTS 需模型文件存在（kokoro 已确认存在）。

- [ ] **Step 4: 提交**

```bash
git add examples/demo_audio
git commit -m "feat(examples): rewrite audio capi demos to capi TTS/ASR"
```

---

### Task 6: Image examples（demo_image）capi/null 处理

**Files:**
- Modify: `examples/demo_image/demo_image_from_base64.cpp`
- Modify: `examples/demo_image/demo_image_from_bgr24.cpp`
- （`demo_image_rotate.cpp` 若用 v1 capi 则重写为 capi，否则为 C++ 保留）

**Interfaces:**
- Consumes: `md_image_from_base64`、`md_image_from_bgr24`、`md_image_size`、`md_image_save`
- Produces: 可运行的 image utils capi example

- [ ] **Step 1: 评估并重写**
读三个文件，判断各自是 C++ 还是 v1 capi：
- 若引用 `capi/utils/md_*.h` → v1 capi，重写为 capi（`md_image_from_base64`、`md_image_from_bgr24`，构造 BGR 样本 → `md_image_size` → `md_image_save`）
- 若用 C++ SDK 且用 `imshow` → 改 `save`
- 重写后编译 + 运行验证落盘
- （此 Task 小，可与 Task 3/4 合并提交；若单独提交用 `git commit -m "feat(examples): rewrite image utils demos to capi"`）

---

### Task 7: C++ `.cxx` examples 修复 GPU/imshow + 验证

**Files:**
- Modify: 遍历 `examples/**/*.cxx`（及非 capi `.cpp`）中带 `use_gpu`/`enable_trt`/`use_trt_backend`/`imshow`/`waitKey` 的文件
- 涉及：demo_det/pose/obb/iseg/sem/depth/cls/lpr/ocr/face/pedestrian 等（以 grep 结果为准）
- Test: 每个改动目标编译 + 实际推理运行

**Interfaces:**
- Consumes: C++ SDK（`csrc/vision.h`），无需 capi
- Produces: 本机 CPU 可编译可运行的 C++ examples

- [ ] **Step 1: 全量 grep 清单**

```
rg -ln "use_gpu|enable_trt|use_trt_backend|imshow|waitKey|use_mnn_backend" examples --glob "*.cxx" --glob "*.cpp"
```
逐文件建立清单。

- [ ] **Step 2: 逐文件最小修复**
对每个匹配文件：
- `option.use_gpu(0)`、`option.enable_trt`、`option.use_trt_backend()` → 替换为 CPU 可跑（`use_ort_backend()` 已隐含；移除 GPU/TRT 选项；`enable_fp16` 对 CPU ORT 可忽略或移除）
- `option.use_mnn_backend()`（若用 mnn）→ CPU 下保留或改 ORT
- `img.imshow("...")`/`cv::imshow`/`vis_image.imshow` → `img.imwrite`/`save("..._out.jpg")`（或 `#ifdef` 有头环境才 show）
- 保留模型路径（沿用文件里已有 test_data 路径）
- 若有 `set_*_input_size`/`set_*_threshold` 等 C++ SDK setter 调用，**保留**并额外演示一次（这些是对应 C++ 参数 API）

> 实施者务必读取每个文件再改，不臆测；对每个改动目标编译 + 运行验证出结果。效率起见，可把"C++ 修复 + 验证"按 demo 目录分批（如 batch 1: det/kps/obb/iseg；batch 2: cls/sem/depth/lpr；batch 3: ocr/face/ped）。

- [ ] **Step 3: 分批编译 + 运行验证**
逐目录编译所有改动的 `.cxx` 目标，并在 `build_tdc/bin` 运行。Expected：实际推理输出 + 落盘图片（`*_out.jpg`）。避免 GPU/TRT 选项导致 init 失败。

- [ ] **Step 4: 提交（按逻辑分批）**

```bash
git add examples/demo_det examples/demo_kps examples/demo_obb examples/demo_iseg
git commit -m "fix(examples): C++ demos -> CPU/ORT + save instead of imshow (part 1)"
```
（后续批次依此类推：part 2 = cls/sem/depth/lpr；part 3 = ocr/face/ped。）

---

### Task 8: C# example 全面启用 + 新 API 演示

**Files:**
- Modify: `csharp/ModelDeployExample/Program.cs`
- Test: `dotnet build csharp/ModelDeployExample/ModelDeployExample.csproj -c Debug` + 运行

**Interfaces:**
- Consumes: `Model.SetParam/ParamNames/ParamType`（Task 3-of-param-plan 已实现）、既有 `DetectionModel/ClassificationModel/PoseModel/OcrModel/InsightFaceModel/SenseVoiceModel/KokoroModel`
- Produces: 覆盖多模型的 C# example + SetParam 演示

- [ ] **Step 1: 启用各测试 + 新增 SetParam 演示**

在 `TestDetection()` 里模型创建后、predict 前插入参数设置（演示新 API）：
```csharp
        det.SetParam("conf_threshold", 0.4);
        det.SetParam("nms_threshold", 0.45);
        Console.WriteLine("params: " + string.Join(", ", det.ParamNames()));
```
- 取消注释并启用 `TestClassification()`、`TestPose()`、`TestOCR()`、`TestInsightFace()`（模型均已确认在 test_data）
- `TestSenseVoice()`/`TestKokoro()`：模型存在则启用（sense_voice/kokoro onnx 已确认存在），运行 ASR/TTS 并落盘 wav
- 在 `Main` 去掉对应方法的 `//` 注释，按顺序调用
- 保持 `TestImage()`、`TestDetection()` 现有逻辑

- [ ] **Step 2: 编译 + 运行验证**

```
cd E:\CLionProjects\ModelDeploy\csharp
dotnet build ModelDeployExample/ModelDeployExample.csproj -c Debug
```
Expected 0 error/0 warning。然后运行，Expected 各模型实际推理输出（detected objects / 分类结果 / OCR 文本 / 人脸 / ASR 文本 / TTS wav），并打印 params 列表。

> 若某模型（如 insightface 需要多 onnx）在 CPU ORT 下加载/推理失败，报告该障碍（可能是模型/后端约束），不强制但必须尝试跑通；确无法跑通的记录说明。

- [ ] **Step 3: 提交**

```bash
git add csharp/ModelDeployExample/Program.cs
git commit -m "feat(csharp): enable all model demos + set_param demonstration in example"
```

---

### Task 9: 全量构建回归（CPU）+ 汇总验证

**Files:**
- Test: 全量 `cmake --build build_tdc`（含所有 examples）
- Docs: 无新增

**Interfaces:**
- Consumes: 全部 Task 1-8 产出

- [ ] **Step 1: 全量构建**

```
cd E:\CLionProjects\ModelDeploy
"<VS>\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build build_tdc --parallel
```
Expected: 0 error（所有非平台特定 example 目标编译通过；sophgo/trt 目标被 CMake 保护不参与）。

- [ ] **Step 2: 汇总运行验证清单**
对每个新增/改动 example 目标，在 `build_tdc/bin` 运行并记录：目标名、实际推理结果摘要（检测数/分类 topN/OCR 文本/合成 wav 等）、落盘文件。
生成 `examples/EXAMPLES.md`（若无）或更新现有说明，列每个 example 的用途、所需模型、运行命令（用确认可跑的命令）。

- [ ] **Step 3: 清理未跟踪产物**
确认 build 产物（`*_out.jpg`、`capi_*`）不在 working tree 中被误提交（若在根目录生成，确认删除或用 .gitignore）。

- [ ] **Step 4: 提交（文档）**

```bash
git add examples/EXAMPLES.md
git commit -m "docs(examples): catalog runnable demos with model deps + run commands"
```

---

## Self-Review

**Spec 覆盖：**
- A. capi 重写（22 个）→ Task 3/4/5/6
- B. 修 CMake 保护 → Task 2
- C. C# 全面启用 + 新 API → Task 8
- D. C++ `.cxx` 修复 → Task 7
- 全量回归验收 → Task 9
- 新 API 演示（md_model_set_param_* 在各 capi + C#）→ Task 3/4/8

**占位符扫描：**
- Task 5 Step 2（ASR）"若 v1 无则跳过"——条件性，但明确指示。可接受。
- Task 4 Step 4（ocr batch/structure/lpr 子示例处置）明确"报告处置"，非 TBD——给定了判定规则。
- Task 7 模型路径"沿用文件已有路径"——实施者读文件获取，非臆测。可接受。
- 无 TBD/TODO/未定义类型。

**类型/命名一致性：**
- `die()` 在 Task 1 定义，Task 3/4/5 复用——一致。
- `MDModelKind` 常量（MD_MODEL_DETECTION/OCR/TTS/...）与 capi/md_capi.h 一致。
- `md_model_set_param_*` 名称与 Task 2-of-param-plan 一致。
- `md_result_{detection,pose,obb,instance_seg,classification,face,ocr,lpr,attribute}` 均存在于 md_capi.h。
- C# `SetParam/ParamNames` 与既有绑定一致。
