# ModelDeploy 全模型性能分析报告（2026-08）

本文档基于新增的 `benchmark/benchmark_models.cpp`（全模型 + 全后端 pre/infer/post 分解）
与 `tests/test_pipelines.cpp`（多阶段 pipeline 回归）实测数据。

## 1. 测试环境

| 项 | CPU 构建 | GPU 构建 |
|---|---|---|
| 构建目录 | build_tdc | build_tdc_gpu |
| 后端 | ORT + MNN | ORT + TRT |
| 设备 | CPU（ORT CPU EP / MNN CPU） | GPU RTX 4060 Ti（TRT FP16） |
| 图片 | test_data/test_images/* | 同左 |

- 单模型跑 20 次（insightface 10 次）取平均，`pre/infer/post` 用 `TimerArray` 分解。
- 后端由模型文件后缀自动推断：`.onnx`→ORT、`.mnn`→MNN、`.engine`→TRT。
- 运行：`build_tdc/bin/benchmark.exe "[all_models][benchmark]"`、`"[pipeline][benchmark]"`。

## 2. 单模型耗时（CPU ORT vs MNN）

| 模型 | 输入 | ORT pre | ORT infer | ORT post | ORT total | MNN total |
|---|---|---|---|---|---|---|
| det yolo11n | 640 | 7.97 | 23.44 | **463.29** | **494.70** | 45.60 (mnn 内 NMS) |
| cls yolo11n-cls | 224 | - | - | - | 3.57 | 4.44 |
| obb yolo11n-obb | 640 | 10.96 | 23.10 | 18.26 | 52.32 | 47.56 |
| pose yolo11n-pose | 640 | 8.51 | 24.38 | 14.63 | 47.52 | 44.52 |
| seg yolo11n-seg | 640 | 8.33 | 32.77 | 34.93 | 76.03 | 59.99 |
| face-det scrfd | 640 | 10.42 | 10.73 | 0.10 | 21.24 | - |
| face-age | 112 | - | - | - | 5.75 | - |
| face-gender | 112 | - | - | - | 1.19 | - |
| face-rec | 112 | 1.75 | 6.63 | 0.03 | 8.41 | - |
| lpr-det yolov5plate | 640 | 6.76 | 13.23 | 4.03 | 24.01 | - |
| lpr-rec plate_rec | 48 | 0.23 | 0.37 | 0.01 | 0.62 | - |
| ocr-det ppocrv4 | 960 | 17.19 | 51.66 | 9.38 | 78.23 | - |
| ocr-rec ppocrv4 | 动态 | 0.34 | 6.69 | 0.22 | 7.25 | - |
| ocr-cls ppocrv4 | 48 | - | - | - | 1.46 | - |
| insightface det_10g | 640 | 5.39 | 38.15 | 2.15 | 45.69 | - |
| insightface 2d106 | 192 | 0.39 | 2.70 | 0.01 | 3.11 | - |
| insightface 1k3d68 | 192 | 0.39 | 11.62 | 0.02 | 12.03 | - |
| insightface w600k | 112 | 0.25 | 37.72 | 0.00 | 37.97 | - |
| insightface genderage | 96 | 0.09 | 0.28 | 0.00 | 0.37 | - |

## 3. 单模型耗时（GPU：ORT-onnx vs TRT-engine）

| 模型 | ORT total | TRT total | 加速 |
|---|---|---|---|
| det yolo11n | 118.17 | **11.78** | 10x |
| cls yolo11n-cls | 47.80 | **1.59** | 30x |
| obb yolo11n-obb | 676.21 | **6.15** | 110x |
| pose yolo11n-pose | 680.97 | **6.36** | 107x |
| seg yolo11n-seg | 823.60 | **23.89** | 34x |

> 注：GPU 构建下 onnx 模型走 ORT CUDA EP，某些模型（obb/pose/seg）infer 异常慢
> （600-800ms），远不如 CPU ORT（23-33ms）。这是 ORT CUDA EP + 该模型图优化的
> 问题，建议此类模型直接走 TRT engine。

## 4. Pipeline 耗时（CPU ORT，单图）

| Pipeline | total | 说明 |
|---|---|---|
| insightface（det+2d106+3d68+rec+genderage） | 49.41 | 结构良好，infer 占绝对主导 |
| face-rec（scrfd + seetaface rec） | 81.75 | det + rec batch |
| face-as（scrfd + fas_first + fas_second） | 133.57 | fas_second 整图推理 + 每脸串行 first |
| pedestrian-attr（zhgd_det + zhgd_ml） | 220.83 | 1280 大图 det + 每行人 cls |
| **OCR（det+cls+rec）** | **~2000** | **det 后处理 + rec 逐行推理是最大瓶颈** |

## 5. 瓶颈分析（耗时点 / 堵点 / 影响吞吐量的点）

### ★ 首要瓶颈：det 后处理 O(n²) NMS（`csrc/vision/utils.cpp:307-321`）

**现象**：`yolo11n.onnx`（无内嵌 NMS）post 处理 **463ms**；同模型 MNN（模型内 NMS）post 仅 **0.008ms**。

**根因**：
- 非 NMS 模型输出 `[1,84,8400]`，过阈值候选框数百~上千个；
- `utils::nms` 是 O(n²) 双层循环（utils.cpp:307-321），每对框调用 `rect2f_to_cv_type` 两次 +
  `cv::Rect2f` 的 `operator&` + 3 次 `area()`（utils.cpp:249-284），全是函数调用级开销；
- 候选 1000 个 → ~50 万对 × ~1µs ≈ 500ms，与实测吻合。

**影响**：所有走 C++ NMS 的模型（det/obb/pose/seg/lpr/行人属性）后处理都被拖慢；
行人属性 1280 大图候选框更多，O(n²) 放大。

**优化方向**：
1. 用向量化 FastNMS / 按类别分组 NMS（class-aware），候选数骤降；
2. 去掉 `cv::Rect2f` 中转，内联标量 IoU；
3. 换带内嵌 NMS 的模型导出（如 MNN 同款），post 归零。

### 次要瓶颈 1：OCR 逐行推理 + 整图拷贝（ppocr.cpp + image_data.cpp）

**现象**：OCR pipeline ~2000ms。

**根因**：
- `cls_batch_size_=1`（ppocr.h:68）：每个文字行一次 ORT Run，L 行 = L 次串行推理；
- `rec_batch_size_=6`（ppocr.h:69）+ 动态宽 pad 到最长行（rec_preprocessor.cpp:38-46）：L 行分
  `ceil(L/6)` 次，行宽参差时浪费严重；
- `ImageData::rotate_crop`（image_data.cpp:175）每行 `copyTo` **整张原图**再透视变换，L 行 = L 次
  2.7MB 整图拷贝；
- DB 后处理每轮廓 ClipperLib 偏移（ocr_postprocess_op.cpp:40-42）+ 每框 mask 分配/fillPoly/mean
  （:176-207）。

**优化方向**：cls/rec batch 提到 16-32；rotate_crop 先裁 ROI 再透视；DB 后处理用固定缓冲 +
  cv::threshold 替代逐像素循环。

### 次要瓶颈 2：活体检测整图 clone + 串行 first（face_as_pipeline.cpp:130-133,145）

**现象**：133ms。两次整图 clone（:130-131）+ fas_second 整图推理 + 每张脸串行 fas_first（无 batch）。

### 次要瓶颈 3：行人属性 1280 det（pedestrian_attribute.cpp:97）

**现象**：220ms。1280×1280 det（infer + 放大版 O(n²) NMS）+ 每行人独立 cls。

### 次要瓶颈 4：pre 阶段 SIMD 依赖（fused_preproc_simd.cpp）

det pre 8ms / ocr-det pre 17ms 走融合核，MSVC x64 默认 AVX2 已启用，标量兜底核仍存在。
AVX512 构建或双线性核 SIMD 化可进一步压缩。

## 6. 代码结构 / 质量审查发现

### 6.1 严重：回归测试"假通过"（已修复）

- **test_vision_models.cpp 模型路径错误**：`model_path("yolo11n-cls.onnx")` 解析到
  `test_models/yolo11n-cls.onnx`，实际文件在 `test_models/onnx/yolo11n/`。全部 13 个用例因
  `if(!fs::exists) return` 静默跳过，**0 断言**。已修正路径 + 修 DBDetector name 断言，
  现为 **6664 断言真通过**（7795 总断言）。
- **baseline_compare.cpp 依赖 baseline JSON**：`tests/baseline/` 目录为空，所有 ORT/MNN/TRT
  回归用例因缺 baseline 文件 return 跳过，未真正验证推理。

### 6.2 中：测试路径与构建配置耦合

- GPU 构建（ENABLE_MNN=OFF）下跑 baseline_compare 的 MNN 用例，`.mnn` 文件被 ORT 尝试加载
  → protobuf failed → 9 个用例失败。应在用例内按 `#ifdef ENABLE_MNN` 门控（同
  test_insightface.cpp 的做法），而不是依赖模型文件存在性。

### 6.3 中：LPR 检不出车牌的代码风险（lpr_pipeline.cpp:100-102）

- `det_result[i].keypoints.size() != 4` 直接 `return false`（整帧无输出），应跳过该车而非整体失败；
- lpr_det/postprocessor.cpp:43 `confidence = obj_conf * cls_conf`，cls_conf 未做 sigmoid，
  端到端模型会因阈值 0.25 滤掉真实车牌（对比检测 postprocessor.cpp:64 有 sigmoid）。

### 6.4 中：face-age / face-gender predict 无 TimerArray

- `SeetaFaceAge::predict` / `SeetaFaceGender::predict` 不带 `TimerArray*`，无法在 pipeline
  层分解耗时（benchmark 里只能整体计时）。建议统一签名。

### 6.5 低：insightface pipeline create_from_dir 只拼 .onnx

- `InsightFaceAnalysis::create_from_dir` 硬编码 `.onnx` 后缀，无法用于 MNN/TRT 目录。
  benchmark 已绕过（显式构造），但 API 本身有局限。

### 6.6 低：pedestrian attribute size 语义 w/h 与模型 H/W 易错

- `set_cls_input_size` 语义为 `{w,h}`，而模型输入标注为 `[C,H,W]`。zhgd_ml 需传
  `{192,256}`（w=192,h=256）但模型打印 `[0,3,256,192]`（H=256,W=192），极易配反。
  建议在文档/注释中明确，或自动从模型输入推断。

### 6.7 正面：架构一致性

- 标准 BaseModel + Preprocessor + Postprocessor + Runtime 架构统一，多后端由文件后缀
  自动推断，insightface 已完全对齐该架构；
- 融合预处理（fused_preprocess）+ 运行时 ISA 派发设计良好；
- 设备内存与 Tensor 解耦（B 方案）、CAPI/pybind/Rust/C# 绑定齐全。

## 7. 吞吐量结论

| 场景 | CPU 单线程 | GPU TRT |
|---|---|---|
| det 单帧 | 45 fps（用 NMS 模型） | 85 fps |
| insightface 全流程 | 20 fps | - |
| OCR 整页 | 0.5 fps | - |
| face-rec pipeline | 12 fps | - |

**提升吞吐量的关键动作**（按优先级）：
1. **修复 O(n²) NMS**（最快收益，det/行人/OCR/LPR 全受益，预计 det post 463ms→<10ms）；
2. **OCR batch 化**（cls batch=1 → 16，rec batch=6 → 32），预计 OCR 2000ms→<600ms；
3. 生产环境换带内嵌 NMS 的模型 + GPU TRT engine。

## 8. 复现命令

```bash
# 回归
TEST_DATA_DIR=repo cmake 构建后:
build_tdc/bin/test_modeldeploy        # CPU 全量（147 用例 / 7795 断言）
build_tdc_gpu/bin/test_modeldeploy    # GPU 全量（152 用例，9 个为 MNN 用例在无 MNN 构建下失败）

# 性能
build_tdc/bin/benchmark.exe "[all_models][benchmark]"   # CPU ORT + MNN
build_tdc_gpu/bin/benchmark.exe "[all_models][benchmark]"  # GPU ORT + TRT
build_tdc/bin/benchmark.exe "[pipeline][benchmark]"     # pipeline
```
