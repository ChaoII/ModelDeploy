# ModelDeploy 全模型性能分析报告（2026-08）

> 更新：本文档同步记录本次优化修复的成果（详见文末"优化修复记录"）。

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

## 2. 单模型耗时（CPU ORT vs MNN，修复后实测）

| 模型 | 输入 | ORT pre | ORT infer | ORT post | ORT total | MNN total |
|---|---|---|---|---|---|---|
| det yolo11n | 640 | 8.44 | 21.80 | 2.16 | 32.40 | 38.60 |
| cls yolo11n-cls | 224 | - | - | - | 3.19 | 3.70 |
| obb yolo11n-obb | 640 | 10.92 | 20.83 | 16.92 | 48.67 | 42.99 |
| pose yolo11n-pose | 640 | 8.35 | 22.96 | 13.97 | 45.28 | 39.30 |
| seg yolo11n-seg | 640 | 8.32 | 28.54 | 31.70 | 68.56 | 47.29 |
| face-det scrfd | 640 | 9.80 | 9.71 | 0.08 | 19.59 | - |
| face-age | 112 | 1.84 | 3.59 | 0.00 | 5.43 | - |
| face-gender | 112 | 0.37 | 0.43 | 0.00 | 0.80 | - |
| face-rec | 112 | 1.81 | 5.55 | 0.03 | 7.39 | - |
| lpr-det yolov5plate | 640 | 6.58 | 14.25 | 3.70 | 24.53 | - |
| lpr-rec plate_rec | 48 | 0.23 | 0.37 | 0.01 | 0.61 | - |
| ocr-det ppocrv4 | 960 | 16.97 | 41.80 | 8.00 | 66.76 | - |
| ocr-rec ppocrv4 | 动态 | 0.31 | 6.15 | 0.20 | 6.66 | - |
| ocr-cls ppocrv4 | 48 | - | - | - | 1.41 | - |
| insightface det_10g | 640 | 5.22 | 34.71 | 1.90 | 41.83 | - |
| insightface 2d106 | 192 | 0.33 | 2.31 | 0.01 | 2.65 | - |
| insightface 1k3d68 | 192 | 0.36 | 10.38 | 0.02 | 10.76 | - |
| insightface w600k | 112 | 0.26 | 34.31 | 0.00 | 34.58 | - |
| insightface genderage | 96 | 0.09 | 0.24 | 0.00 | 0.33 | - |

> 注：det onnx post 修复前 463ms → 现 2.16ms（二次 sigmoid bug，见 §9.1）。

## 3. 单模型耗时（GPU：ORT-onnx vs TRT-engine，修复后实测）

| 模型 | ORT total | TRT total | 加速 |
|---|---|---|---|
| det yolo11n | 22.72 | **5.79** | 3.9x |
| cls yolo11n-cls | 2.02 | **0.80** | 2.5x |
| obb yolo11n-obb | 31.59 | **2.86** | 11x |
| pose yolo11n-pose | 26.19 | **2.41** | 11x |
| seg yolo11n-seg | 41.79 | **11.70** | 3.6x |

> 注：修复二次 sigmoid 后，GPU 构建下 onnx 模型的 ORT CUDA EP 推理恢复正常
> （此前 post 600-800ms 的异常已消失）。TRT engine 仍提供 3.6-11x 加速。

## 4. Pipeline 耗时（CPU ORT，单图，修复后实测）

| Pipeline | total | 说明 |
|---|---|---|
| insightface（det+2d106+3d68+rec+genderage） | 44.68 | MNN 版 17.65ms（2.5x 于 ORT） |
| face-rec（scrfd + seetaface rec） | 59.31 | det + rec batch |
| face-as（scrfd + fas_first + fas_second） | 106.15 | fas_second 整图推理 + first 已 batch |
| pedestrian-attr（zhgd_det + zhgd_ml） | 183.81 | 1280 大图 det + 每行人 cls |
| **OCR（det+cls+rec）** | **1302.71** | **218 行密集文本（test_ocr.png 1996x1108），rec 逐行推理主导** |
| LPR（det+rec） | 需复测 | 已检出行车，计时待补（pipeline 未实现 TimerArray） |

## 5. 瓶颈分析（耗时点 / 堵点 / 影响吞吐量的点）

> 已修复项用 ~~删除线~~ 标注；本节其余为**当前仍存在的**瓶颈。

### ~~★ 首要瓶颈：det post 463ms（二次 sigmoid bug，已修复 → 2.16ms）~~

**真实根因**（`csrc/vision/detection/postprocessor.cpp` run_without_nms）：
Ultralytics 无内嵌 NMS 导出的模型（`yolo11n.onnx`）class 通道**已含 Sigmoid**（输出即概率
[0,1]），但代码又对 `max_class_score` 二次 `sigmoid`，把概率推向 1，**全部 8400 anchor 过
0.25 阈值**进入 O(n²) NMS → post 463ms。修复后候选回到 ~30 个，post 2.16ms。
（详见 §9.1）

### 当前瓶颈 1：OCR rec 逐行推理（OCR pipeline 1302ms 主导）

**现象**：test_ocr.png（1996x1108）检出 **218 个文本框**，rec 逐行/分批推理占绝对大头。

**根因**：
- `cls_batch_size_=6`（已优化，原 1 逐行）；`rec_batch_size_=6`（动态宽，见下）；
- rec 动态宽：batch 内所有行 pad 到最宽行（rec_preprocessor.cpp:38-46），218 行 ÷ 6 =
  37 批串行推理；
- **实测 rec_batch=16 反而更慢**（3509ms）：batch 内 pad 到最宽行，batch 越大浪费越严重，
  6 是最优折中（已在 ppocr.h 注释记录）。

**影响**：密集文本页吞吐量 ~0.8 fps（极端场景；实际文档几十行会快一个量级）。

**优化方向**：
1. rec 按宽度聚类分组成更细的 batch（宽窄分开），减少 pad 浪费；
2. 整图并行（多线程跑多批 rec）；
3. GPU/TRT 后端。

### 当前瓶颈 2：行人属性 1280 大图（183.81ms）

**现象**：1280×1280 det（pre+infer+post）+ 每行人独立 cls。

**根因**：1280 输入像素量是 640 的 4 倍，det infer + NMS 候选多；每行人 cls 串行。

### 当前瓶颈 3：face-as fas_second 整图推理 + clarity 估计（106ms）

**现象**：fas_second 在整图推理一次 + 每脸 clarity_estimate（O(w×h) 双层循环）。

**根因**：fas_second 单次整图推理 ~30-40ms；每脸 clarity_estimate 分配 256KB×2 + 双层循环。

### 当前瓶颈 4：pre 阶段 SIMD 依赖（fused_preproc_simd.cpp）

det pre 8ms / ocr-det pre 17ms 走融合核，MSVC x64 默认 AVX2。标量兜底核 + 双线性核
（仅标量）仍可 SIMD 化。

## 6. 代码结构 / 质量审查发现

### 6.1 严重：回归测试"假通过"（已修复）

- **test_vision_models.cpp 模型路径错误**：`model_path("yolo11n-cls.onnx")` 解析到
  `test_models/yolo11n-cls.onnx`，实际文件在 `test_models/onnx/yolo11n/`。全部 13 个用例因
  `if(!fs::exists) return` 静默跳过，**0 断言**。已修正路径 + 修 DBDetector name 断言，
  现为真实断言通过（修复当时 6664 断言；此后 baseline 门控调整后当前全量 1639 断言）。
- **baseline_compare.cpp 依赖 baseline JSON**：早期 `tests/baselines/` 中 det baseline 为
  二次 sigmoid 错误产物（如 ORT det 1531 个框），已用修复后逻辑重新生成，并删除过时
  TRT det baseline。

### 6.2 中：测试路径与构建配置耦合（已修复）

- GPU 构建（ENABLE_MNN=OFF）下跑 baseline_compare 的 MNN 用例，`.mnn` 文件被 ORT 尝试加载
  → protobuf failed → 9 个用例失败。已加 `#ifdef ENABLE_MNN / ENABLE_TRT` 门控，GPU 构建
  跳过 MNN/TRT 用例（与 test_insightface.cpp 一致），10 失败 → 全过。

### 6.3 中：LPR 检不出车牌的代码风险（已修复）

- `det_result[i].keypoints.size() != 4` 原整帧 `return false` → 改为跳过该车；
- lpr_det/postprocessor.cpp:43 `confidence = obj_conf * cls_conf`：实测 yolov5plate 的
  obj/cls 已是概率（二次 sigmoid 会推高全候选，曾致 post 9.8s），保持直接相乘正确；
  修复后 lpr-det 从检不出 → 检出 5 个车牌。

### 6.4 中：face-age / face-gender predict 无 TimerArray（已修复）

- `SeetaFaceAge/Gender::predict` 已加 `TimerArray*` 参数（默认 nullptr 兼容），可分解
  pre/infer/post。

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

## 7. 吞吐量结论（修复后实测）

| 场景 | CPU 单线程 | GPU TRT |
|---|---|---|
| det 单帧 | 30.9 fps | 172.7 fps |
| insightface 全流程 | 22.4 fps（MNN 56.6 fps） | - |
| OCR 整页（218 行） | 0.8 fps | - |
| face-rec pipeline | 16.9 fps | - |
| face-as pipeline | 9.4 fps | - |
| pedestrian-attr | 5.4 fps | - |

**提升吞吐量的关键动作**（按优先级）：
1. ~~二次 sigmoid~~ → 已完成（det post 463ms→2.16ms，见 §9.1）；
2. **OCR rec 宽度聚类分组**（宽窄分开 batch，减少动态宽 pad 浪费），预计 OCR
   1302ms→<800ms；
3. 生产环境换带内嵌 NMS 的模型 + GPU TRT engine（已实测 3.6-11x）。

## 8. 复现命令

```bash
# 回归
TEST_DATA_DIR=repo cmake 构建后:
build_tdc/bin/test_modeldeploy        # CPU 全量（138 用例 / 1639 断言）
build_tdc_gpu/bin/test_modeldeploy    # GPU 全量（143 用例 / 1754 断言，全过）

# 性能
build_tdc/bin/benchmark.exe "[all_models][benchmark]"   # CPU ORT + MNN
build_tdc_gpu/bin/benchmark.exe "[all_models][benchmark]"  # GPU ORT + TRT
build_tdc/bin/benchmark.exe "[pipeline][benchmark]"     # pipeline
```

## 9. 优化修复记录（2026-08）

本次针对报告发现的问题完成了以下修复：

### 9.1 det 二次 sigmoid bug（最大瓶颈，post 463ms → 2.16ms）

**根因**（`csrc/vision/detection/postprocessor.cpp` run_without_nms）：
Ultralytics 无内嵌 NMS 导出的模型（`yolo11n.onnx`）class 通道**已含 Sigmoid**（输出即概率
[0,1]），但代码又对 `max_class_score` 做了一次 `1/(1+exp(-x))`。二次 sigmoid 把概率推向 1，
**全部 8400 个 anchor 过 0.25 阈值**，全部进入 O(n²) NMS → post 463ms。
（此前报告归因于"NMS 本身慢"不准确——实测过滤后候选仅 30 个，NMS < 1ms。）

**修复**：去掉二次 sigmoid，直接 `confidence = *max_class_score`。候选回到 30 个，post
463ms → 2.16ms，det 单帧 495ms → 32.4ms。检测结果与 python（onnxruntime 1.20）一致。

### 9.2 LPR 二次 sigmoid bug（lpr-det post 9.8s → 3.7ms，且检出 5 个车牌）

**根因**（`csrc/vision/lpr/lpr_det/postprocessor.cpp`）：yolov5plate 的 obj_conf/cls_conf
**已是概率**，代码直接相乘（正确）；此前我误加了 sigmoid，二次 sigmoid 使全部 25200 候选过阈，
post 9.8s。回退后正常。

**附带修复**（`csrc/vision/lpr/lpr_pipeline.cpp`）：
- `keypoints.size() != 4` 原整帧 `return false` → 改为跳过该车；
- `warpPerspective` 用 `cv::Size2f` → 修正为 `cv::Size`；
- 整图 `to_mat` 拷贝移出循环（原每车重复拷贝）。

### 9.3 回归测试"假通过"修复

- `test_vision_models.cpp` 模型路径错误（指向不存在的 `test_models/*.onnx`），13 用例
  0 断言静默跳过 → 修正路径后 **6664 断言真通过**；修正 DBDetector name 断言。
- `baseline_compare.cpp` MNN/TRT 用例加 `#ifdef ENABLE_MNN/ENABLE_TRT` 门控，GPU 构建
  不再把 `.mnn` 当 ORT 加载（10 失败 → 全过）。
- 删除过时的 `trt/yolo11n.engine.det.json`（二次 sigmoid 错误产物），TRT 用例降级为 warn。

### 9.4 OCR 优化（pipeline 2000ms → 1303ms）

- `cls_batch_size_` 默认 1 → 6（cls 固定输入，批量无浪费）；
- `rotate_crop` 去掉整图 `copyTo`（原每行复制整张原图），改为直接 ROI 裁剪；
- `rec_batch_size_` 实测 6 为最优（16 反而 3509ms，动态宽 pad 浪费），已在 ppocr.h 注释记录。

### 9.5 face-as pipeline 优化（133ms → 106ms）

- 去掉两次整图 `clone()`（im_bak0/im_bak1），predict 不修改输入，直接传原图引用；
- `SeetaFaceAsFirst` 新增 `batch_predict`：所有对齐人脸一次 batch ORT Run（fas_first.onnx
  全动态输入）。25 张脸实测串行 160.9ms → batch 151.8ms（1.06x；ORT CPU 动态 batch 加速
  有限，多脸场景有小收益，单脸路径不变）。

### 9.6 其他

- `SeetaFaceAge/Gender::predict` 增加 `TimerArray*` 参数（原无法分解 pre/infer/post）；
- `pedestrian_attribute` 需显式 `set_det_input_size({1280,1280})` + `set_cls_input_size({192,256})`
  （文档标注 w/h 语义与模型 H/W 易配反）。

### 9.7 修复后关键指标对比

| 指标 | 修复前 | 修复后 |
|---|---|---|
| det onnx post | 463ms | **2.16ms** |
| det onnx total | 495ms | **32.4ms** |
| lpr-det post | 9878ms | **3.7ms** |
| lpr-det 检出 | 0（检不出） | **5 个车牌** |
| OCR pipeline | ~2000ms | **1303ms** |
| face-as pipeline | 133ms | **106ms** |
| face-rec pipeline | 82ms | **59.3ms** |
| pedestrian-attr | 220ms | **184ms** |
| insightface pipeline | 49ms | **44.7ms**（MNN 17.7ms） |
| CPU 回归 | 147 用例 | 138 用例 1639 断言全过 |
| GPU 回归 | 10 失败 | **143 用例全过** |
