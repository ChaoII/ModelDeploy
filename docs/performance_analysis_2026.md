# ModelDeploy 全模型性能分析报告（2026-08 全面版）

> 覆盖：每个模型 × {ORT CPU, MNN, TRT backend} 的 pre/infer/post 分解 + 每个 pipeline 的 total。
> 文档末尾记录本次性能优化的修复成果。

## 1. 测试环境与测法

| 项 | CPU 构建 | GPU 构建 |
|---|---|---|
| 构建目录 | build_tdc | build_tdc_gpu |
| 后端 | ORT CPU + MNN | TRT backend（trtexec 预编译 .engine，FP16） |
| 设备 | CPU | GPU RTX 4060 Ti（FP16） |
| 图片 | test_data/test_images/* | 同左 |

- 单模型跑 20 次（insightface 10 次）取平均，`pre/infer/post` 用 `TimerArray` 分解。
- **onnx → ORT CPU**（基线）；**.mnn → MNN**；**.engine → TRT backend**（GPU，生产推荐）。
- **TRT backend 说明**：每个模型的 .engine 由 trtexec 预编译（FP16），生产部署即此配置。
- **关于 ORT TRT-EP**（enable_trt=true）：仅对**内嵌 NMS/固定 shape 的模型**快
  （yolo11n_nms.onnx ≈7ms）；对动态 shape 的非 NMS onnx（obb/pose/seg 等）实测 **16 秒/帧**
  （图优化极差）。故生产 GPU 用 TRT backend（.engine），而非 ORT TRT-EP。
- 运行：GPU benchmark 在 `build_tdc_gpu` 目录根运行 `bin/benchmark.exe "[all_models][benchmark]"`。

## 2. 单模型耗时总表（CPU ORT 基线 vs GPU TRT backend）

| 模型 | ORT CPU (ms) | TRT GPU (ms) | TRT 加速 |
|---|---|---|---|
| det yolo11n_nms | 20.34 | **3.59** | 5.7x |
| cls yolo11n-cls | 2.00 | **0.80** | 2.5x |
| obb yolo11n-obb | 32.52 | **2.84** | 11.5x |
| pose yolo11n-pose | 27.86 | **2.51** | 11.1x |
| seg yolo11n-seg | 42.46 | **11.36** | 3.7x |
| face-det scrfd | 10.53 | **1.40** | 7.5x |
| face-gender | 0.39 | **0.19** | 2.1x |
| face-rec | 7.13 | **1.06** | 6.7x |
| lpr-det yolov5plate | 12.50 | **4.58** | 2.7x |
| lpr-rec | 0.35 | **0.23** | 1.5x |
| ocr-det | 60.57 | **7.40** | 8.2x |
| ocr-rec | 6.46 | **1.17** | 5.5x |
| ocr-cls | 1.31 | **0.62** | 2.1x |
| insightface det_10g | 37.74 | **5.37** | 7.0x |
| insightface 2d106 | 2.64 | **0.88** | 3.0x |
| insightface 1k3d68 | 10.25 | **1.25** | 8.2x |
| insightface w600k | 35.10 | **1.62** | 21.7x |
| insightface genderage | 0.31 | **0.47** | 0.7x |

> **模型缺陷（非转换问题）**：face-age（age_predictor.onnx）的 Gemm 权重维度错
> （ORT 推理也报 Dimension mismatch）、fas_second 的 Conv 动态 shape 无效（ORT 也崩）——
> 这两个模型文件本身有缺陷，**任何后端都无法推理**，非 TRT 转换限制。
> 所有 TRT 数据均为 trtexec 预编译 .engine 在 RTX 4060 Ti FP16 实测。

## 3. 单模型 pre/infer/post 分解（关键模型）

| 模型 | 后端 | pre | infer | post | total |
|---|---|---|---|---|---|
| det yolo11n_nms | ORT CPU | 0.97 | 19.38 | 0.00 | 20.34 |
| det yolo11n_nms | **TRT** | 0.24 | **3.35** | 0.00 | **3.59** |
| ocr-det | ORT CPU | 1.34 | 55.31 | 3.92 | 60.57 |
| ocr-det | **TRT** | 1.08 | **3.36** | 2.96 | **7.40** |
| insightface det | ORT CPU | 2.57 | 34.76 | 0.41 | 37.74 |
| insightface det | **TRT** | 2.21 | **2.84** | 0.31 | **5.37** |
| insightface rec | ORT CPU | 0.14 | 34.96 | 0.00 | 35.10 |
| insightface rec | **TRT** | 0.11 | **1.51** | 0.00 | **1.62** |

## 4. Pipeline 耗时（CPU vs GPU TRT backend，单图）

| Pipeline | ORT CPU | MNN | TRT GPU | 说明 |
|---|---|---|---|---|
| insightface（det+2d106+3d68+rec+genderage，17 脸，batch） | 978 | 1345 | **55** | batch 后 TRT 从 79.5→55ms |
| face-rec（scrfd + seetaface rec） | 69 | - | **2.8** | TRT 25x |
| face-as（scrfd + fas_first + fas_second） | 130 | - | onnx only（fas_second 模型缺陷） | - |
| pedestrian-attr（zhgd_det + zhgd_ml） | 202 | - | **21.4** | TRT 9.4x |
| **OCR（det+cls+rec，218 行，rec batch=8）** | **1474** | - | **199** | **TRT 7.4x** |
| LPR（det+rec） | 待补 | - | 待补 | 已检出，pipeline 计时未实现 |

> **insightface batch 优化**：17 张脸从逐脸串行（79.5ms）改为 batch 推理（每子模型一次
> [17,3,H,W] infer）→ TRT **55ms**（max24 profile）或 **18.7ms**（opt8 max16 profile）。
> CPU onnx 从 3413ms → 978ms（3.5x），MNN 1486→1345ms。

### 4.0 资源受限场景（SOPHGO/嵌入式）的模型裁剪
insightface pipeline 支持按需跳过子模型（`analyze` 的 with_2d106/with_3d68/with_recognition/
with_genderage 参数），资源受限设备可只保留关键模型：
- 仅人脸识别：`analyze(img, &r, false, false, true, false)`（跳过 landmark 和 age）
- 仅检测：`detect()`（只跑 det_10g）
- 人脸识别业务：`analyze_max_face()` 只识别**最大人脸**，`face_count` 报告画面人数
  （>1 可触发"多人警告"）；`FaceRecognizerPipeline::predict_max_face` 同理。

### 4.1 OCR 199ms 的构成（218 行密集文本，rec batch=8）
- det（960 max side，动态）：7.4ms
- cls：218 行 ÷ 6 = 37 批 × 0.62ms ≈ 23ms
- **rec：218 行 ÷ 8 = 28 批（batch 6→8 减少批次数），每批动态宽 pad 到该批最宽行**
  - rec 单行（48x320）仅 1.2ms，但 864px 宽行 pad 放大计算量 ~2.7x
  - **28 批 × ~6ms ≈ 165ms（主要瓶颈）**
- crop/透视变换（CPU）：218 次 ≈ 11ms

> **优化记录**：rec_batch 6→8（GPU 203→199ms，TRT engine 支持 batch 8）。宽度排序已
> 实现聚类（arg_sort 让相近宽度同批）；进一步优化需宽度分桶（>512px 宽行单独处理）。

## 5. 瓶颈分析（当前仍存在的优化点）

### 5.1 OCR rec 动态宽 pad（GPU 下 230ms 的主要成本，~185ms）
- 218 行 → rec batch=6，37 批；每批 pad 到该批最宽行（最长 864px），pad 浪费放大 2.7x
- rec 单行仅 1.2ms；**优化方向：按宽度聚类分组**（宽窄分开 batch），预计可省 ~100ms

### 5.2 insightface pipeline 逐脸推理（79.5ms，17 脸）
- 每张脸串行 lmk2d/lmk3d/rec/genderage（TRT 各 ~1ms）
- **优化方向**：人脸 batch 化（同 face-as first 的做法），多脸场景可显著提速

### 5.3 det post 的 CPU NMS
- 非内嵌 NMS 模型（obb/pose/seg onnx）post 在 CPU 做 NMS（3.9-10ms）。
- TRT engine 用内嵌 NMS 模型后 post≈0.005ms（已解决）。

### 5.4 face-as 的 fas_second
- fas_second 模型缺陷（Conv 动态 shape 崩），face-as 无法完整 TRT。

## 6. 吞吐量结论（fps，最新实测）

| 场景 | CPU | GPU TRT | 加速 |
|---|---|---|---|
| det 单帧 | 47 fps | **278 fps** | 5.9x |
| insightface 全流程（17 脸） | 0.29 fps | **12.6 fps** | 43x |
| face-rec pipeline | 12 fps | **319 fps** | 26x |
| pedestrian-attr | 5 fps | **62 fps** | 12.4x |
| OCR 整页（218 行） | 0.73 fps | **4.3 fps** | 6x |

## 7. 代码结构 / 质量审查发现（已修复项标注）

### 已修复
- **二次 sigmoid bug**（det 463ms→2.2ms；lpr 9.8s→3.7ms）：见 §9.1/§9.2
- **goto cleanup**：fused_preproc.cu 改用 fail() lambda（RAII 风格）
- **回归假通过**：test_vision_models 路径修正、baseline 后端门控
- **LPR 检不出**：keypoints!=4 跳过、Size2f→Size
- **face-as**：去整图 clone + first batch
- **face-age/gender** 加 TimerArray

### 待优化（低优先级）
- insightface `create_from_dir` 硬编码 .onnx
- pedestrian size 语义 w/h vs H/W 易配反
- LPR pipeline 未实现 TimerArray

## 8. 复现命令

```bash
# 回归
build_tdc/bin/test_modeldeploy        # CPU 全量（138 用例 / 1639 断言）
build_tdc_gpu/bin/test_modeldeploy    # GPU 全量（143 用例 / 1754 断言）

# 性能（GPU 在 build_tdc_gpu 根目录运行）
build_tdc/bin/benchmark.exe "[all_models][benchmark]"     # CPU ORT + MNN
cd build_tdc_gpu && ./bin/benchmark.exe "[all_models][benchmark]"  # GPU TRT backend
cd build_tdc_gpu && ./bin/benchmark.exe "[pipeline][benchmark]"    # pipeline
```

## 9. 优化修复记录（2026-08）

### 9.1 det 二次 sigmoid bug（最大瓶颈，post 463ms → 2.16ms）
Ultralytics 无内嵌 NMS 导出的 onnx class 通道已含 Sigmoid，代码又二次 sigmoid 把概率推向 1，
**全部 8400 anchor 过 0.25 阈值**进入 O(n²) NMS → 463ms。去掉二次 sigmoid 后候选回 ~30，
post 2.16ms。

### 9.2 LPR 二次 sigmoid bug（lpr-det post 9.8s → 3.7ms，检出 5 车牌）
yolov5plate 的 obj/cls 已是概率，误加 sigmoid 致全候选过阈。回退后正常 + 修 keypoints/Size 问题。

### 9.3 回归测试修复
- test_vision_models 路径（13 用例 0 断言 → 6664 断言真通过）
- baseline_compare 后端门控（GPU 构建 10 失败 → 全过）

### 9.4 OCR / face-as / 其他
- cls_batch 1→6；rotate_crop 去整图拷贝；rec_batch=6 最优（实测）
- face-as 去 clone + first batch；face-age/gender 加 TimerArray

### 9.5 本版新增（全模型 TRT 覆盖）
- 为 face/lpr/ocr/insightface/zhgd 生成全部 TRT engine（trtexec FP16）
- benchmark 每个模型测 {ORT CPU, MNN, TRT backend}，pipeline 测 {CPU, TRT}
- 修正 TRT engine 的 profile（det 动态 320-1280、cls/rec batch 8、rec 宽 1024）

### 9.6 本版修复：TimerArray 累计语义 bug（重要）
**问题**：`Timer::average_ms()` 返回"平均"而非"总和"。insightface analyze 把同一个
TimerArray 传给 det + 每张脸的 lmk2d/lmk3d/rec/genderage（17 脸 = 69 次 start/stop），
平均后严重低估——insightface pipeline 被误报为 **1.1ms**（实际 **79.5ms**）。

**修复**：`average_ms()` 改为返回计时段**总和**。单模型（一次 start/stop）sum=实际耗时不变；
pipeline（多子模型累积）sum=累计推理耗时，正确反映真实耗时。

**影响**：单模型数据不变；仅 insightface pipeline 数据修正（1.1ms→79.5ms）。
其他 pipeline（face-rec/pedestrian/OCR）内部用单次计时，未受影响。

### 9.7 模型缺陷确认
- **age_predictor.onnx**：Gemm219 权重维度错（W{1024,6144} K:1536，ORT 推理即报
  Dimension mismatch）——模型文件损坏，无法在任何后端推理。
- **fas_second.onnx**：动态 shape 下 Conv 输入 {1,1} 无效（ORT 也崩）——模型文件缺陷。
- 这两个不是"无法转 TRT"，而是模型本身不可用（应重新导出模型）。

### 9.8 insightface 子模型 batch 推理（17 脸 pipeline 79.5ms → 55ms，CPU 3.5x）
给 InsightFaceLandmark / InsightFaceRecognition / InsightFaceGenderAge 增加 batch_predict：
- 每张脸单图 preprocess → memcpy 合并 [N,3,H,W] → 一次 infer → 按行切分 postprocess
- face_analysis.analyze 改为 batch 调用（det 后 4 个子模型各一次 batch 推理）
- TRT engine 重新生成动态 batch profile（min 1 / opt 8-16 / max 24，覆盖实际脸数）

**实测（17 张脸）**：
| 后端 | 逐脸串行 | batch | 提速 |
|---|---|---|---|
| onnx CPU | 3413ms | 978ms | 3.5x |
| MNN | 1486ms | 1345ms | 1.1x |
| TRT（opt8 max16） | 79.5ms | **18.7ms** | 4.3x |
| TRT（opt16 max24，通用） | 79.5ms | **55ms** | 1.45x |

> **TRT profile 权衡**：opt=8 时引擎针对小 batch 优化最快，但严格 profile 下 batch 超出
> max 会失败；opt=16/max24 覆盖更多脸但性能略降。生产按场景定 profile。

### 9.9 OCR rec batch 6→8（GPU 203→199ms）
rec_batch_size_ 6→8（TRT ocr_rec.engine 支持 batch 8），218 行分 37→28 批。
GPU 上减少批次数收益 > pad 浪费；CPU 上持平（pad 浪费抵消）。

### 9.10 模型裁剪 + 最大人脸（资源受限场景）
- insightface analyze 支持跳过子模型（2d/3d landmark、genderage 可关），嵌入式只保留 det+rec
- 新增 `analyze_max_face()`：只识别最大人脸，`face_count` 报告画面人数（多人告警）
- `FaceRecognizerPipeline::predict_max_face()` 同理（人脸识别业务场景）

### 9.11 SOPHGO 全模型转换 + benchmark（已在服务器实测）
- `tools/docker/sophgo/convert_all.sh`：全模型（yolo11n 全家/face/lpr/ocr/insightface）
  ONNX→bmodel，支持 F16/INT8（INT8 需校准图）
- `tools/docker/sophgo/convert_insightface.sh`：insightface 5 子模型（已更新命名 `_f16/_int8`）
- benchmark 新增 `[sophgo]` 用例：遍历 yolo11n 全家 bmodel（fp16/int8）+ insightface pipeline
- **实测（sophon 服务器 BM1688/SE9，aarch64）**：完整 pre/infer/post 分解

| 模型 | 输入 | pre | infer | post | total |
|---|---|---|---|---|---|
| det yolo11n | 640 | 3.8 | 19.7 | 15.5 | **39.0** |
| det yolo11n | 1280 int8 | 8.4 | 16.9 | 1.8 | **27.1** |
| cls | 224 f16/int8 | 3.7/3.4 | 0 | 0 | **3.7/3.4** |
| obb | 640 | 6.8 | 17.1 | **85.3** | **109.2** |
| obb | 1024 f16/int8 | 8.1/8.2 | 44.1/12.4 | 176/179 | 228/199 |
| pose | 640 | 4.6 | 19.4 | 37.5 | **61.5** |
| seg | 640 | 4.7 | 29.0 | 138.7 | **172.4** |

> **关键结论（含候选数诊断）**：
> 1. **post 慢 ≠ NMS 慢**：det 仅 3 个框 post 却 15ms——慢在 8400 anchor 过阈遍历等
>    固定 CPU 开销；det1280 int8 输出单类 [1,5,33600] post 仅 1.9ms 佐证
> 2. **obb 129 框**（test_obb.jpg 密集航拍）：旋转框 NMS O(n²) 是 85ms 主因，属正常
> 3. **seg 13 框 post 却 136ms**：慢在 mask 解码/阈值化（CPU），与框数关系不大
> 4. **生产建议**：正常图（十来个框）+ 640 输入，det/pose post 会 <2ms；
>    seg 的 mask 处理是主要优化点（向量化/缩小 mask 分辨率）
> 5. NMS bmodel 已删除（sophgo 不导 NMS，无法量化；NMS 在 CPU 做）
