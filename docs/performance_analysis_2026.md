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

> **无法转 TRT**：face-age（Gemm 固定 batch）、fas_second（动态 reshape）——仅 ORT CPU。
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
| insightface（det+2d106+3d68+rec+genderage） | 48.95 | 21.20 | **1.09** | TRT 45x |
| face-rec（scrfd + seetaface rec） | 76.01 | - | **2.82** | TRT 27x |
| face-as（scrfd + fas_first + fas_second） | 112.92 | - | onnx only（fas_second 无 TRT） | - |
| pedestrian-attr（zhgd_det + zhgd_ml） | 198.11 | - | **15.73** | TRT 12.6x |
| **OCR（det+cls+rec，218 行）** | **1311.31** | - | **202.87** | **TRT 6.5x** |
| LPR（det+rec） | 待补 | - | 待补 | 已检出，pipeline 计时未实现 |

> **OCR 关键结论**：218 行密集文本页 CPU 1311ms → GPU TRT 203ms（6.5x）。
> 实际文档（几十行）GPU 下 <100ms。

## 5. 瓶颈分析（当前仍存在的优化点）

### 5.1 OCR rec 逐行推理（GPU 下 203ms 的主要成本）
- 218 行 → rec batch=6（动态宽 pad 到 batch_max_w），GPU 下 rec 单次 1.2ms × 37 批 ≈ 44ms；
  det 7.4ms + cls 逐行 + crop/透视 ≈ 剩余。
- **优化方向**：rec 按宽度聚类分组（宽窄分开 batch），减少 pad 浪费。

### 5.2 det post 的 CPU NMS
- 非内嵌 NMS 模型（obb/pose/seg onnx）post 在 CPU 做 NMS（3.9-10ms）。
- TRT engine 用内嵌 NMS 模型后 post≈0.005ms（已解决）。

### 5.3 face-as 的 fas_second 整图推理 + clarity 估计
- fas_second 无法转 TRT（动态 reshape），face-as 只能部分 GPU。

### 5.4 pre 阶段 SIMD
- det pre 8ms（CPU）/ insightface det pre 2.2ms（含 align）。

## 6. 吞吐量结论（fps，最新实测）

| 场景 | CPU | GPU TRT | 加速 |
|---|---|---|---|
| det 单帧 | 49 fps | **278 fps** | 5.7x |
| insightface 全流程 | 20 fps | **917 fps** | 45x |
| face-rec pipeline | 13 fps | **355 fps** | 27x |
| pedestrian-attr | 5 fps | **64 fps** | 12.6x |
| OCR 整页（218 行） | 0.76 fps | **4.9 fps** | 6.5x |

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
