# 跨后端回归基线测试系统 — 设计文档

日期：2026-08-07
分支：main

## 背景与动机

现有回归基线测试系统（`docs/superpowers/specs/2026-08-07-regression-baseline-design.md`）已覆盖 11 个 ORT 模型。但项目支持多种推理后端（ORT/MNN/TRT/SOPHGO），各后端用不同模型格式（onnx/mnn/engine/bmodel）。

需求：
1. **全覆盖迁移**：所有视觉模型（yolo 系、plate、zhgd、face、ocr 结构等）的 onnx 移入 `onnx/` 分目录，音频模型保持原位
2. 按**后端分目录**组织模型文件和基线
3. **yolo 系优先**转换多后端格式（mnn/engine/bmodel），验证跨后端精度一致性——特别是 **int8 量化模型**的精度损失对比（用户明确视为价值）
4. 后续新加后端可无缝扩展测试

## 目录结构

### 模型文件
```
test_data/test_models/
├── onnx/   yolo11n.onnx, yolo11n_nms.onnx, yolo11n-cls.onnx, yolo11n-obb.onnx, yolo11n-obb_nms.onnx,
│           yolo11n-pose.onnx, yolo11n-pose_nms.onnx, yolo11n-seg.onnx, yolo11n-seg_nms.onnx,
│           yolo11n_without_nms.onnx, yolov5plate.onnx, plate_recognition_color.onnx,
│           zhgd_det.onnx, zhgd_det_20251219.onnx, zhgd_ml.onnx, zhgd_ml_20251219.onnx,
│           zc.onnx, best.onnx, line_edit.onnx, model.onnx
│           face/（scrfd、age、gender、recognition、fas 等全部移入）
│           ocr/（SLANet、SLANeXt 结构分析等全部移入）
├── mnn/    yolo11n.mnn, yolo11n_nms.mnn, ...（yolo 系转换）
├── trt/    yolo11n.engine, yolo11n_nms.engine, ...（yolo 系转换）
└── sophgo/ yolo11n.bmodel, ...（yolo 系转换）
```
- **全覆盖迁移**：所有根目录 onnx + face/ + ocr/ 结构模型移入 `onnx/`
- **音频保持原位**：sense_voice/、kokoro_v1_1/ 不迁移（音频不走 vision 后端路径）
- **杂项归类**：`model.engine`、`model.mdenc`、加密模型（yolo11n_nms_encrypted.mdenc 等）归入对应后端目录或 `misc/`
- 原 `face/`、`ocr/` 子目录在 onnx/ 下**保留子目录结构**（onnx/face/、onnx/ocr/），避免与根目录 onnx 冲突

### 基线文件
```
tests/baselines/
├── ort/    现有 15 个基线迁入
├── mnn/    yolo11n.mnn.det.json, ...
├── trt/    yolo11n.engine.det.json, ...
└── sophgo/ yolo11n.bmodel.det.json, ...
```
- 基线文件名沿用 `<模型文件名>.<type>.json` 约定
- 迁移：`tests/baselines/*.json` → `tests/baselines/ort/`

## 模型清单

### 全覆盖迁移（onnx/）
所有视觉模型 onnx 移入 `onnx/`，包括：yolo11n 全家、yolov5plate、plate_recognition_color、zhgd_det/zhgd_ml（含 20251219 版）、zc、best、line_edit、model、face/ 全部、ocr/ 结构模型全部。

### 多后端转换（yolo 系优先）
| 模型 | onnx | mnn | trt | bmodel |
|------|------|-----|-----|--------|
| yolo11n | ✓ | 转换 | 已有(移入) | 转换 |
| yolo11n_nms | ✓ | 已有(移入) | 已有(移入) | 转换 |
| yolo11n-seg | ✓ | 转换 | 转换 | 转换 |
| yolo11n-seg_nms | ✓ | 转换 | 已有(移入) | 转换 |
| yolo11n-obb | ✓ | 转换 | 转换 | 转换 |
| yolo11n-obb_nms | ✓ | 转换 | 转换 | 转换 |
| yolo11n-pose | ✓ | 转换 | 转换 | 转换 |
| yolo11n-pose_nms | ✓ | 转换 | 转换 | 转换 |
| yolo11n-cls | ✓ | 转换 | 转换 | 转换 |

yolo 系优先全后端转换（含多分类 yolo11n-cls）；其余模型（plate/zhgd/结构等）只保留 onnx，按需后续转换。

"应转尽转"：yolo 系尽量齐全 4 后端。转换工具：
- TRT：`trtexec --onnx=... --saveEngine=...`（本机 TensorRT 10.9）
- MNN：MNNConvert（Linux 或 pip 安装）
- bmodel：Sophgo 服务器 docker + tpu-mlir（`tools/docker/sophgo/convert.sh`）

## 组件改动

### baseline_collect.exe（收集器）
- 新增 `--backend <ort|mnn|trt|sophgo>` 参数
- 模型路径路由：`test_data/test_models/<backend>/<模型名>.<ext>`
- 基线路径路由：`tests/baselines/<backend>/`
- 模型扩展名 → 后端映射复用 `RuntimeOption::set_model_path` 的自动路由（`backend_for_format`：onnx→ORT, mnn→MNN, engine→TRT, bmodel→SOPHGO）

### baseline_compare.cpp（对比器）
- 每个"模型×后端"一个 TEST_CASE，标签 `[regression]` + `[backend:<name>]`
- **自对比**：当前后端输出 vs 同后端基线（严格阈值）
- **基准对比**：当前后端输出 vs ORT 基线（严格阈值）——跨后端精度验证，直接暴露 int8/FP16 量化的精度差异
- 模型/基线缺失 → `return` skip（不误报）

### baseline_utils
- 复用现有 `compare_*` 函数（阈值不变：坐标±1px/angle±0.5°/score±0.01/label严格/tensor±1e-4）
- 无需新增宽阈值变体——用户选择**严格阈值**跨后端对比（验证量化精度损失）

## 测试策略

- ORT 全模型 → 标签 `[regression]`（覆盖所有已迁移 onnx 模型）
- 各后端 → 标签 `[regression]` + `[backend:ort/mnn/trt/sophgo]`，可按后端单独运行
- 例如跑 MNN 全部：`test_modeldeploy "[backend:mnn]"`
- 多分类模型（yolo11n-cls）→ `compare_cls` 严格对比 topN label/scores

## 对比标准（严格阈值）

| 类型 | 阈值 |
|------|------|
| 坐标（det/obb/pose/ocr） | ±1px |
| Obb angle | ±0.5° |
| score | ±0.01 |
| label_id | 严格相等 |
| Seg mask | nonzero_ratio 差异 < 0.001 |
| OCR text | 严格相等 |
| Tensor 数值 | ±1e-4 |
| Tensor shape | 严格相等 |

自对比与基准对比使用**相同**严格阈值。

## 迁移步骤

1. 建目录 `test_data/test_models/{onnx,mnn,trt,sophgo}/`、`tests/baselines/{ort,mnn,trt,sophgo}/`
2. **全覆盖迁移**：所有根目录 onnx + face/ + ocr/ 结构模型移入 `onnx/`（保留子目录结构）；现有 mnn/engine 移入对应目录；杂项（加密/engine/mdenc）归类
3. 现有 15 个基线移入 `tests/baselines/ort/`
4. **yolo 系优先转换**：缺失的 mnn/engine/bmodel 模型（trtexec/MNNConvert/tpu-mlir）
5. 各后端跑 collect 生成基线
6. 对比器按后端加载基线，验证自洽 + 跨后端

## 错误处理

- 缺失模型/基线/图片 → skip（与现有一致）
- 模型加载失败（文件存在但损坏）→ FAIL（不静默跳过）
- 转换工具缺失 → 该后端基线不生成，测试 skip

## 测试

- 迁移后 ORT 基线可被对比器无差异通过（自洽）
- 各后端自对比通过（同后端无回归）
- 跨后端对比：int8/FP16 模型会暴露精度差异（预期行为，用户确认要这个）
- 人为篡改某后端基线值 → 对应测试 FAIL
