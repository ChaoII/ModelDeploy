# 回归基线测试系统 — 设计文档

日期：2026-08-07
分支：refactor/tensor-merge-view

## 背景与动机

2026-08-07 在修复 OBB 后处理 bug 时，将 `run_with_nms` 的角度索引从 `attr_ptr[dim2-1]`（索引 6）错误改为 `attr_ptr[4]`（score），导致所有 OBB 检测框角度绘制错误。该错误由用户通过绘制可视化发现。现有测试（`test_vision_models.cpp`）只做粗粒度断言（非空、数量 > 0、值 > 0），无法发现此类"数值取错位置但类型正确"的回归。

目标：为每个模型建立**输出基线**，后续任何修改（后处理 / 预处理 / 推理）后运行测试与基线精确对比，防止类似改错。

## 范围

基线覆盖三类输出：
1. **预处理输出张量**（`preprocessor.run()` 之后的输入 Tensor）
2. **模型原始推理输出**（`infer()` 之后的输出 Tensor，不经过后处理）
3. **后处理结果**（`predict()` 后的结构化结果：Detection/Obb/Seg/Pose/Cls/OCR det/rec/cls）

明确**不包含**：图像可视化结果对比（像素级）。

## 存储格式

文本 JSON 文件，存放于 `tests/baselines/`，由 git 管理。文件命名：

```
<model文件名>.det.json     # Detection 结果
<model文件名>.obb.json     # OBB 结果
<model文件名>.seg.json     # Seg 结果
<model文件名>.pose.json    # Pose 结果
<model文件名>.cls.json     # Cls 结果
<model文件名>.pre.json     # 预处理输出张量
<model文件名>.raw.json     # 原始推理输出
```

JSON 顶层结构：

```json
{
  "meta": { "model": "yolo11n-obb_nms.onnx", "image": "test_obb1.jpg",
            "date": "2026-08-07", "backend": "ort-cpu" },
  "results": [ ... 每实例结构化字段 ... ]
}
```

## 三类基线内容定义

### 预处理张量（*.pre.json）
```json
{
  "tensor": {
    "name": "images",
    "shape": [1, 3, 640, 640],
    "dtype": "FP32",
    "values": [0.123456, -0.987654, ...]   // 归一化后，6 位小数截断
  }
}
```
若张量过大（元素数 > 10000），只存前 10000 元素 + `stats: {min, max, mean}`。

### 原始推理输出（*.raw.json）
同理，但额外存每个输出张量的统计值（min/max/mean）和前 N 元素。原始输出可能很大（如 seg 的 mask 或 8400 候选框），完整存储会臃肿，故以"统计值 + 抽样"覆盖。

### 后处理结果
按模型类型存结构化字段：

| 模型 | 字段 |
|------|------|
| Detection | `box: {x1,y1,x2,y2}`, `score`, `label_id` |
| Obb | `rotated_box: {xc,yc,width,height,angle}`, `score`, `label_id` |
| Seg | `box`, `score`, `label_id`, `mask: {w,h,data}`（数据为 float32） |
| Pose | `box`, `keypoints: [[x,y,score]...]`, `scores` |
| Cls | `label_ids`, `scores` |
| OCR det | `boxes: [[x1,y1,...,x4,y4]...]`（4 角点 8 值） |
| OCR rec | `text`（字符串）, `rec_score` |
| OCR cls | `cls_label`, `cls_score` |

## 对比阈值（浮动）

| 类型 | 阈值 |
|------|------|
| Detection box 坐标 | ±1px |
| Obb xc/yc/w/h | ±1px |
| Obb angle | ±0.5° |
| score | ±0.01 |
| label_id | 必须严格相等 |
| Pose keypoints | ±1px |
| Seg mask | 像素值差异 < 0.1%（计算不相等的像素比例） |
| Cls | label 严格相等，top1 score ±0.01 |
| OCR det 角点 | ±1px |
| OCR rec text | 字符串严格相等 |
| OCR rec/cls score | ±0.01 |
| Tensor / 预处理数值 | ±1e-5 |
| Tensor shape | 必须严格相等 |

失败时打印差异明细：哪个实例、哪个字段、基线值 vs 当前值。

## 组件

### baseline_utils.h / baseline_utils.cpp
- `serialize_tensor(const Tensor&) -> json` / `deserialize_tensor(...)`
- `serialize_xxx_result(...) -> json` / `deserialize_xxx_result(...)`（Detection/Obb/Seg/Pose/Cls/OCR det/OCR rec/OCR cls）
- `compare_results(基线, 当前) -> vector<string>`（返回差异描述，空则通过）
- `compare_tensors(基线, 当前) -> vector<string>`
- `load_baseline(path) / save_baseline(path, json)`

### baseline_collect.cpp（独立可执行）
用法：
```
baseline_collect.exe --model <path> --image <path> --out <dir>
```
- 加载模型（CPU ORT）
- 跑 preprocess → 存 *.pre.json
- 跑 infer → 存 *.raw.json
- 跑 predict → 存 *.det/obb/seg/pose/cls/ocr-det/ocr-rec/ocr-cls.json
- 输出提示已生成的基线文件

### baseline_compare.cpp（Catch2 测试，标签 `[regression]`）
每个模型一个 TEST_CASE：
1. 检查模型文件和基线文件都存在，缺失即 `return`（skip）
2. 重新加载模型跑图，得到三类输出
3. 与基线 JSON 对比
4. 有差异则打印明细并 FAIL

## 触发方式

集成到 ctest 自动运行。`[regression]` 标签的测试默认随 `test_modeldeploy` 一起跑。模型或基线缺失时跳过（与现有测试行为一致，CI 无模型也能过）。

## 基线更新策略

手动：
1. 用户确认当前输出正确（如模型替换、后处理优化）
2. 运行 `baseline_collect.exe` 重新生成基线文件
3. git 提交基线文件

## 覆盖模型清单

| 模型文件 | 测试图片 | 结果类型 |
|----------|----------|----------|
| yolo11n.onnx | test_detection0.jpg | det + pre + raw |
| yolo11n_nms.onnx | test_detection0.jpg | det |
| yolo11n-seg.onnx | test_person.jpg | seg |
| yolo11n-pose.onnx | test_person.jpg | pose |
| yolo11n-obb.onnx | test_obb1.jpg | obb |
| yolo11n-obb_nms.onnx | test_obb1.jpg | obb |
| yolo11n-cls.onnx | test_person.jpg | cls |
| face/scrfd_*.onnx | test_face_detection.jpg | face det |
| ocr/ppocrv4_mobile/det_infer.onnx | test_ocr.png | ocr det + pre + raw |
| ocr/ppocrv4_mobile/rec_infer.onnx | test_ocr.png | ocr rec |
| ocr/ppocrv4_mobile/cls_infer.onnx | test_ocr.png | ocr cls |

OCR 模型选择 ppocrv4_mobile 作为默认基线模型（v5/v6 可后续补充）。
OCR 的 pre/raw 基线由 det_infer 模型产生。

## 错误处理

- 基线文件缺失 → FAIL，提示"先运行 baseline_collect.exe 生成"
- 模型 / 图片缺失 → skip（return，不误报）
- 原始张量过大 → 截断 + 统计值，避免基线文件过大
- JSON 解析失败 → FAIL 并提示文件损坏

## 测试

- 收集器生成的基线应能被对比器无差异通过（自洽性验证）
- 人为改动基线中一个 angle 值，对比器应报出该差异（验证对比器灵敏度）
- 人为恢复，测试全绿
