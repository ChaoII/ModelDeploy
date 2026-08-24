# OBB NMS 优化 + with-NMS(end2end) 验证 — 设计文档

日期: 2026-08-20
状态: 已批准

## 背景 / 问题

Ultralytics OBB 有两种导出形态:

- **without-NMS**（`yolo26n/yolo26n-obb.onnx`, 输入 `[1,3,640,640]`, 输出 `[1,20,8400]`）:
  模型不内嵌 NMS，由 SDK 在 CPU 侧执行旋转框 NMS（`obb_nms.cpp` 的 `utils::obb_nms`）。
- **with-NMS / end2end**（`yolo26n/yolo26n-obb-end2end.onnx`, 输入 `[1,3,1024,1024]`, 输出
  `[1,300,7]`, opset 19 TopK 式内嵌 NMS）:
  模型内已完成 NMS，SDK 侧 `run_with_nms` 只需解析 `(xc,yc,w,h,score,label_id,angle)`。

观测到的瓶颈: `obb_nms` 是 **O(N²) 朴素实现**，内层对每一对候选框都调用
`rotated_rect_to_cv_type`（cv 转换）和 `cv::rotatedRectangleIntersection`（多边形裁剪）。
后处理耗时**强依赖图像内容**（检测框数量决定 NMS 成本）——密集场景（如 `test_obb.jpg`）
post 可到 ~8.7ms，普通场景（如 `test_detection0.jpg`）仅 ~1ms。此"时快时慢"为真实瓶颈。

## 目标

- 在**不改变对外接口、不改变 postprocessor/preprocessor 分发逻辑、不改变 NMS 判定结果**
  的前提下，显著降低 `obb_nms` 在密集场景的耗时。
- 用 ORT-CPU 验证 with-NMS(end2end) 模型走 `run_with_nms` 路径可用（post≈0、框数>0）。

## 非目标

- 不把 NMS 搬到 GPU。
- 不做按类别分组 NMS（会改变行为与既有基线）。
- 不改 with-NMS 模型的后端覆盖（仅 ORT-CPU 验证）。

## 方案 A — 优化 `obb_nms`（`csrc/vision/obb_nms.cpp`）

现实现（`obb_nms.cpp:118`）:
1. 按 score 降序排序得到 `sorted_indices`。
2. 外层 m 遍历保留框；内层 n 遍历其后未抑制框，两两计算旋转 IoU。
3. 内层 `rotated_iou(rotated_rect_to_cv_type(box_i), rotated_rect_to_cv_type(box_j))`
   每对框都做 **两次 cv 转换**，且调用 `cv::rotatedRectangleIntersection`（多边形裁剪）,
   是稀疏场景的主要开销来源。

优化（保持结果逐位不变）:
1. **预计算 cv 转换**：为每个索引一次性构造 `std::vector<cv::RotatedRect>` 与
   `std::vector<RotatedRect>`（SDK 结构）的映射，内层不再重复转换 `box_i`/`box_j`。
2. **AABB 外接矩形预筛**：为每个框预计算轴对齐外接矩形
   `(minx, miny, maxx, maxy)`（可由 `cv::RotatedRect::boundingRect2f()` 得到）。
   内层先做 4 次比较，若两个 AABB 不相交则 `continue`（此时旋转 IoU 必为 ≤0，等价于
   不高于阈值，抑制决策不变），从而跳过昂贵的 `cv::rotatedRectangleIntersection` 调用。
   - AABB 早筛是**等价变换**：旋转矩形相交 ⇒ 其外接矩形必相交；反之不成立，但不相交时
     旋转 IoU=0 ≤ iou_threshold，不会改变抑制结果。故输出与现状完全一致。
3. 保留原有排序、抑制规则、`keep_indices` 重建逻辑不变。

收益: 稀疏/普通场景（大部分框 AABB 不重叠）能跳过绝大多数多边形相交调用；
密集场景同样受益于一次性的 cv 转换预计算。

## 方案 B — with-NMS(end2end) ORT-CPU 验证（`benchmark/benchmark_models.cpp`）

- 新增独立 benchmark 用例（不进入 `bench_yolo` 多后端循环，避免依赖不存在的
  `mnn`/`trt` end2end 产物）:
  - `set_size({1024,1024})`（预处理器默认 640×640，必须覆盖为模型输入尺寸）。
  - ORT-CPU 后端加载 `onnx/yolo26n/yolo26n-obb-end2end.onnx`。
  - 对 `test_obb.jpg` 执行 `predict`，报告 `pre/infer/post` 与检测框数。
- 期望: 输出 `[1,300,7]` → `run` 按 `shape[2]==7` 自动走 `run_with_nms`，
  **post≈0**（NMS 在模型内），框数 > 0。
- 不做断言失败即崩的硬校验，但输出可读结果以便人工核对；若框数为 0 给出提示。

## 方案 C — 验证 / 回归

1. 现有 yolo26n-obb **无 NMS** 用例（含基线）作为回归护栏：优化后框数与坐标必须与
   现状一致（AABB 等价性）。
2. benchmark 对比优化前后 no-NMS obb 在 `test_obb.jpg`（密集）与普通图的 post 耗时。
3. 运行 end2end 用例确认 `post≈0`、框数>0。

## 涉及文件

- `csrc/vision/obb_nms.cpp` — 优化 `utils::obb_nms`（仅此文件改动核心逻辑）。
- `benchmark/benchmark_models.cpp` — 新增 end2end OBB 验证用例。
- 测试产物 / 临时文件按需清理。

## 验收标准

- `obb_nms` 优化后，yolo26n-obb 无 NMS 回归测试/基线结果不变。
- no-NMS obb post 在 `test_obb.jpg` 上明显下降（预期数倍）。
- end2end (with-NMS) 用例 ORT-CPU 跑通，`post≈0`、框数>0。
- 不引入对外 API / 布局变更。
