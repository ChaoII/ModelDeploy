# Metrics — mAP / 检测评估指标

> 架构层：`modeldeploy::vision::tool`
> 源码：`csrc/vision/tools/metrics.h` / `.cpp`

## 职责

给定预测框与真值框，统计 **TP / FP / FN**，并计算 **precision / recall / f1** 与 **11 点插值 mAP@0.5**。

## 应用场景

- 离线验证检测器精度（demo 小样本演示）
- 回归测试指标断言
- 模型选型对比

## 关键 API

```cpp
using modeldeploy::vision::tool;

struct MetricsCounts { int tp{0}; int fp{0}; int fn{0}; };
struct MetricsScores { double precision{0}; double recall{0}; double f1{0}; double map50{0}; };

MetricsCounts count_tp_fp_fn(const std::vector<Rect2f>& preds,
                             const std::vector<float>& pred_scores,
                             const std::vector<Rect2f>& gt,
                             double iou_threshold = 0.5);
MetricsScores evaluate_metrics(const std::vector<Rect2f>& preds,
                               const std::vector<float>& pred_scores,
                               const std::vector<Rect2f>& gt,
                               double iou_threshold = 0.5);
```

## 算法原理

- **TP/FP/FN**：按分数降序遍历每个预测框，与所有未匹配且 IoU ≥ 阈值的真值中 IoU 最大者匹配 → TP 并标记该真值；无匹配 → FP；`FN = |gt| - TP`。
- **precision = TP/(TP+FP)**，**recall = TP/(TP+FN)**，**f1 = 2·P·R/(P+R)**。
- **map50**：按分数降序扫描累积 (recall, precision) 曲线；做 **11 点插值**：对 recall 在 `0.0,0.1,…,1.0` 共 11 个目标值，取所有 recall≥目标 的点中最大 precision，平均 11 值得 AP（此处即 mAP@0.5）。`gt` 为空时 AP 记 0。

> 注意：只做框级匹配，**不按类别分组**（多类别需调用方自行逐类评估）。

## 典型用法

```cpp
std::vector<Rect2f> preds = {Rect2f(0,0,10,10), Rect2f(5,5,10,10)};
std::vector<float> scores = {0.9f, 0.8f};
std::vector<Rect2f> gt     = {Rect2f(0,0,10,10)};
auto m = tool::evaluate_metrics(preds, scores, gt, 0.5);
printf("map50=%.3f p=%.3f r=%.3f f1=%.3f\n", m.map50, m.precision, m.recall, m.f1);
```
