# 工具（Tools）

工具层（`modeldeploy::vision::tool`）提供**可视化、检测容器、评估指标、切片、平滑、区域判断**等通用后处理能力，是模型推理与解决方案之间的公共底座。

- 源码目录：`csrc/vision/tools/`

## 工具清单

| 工具 | 职责 | 文档 |
|------|------|------|
| Annotator | 画面绘制矩形 / 文字 / 线段 / 圆形 / 半透明多边形 | [annotator.md](tools/annotator.md) |
| Detections | 平行向量式检测结果容器 + IoU / NMS / 类别过滤 / 跟踪转换 | [detections.md](tools/detections.md) |
| Metrics | TP/FP/FN、precision / recall / f1、11 点插值 mAP@0.5 | [metrics.md](tools/metrics.md) |
| InferenceSlicer | 大图切片推理 + 坐标重映射回原图 | [slicer.md](tools/slicer.md) |
| DetectionSmoother | 检测框 EMA 平滑（抑制抖动） | [smoother.md](tools/smoother.md) |
| Zone | LineZone 跨线计数 / PolygonZone 区域判断 / filter_by_zone | [zone.md](tools/zone.md) |

> 组合这些工具可快速搭建业务方案，与解决方案层无缝衔接，见 [solutions.md](solutions.md)。
