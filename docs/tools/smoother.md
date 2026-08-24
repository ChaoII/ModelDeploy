# DetectionSmoother — 检测结果平滑（EMA）

> 架构层：`modeldeploy::vision::tool`
> 源码：`csrc/vision/tools/smoother.h` / `.cpp`

## 职责

对逐帧检测框坐标做**指数移动平均（EMA）平滑**，抑制视频 / 相机抖动导致的框抖动。

## 应用场景

- 实时视频检测框抖动抑制
- 追踪框后处理
- 需要稳定框坐标的统计 / 展示（依赖 `tracker_id` 帧间一致）

## 关键 API

```cpp
using modeldeploy::vision::tool::DetectionSmoother;

DetectionSmoother sm(double alpha = 0.5);  // 新观测权重，越大越跟随最新，越小越平滑
sm.reset();
Detections out = sm.update(const Detections& in);  // 返回平滑后的检测
```

## 算法原理

- 状态：`std::vector<Rect2f> state_` 保存上一帧平滑框。
- 核心为**指数移动平均**，仅对左上角 **x/y** 平滑（宽高不变）：`new_xy = alpha*in_xy + (1-alpha)*state_xy`。
- 两条路径：
  1. **带 tracker_id**（`tracker_id.size()==boxes.size()`，要求索引对齐做帧间匹配）：状态大小变化则用当前帧重置；否则逐索引 EMA，随后 `state_ = out.boxes`。
  2. **不带 tracker_id**（无法可靠匹配）：取 `min(状态数, 当前框数)` 逐索引 EMA，并裁剪状态到该数量（快速启停可能失配，属简化方案）。
- 首次调用（state 空且带 id）以当前帧初始化，故第一帧输出等于输入。

## 典型用法

```cpp
tool::DetectionSmoother sm(0.4);
for (auto& frame : video) {
    tool::Detections d = detector(frame);    // 需保证 tracker_id 填充且索引稳定
    tool::Detections s = sm.update(d);       // 平滑后的框
}
```
