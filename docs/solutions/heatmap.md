# Heatmap — 目标位置热力图

> 架构层：`modeldeploy::vision::solution`
> 源码：`csrc/vision/solutions/heatmap.h` / `.cpp`
> 示例：`examples/demo_solutions/demo_solutions.cpp`

## 职责

把目标中心点累加绘制到一张低分辨率栅格热力图上，反映目标在画面中的**空间聚集 / 活动密度分布**。

## 应用场景

- 商场/场馆人流热区分析
- 安检、展会人群聚集检测
- 园区车辆活动热点统计
- 寻找目标最常出现的位置、密度告警

## 输入输出

- 输入：逐帧 `std::vector<TrackResult>` + 原图/帧宽高（用于坐标缩放）
- 输出：热力值栅格（按行主序 `[y*w+x]`）、峰值坐标

## 关键 API

```cpp
using modeldeploy::vision::solution::Heatmap;

Heatmap heat;
heat.set_size(int w, int h);                       // 设定栅格尺寸（会清零热力值）
heat.update(const std::vector<TrackResult>& tracks, int frame_w, int frame_h);
const std::vector<float>& heat() const;            // 热力值数组（长度 w*h）
std::pair<int,int> peak() const;                   // 峰值栅格坐标 (x,y)
float heat_at(int x, int y) const;                 // 取某栅格热力值
heat.reset();                                      // 清零
```

## 原理与算法

无状态累积型：

- 缩放系数 `sx = w / max(1, frame_w)`、`sy = h / max(1, frame_h)`，把原图像素坐标按比例映射到栅格。
- 对每个目标取中心点 `(cx, cy)`，映射到 `x = clamp(int(cx*sx), 0, w-1)`、`y = clamp(int(cy*sy), 0, h-1)`，然后 `heat_[y*w+x] += 1.0f`。
- 持续 `update` 即累积出密度热区。
- `peak()` 用 `max_element` 找最大热力索引，反解 `(x = idx%w, y = idx/w)`。

## 典型用法

```cpp
solution::Heatmap heat;
heat.set_size(320, 240);                     // 低分辨率栅格
for (每帧) {
    auto tracks = tracker.update(dets);
    heat.update(tracks, frame_w, frame_h);   // frame_w/h 为原图尺寸
}
auto peak = heat.peak();                     // (x, y)
float v = heat.heat_at(x, y);
// 亦可直接遍历 heat.heat() 自行绘制热力图
```

## 效果

以极低代价把长时间目标位置累积成密度图，便于可视化与密度告警。
