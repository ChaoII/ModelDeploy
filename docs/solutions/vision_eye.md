# VisionEye — 视平线透视可视化（针孔 / 鹰眼）

> 架构层：`modeldeploy::vision::solution`
> 源码：`csrc/vision/solutions/vision_eye.h` / `.cpp`

## 职责

基于"视平线"把目标的质心映射到视平线上的对应点，收集这些"眼睛点"，以支撑针孔 / 鹰眼视角的鸟瞰或俯视可视化。

## 应用场景

- 针孔相机 / 鹰眼（bird's-eye）俯视可视化
- 把多个目标质心归一到视平线，叠加透视网格或估计位置
- 观测轨迹基座可视化

## 输入输出

- 输入：目标二维质心 `Point2f(x, y)`（来自检测框中心或关键点）
- 输出：视平线上的点集 `std::vector<Point2f>`

## 关键 API

```cpp
using modeldeploy::vision::solution::VisionEye;

VisionEye eye(float eye_level_y = 0.0f);              // 视平线 y 坐标
static Point2f VisionEye::map_to_eye(Point2f centroid, float eye_level_y);
eye.add(Point2f centroid);                            // 映射后存入 eyes_
const std::vector<Point2f>& eye.eyes() const;         // 已收集的点
eye.reset();                                          // 清空
```

## 原理与算法

```cpp
Point2f VisionEye::map_to_eye(Point2f centroid, float eye_level_y) {
    return Point2f(centroid.x, eye_level_y);   // 保留 x，把 y 改为视平线 y
}
void VisionEye::add(Point2f centroid) {
    eyes_.push_back(map_to_eye(centroid, eye_level_));
}
```

把所有目标质心投影到统一视平线上，得到一串横向分布的"锚点"，可作透视可视化参考点。无复杂后处理。

## 典型用法

```cpp
VisionEye eye(eye_line_y);                          // 视平线 y=eye_line_y
for (auto& box : det.boxes)
    eye.add(Point2f(box.center_x(), box.center_y()));
auto anchors = eye.eyes();                          // 供自绘透视 / 鹰眼可视化
```

## 效果

简单轻量，将检测结果做几何归一化后用于俯视 / 鹰眼可视化。
