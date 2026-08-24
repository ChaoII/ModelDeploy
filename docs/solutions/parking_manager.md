# ParkingManager — 车位占用检测

> 架构层：`modeldeploy::vision::solution`
> 源码：`csrc/vision/solutions/parking_manager.h` / `.cpp`
> 示例：`examples/demo_solutions/demo_solutions.cpp`

## 职责

根据目标质心是否落入各车位多边形，输出**每个车位的占用状态**。配合俯拍相机与车辆检测模型，实现停车场占用实时检测。

## 应用场景

- 停车场车位占用检测与空位引导
- 违停 / 占位判断
- 园区 / 小区车位管理

## 输入输出

- 输入：逐帧 `std::vector<TrackResult>`（源自车辆检测 + 跟踪）
- 输出：`std::vector<bool>`（与车位一一对应的占用布尔）

## 关键 API

```cpp
using modeldeploy::vision::solution::ParkingManager;

ParkingManager parking;
parking.set_slots(const std::vector<std::vector<Point2f>>& slots);  // 车位多边形列表
parking.update(const std::vector<TrackResult>& tracks);             // 每帧喂入
std::vector<bool> occ = parking.occupancy();                        // true=被占用
parking.reset();
```

## 原理与算法

- `set_slots` 把每个车位多边形封装为 `tool::PolygonZone`，`occupied_` 与车位数量一一对应。
- 每次 `update` 先将 `occupied_` 全部置 false，再对每个目标取中心点 `c`，若落在某个车位多边形内则将该车位置 true。
- `PolygonZone::contains` 用经典的**射线法（奇偶规则）**判定点是否在多边形内。

## 典型用法

```cpp
solution::ParkingManager parking;
parking.set_slots({
    {Point2f(0,0), Point2f(40,0), Point2f(40,40), Point2f(0,40)},    // 车位0
    {Point2f(50,0), Point2f(90,0), Point2f(90,40), Point2f(50,40)},  // 车位1
});
for (每帧) {
    auto tracks = tracker.update(dets);
    parking.update(tracks);
}
auto occ = parking.occupancy();
for (size_t i = 0; i < occ.size(); ++i)
    printf("slot[%zu] occupied=%d\n", i, (int)occ[i]);
```

## 效果

即时的车位占用状态输出，空位信息可直接用于诱导屏或 App 展示。
