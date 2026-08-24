# SpeedEstimator — 目标运动速度估计

> 架构层：`modeldeploy::vision::solution`
> 源码：`csrc/vision/solutions/speed_estimator.h` / `.cpp`
> 示例：`examples/demo_solutions/demo_solutions.cpp`

## 职责

利用相邻两帧目标中心点的位移与时间戳差，估计每个目标的运动速度（**像素/秒**与**米/秒**）。

## 应用场景

- 交通车辆测速（配合俯拍标定得到像素→米比例）
- 行人移动速度监测
- 目标移动快慢分析 / 异常移动预警

## 输入输出

- 输入：逐帧 `std::vector<TrackResult>` + 当前时间戳（毫秒，需逐帧单调递增）
- 输出：`std::map<int,float>`（track_id → 速度）

## 关键 API

```cpp
using modeldeploy::vision::solution::SpeedEstimator;

SpeedEstimator speed;
speed.set_meter_per_pixel(float m);                 // 像素→米比例（默认 0.01）
speed.update(const std::vector<TrackResult>& tracks, double timestamp_ms);
std::map<int,float> px = speed.speeds_px_per_s();   // 像素/秒
std::map<int,float> m  = speed.speeds_m_s();        // 米/秒（= 像素/秒 × mpp_）
speed.reset();
```

## 原理与算法

- 对每个目标取中心点 `c`，在上一帧记录 `track_id -> {中心点, 时间戳}` 中查找。
- 位移 `dist = sqrt(dx²+dy²)`（像素）；时间差 `dt = timestamp_ms - last_ts`（毫秒）。
- 若 `dt > 0`：`px_per_s_[track_id] = dist / dt * 1000`。
- `speeds_m_s()` 在像素速度基础上乘 `mpp_` 得到物理速度。
- 每帧把 `{c, timestamp_ms}` 写入上一帧缓存。

> 注意：必须由调用方提供单调递增的时间戳；`mpp_` 需根据相机标定给出才能得到真实米制速度。

## 典型用法

```cpp
solution::SpeedEstimator speed;
speed.set_meter_per_pixel(0.05f);         // 每个像素对应 0.05 米（需标定）
double ts = 0.0;
for (每帧) {
    auto tracks = tracker.update(dets);
    speed.update(tracks, ts);            // ts 毫秒时间戳，逐帧 +33 等
    ts += 33.0;
}
auto kmh = speed.speeds_m_s();           // 米/秒
```

## 效果

在跟踪链路上零外部依赖地得到速度，是测速与移动分析的轻量方案（物理精度取决于标定）。
