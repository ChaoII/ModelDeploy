# DistanceEstimator — 目标两两距离估计

> 架构层：`modeldeploy::vision::solution`
> 源码：`csrc/vision/solutions/distance_estimator.h` / `.cpp`

## 职责

计算当前画面内**所有目标两两之间的质心距离**（像素与米），用于社交安全距离 / 防碰撞等告警场景。

## 应用场景

- 社交安全距离 / 聚集告警（人与人间距检测）
- 车辆 / 行人防碰撞预警
- 目标之间的最小间距监控

## 输入输出

- 输入：当前帧 `std::vector<TrackResult>`
- 输出：`std::vector<std::pair<std::pair<int,int>, float>>`（`{ {id_a, id_b}, 距离 }`）

## 关键 API

```cpp
using modeldeploy::vision::solution::DistanceEstimator;

DistanceEstimator dist;
dist.set_meter_per_pixel(float m);                                  // 像素→米比例（默认 0.01）
auto pairs_m  = dist.pair_distances_m(tracks);                      // 米
auto pairs_px = dist.pair_distances_px(tracks);                     // 像素
dist.reset();                                                       // 空实现，可安全调用
```

返回项说明：`pair<int,int>` 是两个参与距离计算的 `track_id`，`float` 为二者中心距。

## 原理与算法

- 先把所有目标映射到质心 `pts`（`track_id -> Point2f`）。
- 双重循环对每个 `i < j` 的组合计算欧氏距离 `sqrt(dx²+dy²)`，得到全部无序两两组合。
- `pair_distances_m` 复用像素距离后把每个距离乘 `mpp_`。
- 纯计算、无内部状态。

> 两两组合复杂度为 O(n²)，画面目标数量多时注意性能。

## 典型用法

```cpp
solution::DistanceEstimator dist;
dist.set_meter_per_pixel(0.05f);
// 任意帧需要时直接调用，无需逐帧 update
auto d_m = dist.pair_distances_m(tracks);
for (auto& e : d_m)
    printf("dist(id=%d,id=%d)=%.2f m\n", e.first.first, e.first.second, e.second);
// 配置告警：遍历找出距离低于阈值的目标对
```

## 效果

零依赖的实时间距计算，是聚集 / 碰撞告警的轻量基础，物理精度取决于 `mpp_` 标定。
