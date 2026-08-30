# RegionCounter — 多区域逐帧计数

> 架构层：`modeldeploy::vision::solution`
> 源码：`csrc/vision/solutions/region_counter.h` / `.cpp`
> 示例：`examples/demo_solutions/demo_solutions.cpp`

## 职责

对**多个命名区域**进行**逐帧区域内目标计数**：每一帧统计落在各多边形区域内的跟踪目标数量（按区域名返回）。适用于多货架 / 多通道 / 多展位的实时在区人数或车数统计。

## 应用场景

- 多区域（货架、展位、闸口通道）的实时在区人数统计
- 多路口 / 多车道的逐帧车流量监测
- 按类别分别统计区域内目标（如只数人、只数车）

## 输入输出

- 输入：逐帧跟踪结果 `std::vector<modeldeploy::vision::tracking::TrackResult>`（由上游**检测 + 跟踪**链路产生，本类不直接依赖模型）
- 输出：`std::map<std::string, int>`（区域名 -> 当前帧区域内目标数）

## 关键 API

```cpp
using modeldeploy::vision::solution::RegionCounter;
using modeldeploy::vision::tracking::TrackResult;

RegionCounter rc;
rc.add_region(const std::string& name, const std::vector<Point2f>& polygon);  // 添加命名区域（<3 点忽略）
rc.set_classes(const std::vector<int32_t>& cls);     // 类别白名单（空=全部）
rc.update(const std::vector<TrackResult>& tracks);   // 每帧喂入跟踪结果
std::map<std::string,int> counts = rc.region_counts(); // 区域名 -> 当前帧计数
size_t n = rc.total_regions();                       // 已注册区域数
rc.reset();
```

## 原理与算法

- **目标点**：取每个目标包围盒中心 `c = (box.x + box.w*0.5, box.y + box.h*0.5)`。
- **逐帧归零**：每次 `update` 先把所有区域计数清零，因此 `region_counts()` 始终是**当前帧**的在区数量，**非累计**。
- **区域判定**：对每个目标，遍历所有区域，若 `tool::PolygonZone::contains(c)` 为真则对应区域计数 `++`（一个目标落入多个区域时各区域分别累加）。
- **类别过滤**：`classes_` 为空时统计所有目标；否则只统计命中白名单的 `label_id`。
- `add_region` 传入的多边形少于 3 个点时直接忽略。

## 典型用法

```cpp
using namespace modeldeploy::vision;
solution::RegionCounter rc;
rc.add_region("doorA", {Point2f(0,0), Point2f(80,0), Point2f(80,240), Point2f(0,240)});
rc.add_region("doorB", {Point2f(200,0), Point2f(280,0), Point2f(280,240), Point2f(200,240)});
// 可选：rc.set_classes({0});

for (每帧) {
    std::vector<tracking::TrackResult> tracks = tracker.update(dets);
    rc.update(tracks);
}
for (const auto& kv : rc.region_counts())
    printf("%s = %d\n", kv.first.c_str(), kv.second);
```

## 效果

纯后处理即可输出各命名区域的实时在区目标数，多区域共用一次遍历，开销低，适合多点位并行统计。
