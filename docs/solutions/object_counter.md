# ObjectCounter — 目标跨线/区域计数

> 架构层：`modeldeploy::vision::solution`
> 源码：`csrc/vision/solutions/object_counter.h` / `.cpp`
> 示例：`examples/demo_solutions/demo_solutions.cpp`

## 职责

对进入画面的目标进行**跨线进出计数**与**指定区域闯入计数**，并可统计各类别目标出现次数。适用于客流量、车流量、重点区域闯入等业务统计。

## 应用场景

- 商超/地铁出入口的客流量跨线进出统计
- 交通卡口、厂区的车流量统计
- 重点区域（警戒区）人员闯入计数
- 按类别统计对象出现次数（如"今天进了多少辆车 vs 多少人"）

## 输入输出

- 输入：逐帧跟踪结果 `std::vector<modeldeploy::vision::tracking::TrackResult>`（由上游**检测 + 跟踪**链路产生，本类不直接依赖模型）
- 输出：`CounterStats`（跨线进出数 + 区域进入数 + 各类别计数）

## 关键 API

```cpp
using modeldeploy::vision::solution::ObjectCounter;
using modeldeploy::vision::tracking::TrackResult;

ObjectCounter counter;
counter.set_line(Point2f a, Point2f b);                       // 设置计数线（调用后重置计数）
counter.set_region(const std::vector<Point2f>& pts);          // 设置多边形区域（进入区域计数）
counter.set_classes(const std::vector<int32_t>& cls);         // 类别白名单（空=全部）
counter.update(const std::vector<TrackResult>& tracks);       // 每帧喂入跟踪结果
CounterStats st = counter.stats();                            // line_in / line_out / class_count
int n = counter.region_count();                               // 区域进入数
counter.reset();
```

辅助结构体：

```cpp
struct CounterStats {
    int line_in{0}; int line_out{0};
    std::map<int,int> class_count;   // label_id -> 出现次数
};
```

## 原理与算法

- **目标点**：取每个目标包围盒中心 `c = (box.x + box.w*0.5, box.y + box.h*0.5)`。
- **跨线计数**：为每个 `track_id` 维护一个 `tool::LineZone`。用叉积符号判定点相对线段的方向（`(b.x-a.x)*(p.y-a.y) - (b.y-a.y)*(p.x-a.x) < 0` 为"内"侧），当某目标连续两次观测**从一侧翻转到另一侧**时计一次跨越，并按 `in_side()` 区分是 `line_in` 还是 `line_out`。
- **区域计数**：比较目标**上一帧质心与当前帧质心**是否在区域内：仅当"现在在区域、上一帧不在"（`now_in && !prev_in`）时，`region_count_++`——即**只统计进入**事件。
- **类别统计**：`classes_` 为空时统计所有目标；否则只累加命中白名单的 `label_id`。
- 每帧把当前质心写入 `last_centroid_` 供下一帧判断。

## 典型用法

```cpp
using namespace modeldeploy::vision;
solution::ObjectCounter counter;
counter.set_line(Point2f(160, 0), Point2f(160, 240));   // 竖向计数线
// 可选：counter.set_region({Point2f(...), ...});
// 可选：counter.set_classes({0});

for (每帧) {
    std::vector<tracking::TrackResult> tracks = tracker.update(dets);
    counter.update(tracks);
}
auto st = counter.stats();
printf("line_in=%d line_out=%d\n", st.line_in, st.line_out);
printf("区域进入=%d\n", counter.region_count());
```

## 效果

低计算开销，纯后处理即可输出稳定的进出/闯入/分类统计，适合长时间连续监控。
