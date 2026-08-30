# TrackZone — 区域跟踪目标过滤

> 架构层：`modeldeploy::vision::solution`
> 源码：`csrc/vision/solutions/track_zone.h` / `.cpp`
> 示例：`examples/demo_solutions/demo_solutions.cpp`

## 职责

把跟踪结果**过滤为仅保留质心落在指定区域内的目标**，返回过滤后的 `TrackResult` 子集。相当于对跟踪链路做"感兴趣区域（ROI）"裁剪，只关注区域内对象。

## 应用场景

- 只关注禁区 / 指定通道内的目标，过滤掉区外对象
- 进入区域后再触发其它下游处理（计数、识别、测速）
- 多路目标流中按 ROI 分流

## 输入输出

- 输入：逐帧跟踪结果 `std::vector<modeldeploy::vision::tracking::TrackResult>`（由上游**检测 + 跟踪**链路产生，本类不直接依赖模型）
- 输出：`std::vector<TrackResult>`（区域内目标子集）与 `int` 数量

## 关键 API

```cpp
using modeldeploy::vision::solution::TrackZone;
using modeldeploy::vision::tracking::TrackResult;

TrackZone tz;
tz.set_region(const std::vector<Point2f>& polygon);  // 设置感兴趣区域（<3 点忽略）
tz.set_classes(const std::vector<int32_t>& cls);     // 类别白名单（空=全部）
tz.update(const std::vector<TrackResult>& tracks);   // 每帧喂入跟踪结果
std::vector<TrackResult> inside = tz.inside_tracks();// 区域内目标子集
int n = tz.inside_count();                           // 区域内目标数
tz.reset();
```

## 原理与算法

- **目标点**：取每个目标包围盒中心 `c = (box.x + box.w*0.5, box.y + box.h*0.5)`。
- **后处理过滤**：每次 `update` 清空 `inside_`，对每个目标（经类别过滤后）若 `tool::PolygonZone::contains(c)` 命中则拷贝进 `inside_`；区外目标被丢弃出跟踪集合。
- 与 ultralytics 的 TrackZone 相比，这里是**事后过滤**（post-hoc filter），不修改上游跟踪器、也不做画面掩码（frame masking）。
- 未设置区域（`set_region` 多边形 < 3 点）时 `inside_tracks()` 为空。

## 典型用法

```cpp
using namespace modeldeploy::vision;
solution::TrackZone tz;
tz.set_region({Point2f(100,0), Point2f(160,0), Point2f(160,240), Point2f(100,240)});
// 可选：tz.set_classes({0});

for (每帧) {
    std::vector<tracking::TrackResult> tracks = tracker.update(dets);
    tz.update(tracks);
}
for (const auto& t : tz.inside_tracks())
    printf("inside id=%d label=%d\n", t.track_id, t.label_id);
```

## 效果

以极低成本把跟踪流裁剪到感兴趣区域，下游只需消费过滤后的子集，避免区外干扰。
