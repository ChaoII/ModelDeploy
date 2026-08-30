# QueueManager — 单区域排队计数

> 架构层：`modeldeploy::vision::solution`
> 源码：`csrc/vision/solutions/queue_manager.h` / `.cpp`
> 示例：`examples/demo_solutions/demo_solutions.cpp`

## 职责

对**单个多边形排队区域**进行**当前帧排队长度计数**：统计此刻质心落在区域内的跟踪目标数量，即队列长度。适用于窗口 / 闸口 / 收银台的实时排队人数监测。

## 应用场景

- 服务窗口、收银台的实时排队人数监测与超量告警
- 闸口 / 入口的瞬时排队长度统计
- 按类别统计排队对象（如只数人）

## 输入输出

- 输入：逐帧跟踪结果 `std::vector<modeldeploy::vision::tracking::TrackResult>`（由上游**检测 + 跟踪**链路产生，本类不直接依赖模型）
- 输出：`int`（当前帧区域内目标数 = 排队长度）

## 关键 API

```cpp
using modeldeploy::vision::solution::QueueManager;
using modeldeploy::vision::tracking::TrackResult;

QueueManager q;
q.set_region(const std::vector<Point2f>& polygon);   // 设置排队区域（<3 点忽略）
q.set_classes(const std::vector<int32_t>& cls);      // 类别白名单（空=全部）
q.update(const std::vector<TrackResult>& tracks);    // 每帧喂入跟踪结果
int len = q.queue_count();                           // 当前帧区域内目标数
q.reset();
```

## 原理与算法

- **目标点**：取每个目标包围盒中心 `c = (box.x + box.w*0.5, box.y + box.h*0.5)`。
- **逐帧计数**：每次 `update` 先将计数归零，再对每个目标（经类别过滤后）判定 `tool::PolygonZone::contains(c)`，命中则 `count_++`，得到**当前帧**的排队长度。
- 与 ultralytics 的 Queue 方案相比，这里**仅统计当前帧在区目标数**，不额外判定目标"是否从区外进入"，实现更简洁。
- 未设置区域（`set_region` 多边形 < 3 点）时 `queue_count()` 恒为 0。

## 典型用法

```cpp
using namespace modeldeploy::vision;
solution::QueueManager q;
q.set_region({Point2f(100,0), Point2f(160,0), Point2f(160,240), Point2f(100,240)});
// 可选：q.set_classes({0});

for (每帧) {
    std::vector<tracking::TrackResult> tracks = tracker.update(dets);
    q.update(tracks);
    printf("queue_length=%d\n", q.queue_count());
}
```

## 效果

每帧即时给出排队长度，配合阈值即可做超量告警，纯后处理、开销极低。
