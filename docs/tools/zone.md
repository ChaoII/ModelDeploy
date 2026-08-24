# Zone — 区域判断（LineZone / PolygonZone）

> 架构层：`modeldeploy::vision::tool`
> 源码：`csrc/vision/tools/zone.h` / `.cpp`

## 职责

判断空间关系——`LineZone` 检测目标是否**跨过一条参考线**（用于跨线计数 / 进出方向），`PolygonZone` 判断点是否在**多边形区域**内（用于计数 / 驻留 / 过滤）。

## 应用场景

- 车流 / 人流跨线计数、进出方向统计
- 越界 / 驻留检测
- 区域目标过滤（`filter_by_zone`）
- 车位槽位判断（`ParkingManager` 内部用一组 `PolygonZone`）

## LineZone — 跨线触发计数

```cpp
class LineZone {
    LineZone(Point2f start, Point2f end);
    void reset();                 // 清 count_/last_in_/has_last_
    int  trigger_count() const;   // 已触发次数
    bool in_side() const;         // 上一次观测是否在"内侧"
    bool trigger(const Point2f& p);  // 喂入新点，返回是否发生跨线
};
```

**算法**：用叉积符号判定点在线哪一侧 `side_in = (b.x-a.x)(p.y-a.y) - (b.y-a.y)(p.x-a.x) < 0`（视为"内"侧）。`trigger(p)` 计算当前侧，若与上一观测不同且已有上一观测则判定跨线 → `count_++` 返回 true；即**两次相邻观测"两侧翻转"计一次**，单侧停留不重复计。

```cpp
tool::LineZone z(Point2f(160,0), Point2f(160,240));
if (z.trigger(Point2f(cx,cy))) {
    if (z.in_side()) ++in_cnt; else ++out_cnt;
}
```

## PolygonZone — 多边形区域判断 / 计数

```cpp
class PolygonZone {
    PolygonZone() = default;
    explicit PolygonZone(std::vector<Point2f> points);
    void reset();
    bool contains(Point2f p) const;                 // 点是否在多边形内
    int  current_count() const;                     // 累计命中次数
    void update(const std::vector<Point2f>& pts);   // 逐点累加
    const std::vector<Point2f>& points() const;
};
```

**算法**：`contains` 用经典**射线法（奇偶规则）**：遍历每条边，若某边竖直跨越 `p.y` 且交点在右侧则翻转 `inside`；奇数次在内、偶数次在外。

```cpp
tool::PolygonZone region({Point2f(0,0),Point2f(320,0),Point2f(320,240),Point2f(0,240)});
bool in = region.contains(Point2f(100,100));
region.update({Point2f(50,50), Point2f(300,300)});   // 命中一次
int cnt = region.current_count();
```

## 自由函数：filter_by_zone

```cpp
void filter_by_zone(Detections& d, const PolygonZone& zone,
                    const std::vector<int32_t>* keep_classes = nullptr,
                    float score_threshold = 0.0f);
```

就地过滤 `Detections`：保留同时满足——置信度 ≥ `score_threshold`（若 confidence 非空）、`class_id` 在 `keep_classes`（若提供）、且**框中心点**落在 `zone.contains` 内——的检测。

```cpp
tool::filter_by_zone(d, zone, &classes, 0.5f);   // 只留下限区域内的目标
```
