# Detections — 检测结果容器与后处理工具集合

> 架构层：`modeldeploy::vision::tool`
> 源码：`csrc/vision/tools/detections.h` / `.cpp`

## 职责

一个贴合多后端 / 跟踪管线的**平行向量式检测结果容器**，并配套 **IoU / NMS / 类别过滤 / 跟踪转检测**等工具函数。

## 应用场景

- 检测器后处理（NMS、按类过滤）
- 跟踪结果与检测结果互转
- 作为 Annotator / Metrics / Zone / Slicer 的统一数据链路

## 结构体成员（按索引对齐，第 i 项构成一个目标）

```cpp
struct Detections {
    std::vector<Rect2f>              boxes;        // 检测框 (x,y,width,height)
    std::vector<int32_t>             class_id;     // 类别 id
    std::vector<float>               confidence;   // 置信度
    std::vector<Mask>                masks;        // 分割掩码（可空）
    std::vector<int32_t>             tracker_id;   // 跟踪 id（可空）
    size_t size() const;
    void reserve(size_t n);
};
```

## 关键工具函数

```cpp
float iou(const Rect2f& a, const Rect2f& b);                              // 交并比
void nms(Detections& d, float iou_threshold);                             // 贪心 NMS，就地改写
void filter_by_class(Detections& d, const std::vector<int32_t>& keep);    // 保留白名单类别
Detections from_track(const std::vector<tracking::TrackResult>& t);       // 跟踪→检测
Detections from_detections(const std::vector<tracking::Detection>& t);    // 检测→检测
```

## 原理

- 所有向量按索引平行存储；`nms` 按置信度降序排序 + `keep[]` 标记实现贪心抑制，抑制与已保留框 IoU 超阈值的框；`iou`/`nms` 均基于 `Rect2f`，是 Zone / Slicer / Metrics 的公共底层。

## 典型用法

```cpp
tool::Detections d;
d.boxes = {Rect2f(0,0,10,10), Rect2f(1,1,10,10)};
d.class_id = {0,0}; d.confidence = {0.5f, 0.9f};
tool::nms(d, 0.4f);            // 保留置信度最高的
tool::filter_by_class(d, {0}); // 只留 person
tool::Detections t = tool::from_track(track_result);  // 跟踪结果转检测
```
