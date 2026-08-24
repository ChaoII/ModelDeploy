# Annotator — 可视化标注工具

> 架构层：`modeldeploy::vision::tool`
> 源码：`csrc/vision/tools/annotator.h` / `.cpp`

## 职责

在一个绑定好的 `ImageData` 画面上直接绘制**矩形、文字、线段、圆形和半透明填充多边形**，用于结果可视化叠加。

## 应用场景

- 推理结果叠加框 / 标签 / 轨迹线
- 区域遮罩着色（配合 Zone 可视化）
- 生成可视化 demo 输出图

## 关键 API

```cpp
using modeldeploy::vision::tool::Annotator;

Annotator ann(ImageData* frame);      // 绑定画面（持裸指针，需保证生命周期）
ann.rectangle(const Rect2f& b, const cv::Scalar& color, int thickness = 2);
ann.text(const std::string& s, Point2f org, const cv::Scalar& color, double scale = 0.6);
ann.line(Point2f a, Point2f b, const cv::Scalar& color, int thickness = 2);
ann.circle(Point2f c, int r, const cv::Scalar& color, int thickness = 2);
ann.fill_polygon(const std::vector<Point2f>& pts, const cv::Scalar& color, double alpha = 0.3);

// 自由函数（仍在 tool 命名空间）
void draw_box_labels(const Detections& d, ImageData* frame,
                     const std::unordered_map<int,std::string>& labels = {});
void draw_traces(const std::vector<Point2f>& trace, ImageData* frame,
                 const cv::Scalar& color, int thickness = 2);
```

## 原理

- 每次绘图内部先经 `frame_->asMat()` 拿到与 `frame` **共享内存**的 `cv::Mat` 直接绘制，结果即时写回原 `frame`，无需返回新图。
- `fill_polygon` 通过 `clone()` 副本填充再用 `addWeighted` 混合，实现**半透明区域标注**。
- `draw_box_labels`：为每个检测画红色矩形（BGR `(0,0,255)`），上方写"标签 置信度"（黄字 `(0,255,255)`）；`labels` 做 `class_id→名字` 映射，缺失用数字。
- `draw_traces`：把一列轨迹点用直线依次相连。
- 颜色参数均按 **BGR** 解释。

## 典型用法

```cpp
using namespace modeldeploy::vision;
ImageData frame(cv::imread("in.jpg"));
tool::Annotator ann(&frame);
ann.rectangle(Rect2f(10,10,60,40), cv::Scalar(0,0,255), 2);
ann.text("person", Point2f(10,6), cv::Scalar(0,255,255), 0.6);
ann.fill_polygon({Point2f(0,0),Point2f(100,0),Point2f(100,100)}, cv::Scalar(0,200,0), 0.3);
// 或一步到位：
tool::draw_box_labels(d, &frame, {{0,"person"},{1,"car"}});
```
