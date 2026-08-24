# InferenceSlicer — 推断切片 / 滑动窗口

> 架构层：`modeldeploy::vision::tool`
> 源码：`csrc/vision/tools/slicer.h` / `.cpp`

## 职责

把大图按指定瓦片尺寸与重叠像素切成若干小块，分别推理后再把局部检测框**重映射回原图坐标**，破解模型输入尺寸上限 / 大图小目标问题。

## 应用场景

- 超大图像 / 卫星 / 病理切片 / 文档扫描的检测与分割前处理
- 小目标检测（通过重叠保留跨边界目标）
- 模型固定输入尺寸时的通用大图方案

## 关键 API

```cpp
using modeldeploy::vision::tool::InferenceSlicer;
struct Slice { ImageData tile; Rect2f offset; };   // tile=瓦片图; offset=瓦片左上角在原图坐标

InferenceSlicer slicer(int tile_w, int tile_h, int overlap_px = 0);
std::vector<Slice> slice(const ImageData& img) const;
void reassemble(const std::vector<Slice>& slices,
                const std::vector<Detections>& per_slice,
                ImageData* out, std::vector<Rect2f>* mapped_boxes);
```

## 算法原理

- **切图**：步长 `step_w = max(1, tile_w-overlap)`、`step_h = max(1, tile_h-overlap)`，从 `(0,0)` 起横竖滑动；用 `img.crop(box)` 截取，尺寸取 `min(tile, W-x)` / `min(tile, H-y)` 处理右 / 下边界（最后一行 / 列允许小于瓦片），每块记录在原图的 `offset`。
- **重映射**：遍历 `slices` 求全图 `W/H = max(offset.x+w, offset.y+h)`，创建输出空白整图；把第 i 块局部框平移到原图坐标 `mapped = Rect2f(b.x+offset.x, b.y+offset.y, b.width, b.height)`。

> **重要**：`reassemble` 只重建空白整图 + 返回映射后的框坐标，**不拼贴瓦片像素，也不做 NMS 去重**——需调用方自行对 `mapped_boxes` 做 `nms` 合并瓦片重叠产生的重复框。

## 典型用法

```cpp
tool::InferenceSlicer slicer(640, 640, 100);
auto tiles = slicer.slice(img);
std::vector<tool::Detections> per_slice;
for (auto& s : tiles) per_slice.push_back(模型推理(s.tile));
ImageData out; std::vector<Rect2f> mapped;
tool::reassemble(tiles, per_slice, &out, &mapped);
tool::Detections all; all.boxes = mapped;
tool::nms(all, 0.5f);      // 合并瓦片重叠重复框
```
