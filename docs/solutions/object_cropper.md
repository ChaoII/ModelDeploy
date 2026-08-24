# ObjectCropper — 目标裁剪

> 架构层：`modeldeploy::vision::solution`
> 源码：`csrc/vision/solutions/object_cropper.h` / `.cpp`

## 职责

按检测框 `Rect2f` 从原图裁剪出目标子图，用于"先检测、后裁剪送入下游模型 / 保存样本"等场景。

## 应用场景

- 检测 + 识别级联：先检测目标，再裁剪送入分类 / 识别 / OCR 等模型，减少计算量
- 人脸、车辆、车牌等 ROI 提取与样本保存

## 输入输出

- 输入：`ImageData img` + `Rect2f box`
- 输出：`ImageData* out`（裁剪后的子图，宽 = `box.width`，高 = `box.height`）

## 关键 API

```cpp
using modeldeploy::vision::solution::ObjectCropper;

ObjectCropper cr;
cr.crop(const ImageData& img, const Rect2f& box, ImageData* out) const;
```

## 原理与算法

实现极简，直接委托给 `ImageData::crop(box)`：

```cpp
void ObjectCropper::crop(const ImageData& img, const Rect2f& box, ImageData* out) const {
    if (!out) return;
    *out = img.crop(box);
}
```

- `ImageData::crop` 通过设备后端实现，支持 GPU 等后端；设备不支持则快速失败。

## 典型用法

```cpp
ObjectCropper cr;
ImageData face_img;
for (auto& box : face_det.predict(img))   // 每张人脸框
    cr.crop(img, box, &face_img);         // 裁剪后送入识别 / 比对
```

## 效果

与平台后端解耦的 ROI 裁剪，是"检测 → 裁剪 → 识别"级联的标准中间件。
