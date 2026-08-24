# ObjectBlur — 目标模糊（隐私打码）

> 架构层：`modeldeploy::vision::solution`
> 源码：`csrc/vision/solutions/object_blur.h` / `.cpp`

## 职责

对图像中指定矩形框区域内的内容做**高斯模糊**（常用于人脸 / 车牌 / 敏感目标打码），其余区域保持不变。

## 应用场景

- 人脸、车牌等隐私目标的马赛克 / 打码
- 敏感区域内容遮挡
- 作为检测模型之后的**后处理步骤**（先检测出目标框，再逐个模糊）

## 输入输出

- 输入：`ImageData img`（整幅图，需 pack CPU 格式可转 `cv::Mat`）+ `Rect2f box`（`x,y,width,height`）
- 输出：`ImageData* out`（处理后的整幅图）

## 关键 API

```cpp
using modeldeploy::vision::solution::ObjectBlur;
using modeldeploy::vision::ImageData;

ObjectBlur bl(int ksize = 15);                    // 高斯核尺寸，默认 15，越大越模糊
bl.blur(const ImageData& img, const Rect2f& box, ImageData* out) const;
```

## 原理与算法

1. `img.asMat(&src)` 得到 `cv::Mat`，失败则直接返回（不写 `out`）。
2. `cv::Mat dst = src.clone()` 克隆原图。
3. 用 `box` 构造 ROI，**坐标裁剪到合法范围**（`x/y` 取 `max(0,…)`，宽高取 `min(边界,…)`），防止越界。
4. 若 ROI 宽高均 > 0，则 `cv::GaussianBlur(region, region, cv::Size(ksize,ksize), 0)` 对框内原地模糊（`sigma=0` 由核尺寸自动推导）。
5. `*out = ImageData(dst)` 封装回 `ImageData`。

> 只模糊框内、框外不受影响；自动处理越界 box。

## 典型用法

```cpp
ObjectBlur bl(11);                          // 高斯核 11
ImageData out;
for (auto& box : det.boxes)                 // 每个检测框
    bl.blur(img, Rect2f(box[0], box[1], box[2]-box[0], box[3]-box[1]), &out);
```

## 效果

轻量、无模型依赖，即可对任意矩形区域做隐私打码，适合与检测模型级联。
