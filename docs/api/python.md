# Python 绑定（pybind11）

## 1. 安装

```bash
cd ModelDeploy
pip install build
python -m build
pip install dist/modeldeploy-*.whl
```

构建后生成 `.pyi` 存根：

```bash
pybind11-stubgen modeldeploy
```

## 2. 使用

```python
import cv2
import modeldeploy as md

option = md.RuntimeOption()
option.use_ort_backend()
option.use_cpu()
option.set_cpu_thread_num(4)

# 设备（OPENCL/VULKAN 需显式 MNN 后端，否则 fail-closed）
option.use_mnn_backend()
option.set_device(md.Device.OPENCL, 0)   # 或 md.Device.VULKAN / md.Device.GPU / md.Device.TPU

# 其它后端（均无参数）
# option.use_ncnn_backend()  /  option.use_trt_backend()  /  option.use_sophgo_backend()
# GPU：option.use_gpu(0) 或 option.set_device(md.Device.GPU, 0)

# 目标检测
model = md.vision.UltralyticsDet("yolo11n.onnx", option)
model.preprocessor.size = [640, 640]
model.postprocessor.conf_threshold = 0.25

img = cv2.imread("test.jpg")
results = model.predict(img)
for r in results:
    print(r.label_id, r.score, r.box)
```

### 设备帧 NV12（`ImageData.from_device_nv12`）

零拷贝借用 host/device NV12 帧，设备语义经 `dev` 参数与 C/C/C#/Rust 对齐（缺省 CPU）：

```python
import numpy as np
y  = np.zeros(h * step_y, dtype=np.uint8)
uv = np.zeros((h // 2) * step_uv, dtype=np.uint8)
# host NV12（CPU 缺省）
img_cpu = modeldeploy.ImageData.from_device_nv12(y, uv, w, h, dev=modeldeploy.Device.CPU)
# 设备 NV12（OPENCL/VULKAN 等，需显式 use_mnn_backend()）
img_dev = modeldeploy.ImageData.from_device_nv12(y, uv, w, h, dev=modeldeploy.Device.OPENCL)
```
y/uv 指向调用方内存、不拷贝，调用方须保证缓冲在 `predict` 期间存活。

### TTS（Kokoro）

```python
import modeldeploy

option = modeldeploy.RuntimeOption()
option.use_ort_backend()
option.use_cpu()

# Kokoro：24kHz，构造参数较多（onnx/tokens/lexicons/voices.bin/jieba/norm_dir）
kokoro = modeldeploy.audio.Kokoro("kokoro.onnx", "tokens.txt", ["lex_en.txt", "lex_zh.txt"],
                                  "voices.bin", "dict/", "text_normalization/", option)
wav = kokoro.predict("你好，世界。", "zf_001", 1.0)      # 返回 float32 列表

# 统一流式：predict_stream 返回 chunks 列表（每块 float32 数组）
chunks = kokoro.predict_stream("你好，世界。", "zf_001", 1.0, chunk_frames=120)
```

`chunk_frames == 0` 等价一次性合成（单块回调整段）；`predict_stream(text, voice, speed, chunk_frames)` 返回逐块 float32 数组，平铺可 `np.concatenate`。超长文本走 `predict_stream`（分块，`>120` 字符触发多块）。

## 3. 目标检测（UltralyticsDet）

Ultralytics YOLO 检测模型（`.onnx`/`.mnn`/`.engine`/`.param+.bin`/`.bmodel` 均可，取决于后端）。结果类型 `DetectionResult`（字段 `box: Rect2f`、`label_id: int`、`score: float`）。

```python
import cv2
import modeldeploy as md

# 1. 构造（模型路径 + RuntimeOption，选项配置见上节）
option = md.RuntimeOption()
option.use_ort_backend()
option.use_cpu()

det = md.vision.UltralyticsDet("yolo11n.onnx", option)

# 2. 预处理/后处理参数（属性直接赋值）
det.preprocessor.size = [640, 640]                    # letterbox 输入尺寸（默认 [640, 640]）
det.preprocessor.padding_value = [114.0, 114.0, 114.0]  # 填充灰值
det.postprocessor.conf_threshold = 0.25               # 置信度阈值（默认 0.25）
det.postprocessor.nms_threshold = 0.5                 # NMS IoU 阈值（默认 0.5）

# 3. 单图推理：输入 BGR ndarray（也可传 ImageData，如 NV12 设备帧），返回 list[DetectionResult]
img = cv2.imread("test.jpg")
results = det.predict(img)
for r in results:
    print(r.label_id, r.score, r.box.x, r.box.y, r.box.width, r.box.height)

# 4. 批量推理：返回 list[list[DetectionResult]]（按图分组）
batch = det.batch_predict([cv2.imread("a.jpg"), cv2.imread("b.jpg")])
print([len(rs) for rs in batch])

# 5. 可视化：md.vision.vis_det 返回 BGR ndarray
#    签名：vis_det(image, result, threshold=0.5, label_map={}, font_path="", font_size=14, alpha=0.15, save_result=False)
label_map = det.get_label_map("names")   # 从模型元数据读类别表（ultralytics 导出的 onnx 键为 "names"），返回 dict[int, str]
vis = md.vision.vis_det(img, results, threshold=0.5, label_map=label_map,
                        font_path="msyh.ttc", font_size=14, alpha=0.15)
cv2.imwrite("det_vis.jpg", vis)

# 6. 多线程：clone() 深拷贝独立实例（每线程持有一个，互不干扰）
det2 = det.clone()
```

批量推理时各图会按 `preprocessor.size` 统一 letterbox 后拼成 batch，原图尺寸可不同；`batch_predict` 的返回按图分组对应。

## 4. 实例分割（UltralyticsSeg）

Ultralytics YOLO 分割模型（`yolo11n-seg.onnx` 等，后端支持同目标检测）。结果类型 `InstanceSegResult`（字段 `box: Rect2f`、`label_id: int`、`score: float`、`mask: Mask`）。

```python
import cv2
import numpy as np
import modeldeploy as md

# 1. 构造（模型路径 + RuntimeOption，选项配置见上节）
option = md.RuntimeOption()
option.use_ort_backend()
option.use_cpu()

seg = md.vision.UltralyticsSeg("yolo11n-seg.onnx", option)

# 2. 预处理/后处理参数（属性直接赋值）
seg.preprocessor.size = [640, 640]                       # letterbox 输入尺寸（默认 [640, 640]）
seg.preprocessor.padding_value = [114.0, 114.0, 114.0]   # 填充灰值
seg.postprocessor.conf_threshold = 0.25                  # 置信度阈值（默认 0.25）
seg.postprocessor.nms_threshold = 0.5                    # NMS IoU 阈值（默认 0.5）
# 掩码二值化阈值 mask_threshold（默认 0.5）未在 Python 绑定暴露（C++/C API/C# 可设）

# 3. 单图推理：输入 BGR ndarray（也可传 ImageData），返回 list[InstanceSegResult]
img = cv2.imread("test.jpg")
results = seg.predict(img)
for r in results:
    print(r.label_id, r.score, r.box.x, r.box.y, r.box.width, r.box.height)
    # mask：Mask{buffer, shape}，buffer 为 uint8 0/1 的 H*W 平铺列表（行主序）
    mask = np.array(r.mask.buffer, dtype=np.uint8).reshape(r.mask.shape)   # shape = [H, W]
    print("mask", mask.shape, "fg_ratio", mask.mean())

# 4. 批量推理：返回 list[list[InstanceSegResult]]（按图分组）
batch = seg.batch_predict([cv2.imread("a.jpg"), cv2.imread("b.jpg")])

# 5. 可视化：md.vision.vis_iseg 返回 BGR ndarray
#    签名：vis_iseg(image, result, threshold=0.5, font_path="", font_size=14, alpha=0.15, save_result=False)
vis = md.vision.vis_iseg(img, results, threshold=0.5, font_path="msyh.ttc", font_size=14, alpha=0.3)
cv2.imwrite("iseg_vis.jpg", vis)

# 6. 类别表与多线程：get_label_map / clone()（用法同目标检测）
label_map = seg.get_label_map("names")
seg2 = seg.clone()
```

## 5. 轻量分割一切（FastSAM）

`FastSam`（`fastsam-s.onnx` 等）一次性输出 box + mask，结果复用 `InstanceSegResult`（消费方式同实例分割）。
`predict_with_prompts` 在全量结果上按提示过滤实例，**不重跑网络**；提示为空等价全图 `predict`。

> Python 绑定的 `preprocessor`/`postprocessor` 属性存在但未暴露可写参数，使用默认值
> （输入 640×640，conf 0.30 / nms 0.5 / mask 0.5）；需改输入尺寸（如官方推荐的 1024×1024）
> 请用 C++ `set_size` / C API `md_model_set_input_size` / C# `SetInputSize` / Rust `set_input_size`。

```python
import cv2
import modeldeploy as md

sam = md.vision.FastSam("fastsam-s.onnx", option)   # option 构造见上节

# 1. 全图（Everything）分割：返回 list[InstanceSegResult]
img = cv2.imread("test.jpg")
res = sam.predict(img)
print(len(res), res[0].score, res[0].mask.shape)

# 2. 提示过滤：FastSamPrompts{bboxes: list[Rect2f], points: list[Point2f], point_labels: list[int]}
#    - bboxes（x, y, width, height，原图像素）：每个框取 IoU 最大的实例
#    - points + point_labels（等长）：掩码命中该点则保留(1)/剔除(0)
bb = md.vision.Rect2f(); bb.x, bb.y, bb.width, bb.height = 100.0, 80.0, 220.0, 180.0
pt = md.vision.Point2f(); pt.x, pt.y = 150.0, 130.0
p = md.vision.FastSamPrompts()
p.bboxes = [bb]
p.points = [pt]
p.point_labels = [1]                    # 1=前景保留 / 0=背景剔除
pr = sam.predict_with_prompts(img, p)

# 3. 可视化同实例分割（vis_iseg）
vis = md.vision.vis_iseg(img, pr, threshold=0.3)
cv2.imwrite("fastsam_vis.jpg", vis)
```

注意：Python `Rect2f`/`Point2f` 仅绑定默认构造器，坐标须用属性赋值（`Rect2f(100, 80, 220, 180)` 不可用）。

## 6. 语义分割（UltralyticsSem）

语义分割模型（`yolo26n-sem` 等，cityscapes 19 类）。单图返回 `SemSegResult`（字段 `labels: list[int]`、`shape: [H, W]`、`num_classes: int`）。

```python
import cv2
import numpy as np
import modeldeploy as md

sem = md.vision.UltralyticsSem("yolo26n-sem.onnx", option)
sem.preprocessor.size = [640, 640]     # 可调；postprocessor 无参数（逐像素 argmax + 去除 letterbox 边）

d = sem.predict(img)
labels = np.array(d.labels, dtype=np.uint8).reshape(d.shape)   # 每像素类别索引 [0, d.num_classes)
print(d.num_classes, labels.shape, np.unique(labels))

batch = sem.batch_predict([cv2.imread("a.jpg"), cv2.imread("b.jpg")])   # list[SemSegResult]（按图分组）
label_map = sem.get_label_map("names")

# Python 绑定未提供 vis_sem；叠加调色板可视化请用 C++ vis_sem / C API md_draw_result
```

## 7. 深度估计（UltralyticsDepth）

深度估计模型（`yolo26n-depth` 等）。单图返回 `DepthResult`（字段 `depth: list[float]`、`shape: [H, W]`；log 空间深度已 `exp` 还原为米）。

```python
import cv2
import numpy as np
import modeldeploy as md

dep = md.vision.UltralyticsDepth("yolo26n-depth.onnx", option)
dep.preprocessor.size = [640, 640]     # 可调；postprocessor 无参数（exp 还原 + 去除 letterbox 边）

dr = dep.predict(img)
depth = np.array(dr.depth, dtype=np.float32).reshape(dr.shape)  # 每像素深度（米）
print(depth.shape, depth.min(), depth.max())

batch = dep.batch_predict([cv2.imread("a.jpg"), cv2.imread("b.jpg")])   # list[DepthResult]（按图分组）

# Python 绑定未提供 vis_depth；JET 伪彩图请用 C++ vis_depth / C API md_draw_result
```

## 8. 姿态与关键点族（UltralyticsPose / HandKeypoint / VehicleKeypoint / FaceLandmark）

姿态与关键点模型族结果类型统一为 `KeyPointsResult`（字段 `box: Rect2f`、`keypoints: list[Point3f]`（`x/y/z`，`z` 为关键点置信度）、`label_id: int`、`score: float`）。`UltralyticsPose`（COCO 17 点人体骨架）、`HandKeypoint`（21 点手部）在 `md.vision`；`VehicleKeypoint`（4 车轮关键点）与 `FaceLandmark`（InsightFace 2d106 面部 106 点，`z` 恒为 0）在 `md.vision.landmark` 子模块。

```python
import cv2
import modeldeploy as md

# 1. 构造（option 配置见上节；VehicleKeypoint/FaceLandmark 的 option 可省略，默认 RuntimeOption()）
option = md.RuntimeOption()
option.use_ort_backend()
option.use_cpu()

pose = md.vision.UltralyticsPose("yolo11n-pose.onnx", option)
hand = md.vision.HandKeypoint("hand.onnx", option)
vehicle = md.vision.landmark.VehicleKeypoint("vehicle.onnx", option)
face = md.vision.landmark.FaceLandmark("face_landmark.onnx", option)

# 2. 参数设置
pose.preprocessor.size = [640, 640]          # letterbox 输入尺寸（默认 [640, 640]）
pose.postprocessor.conf_threshold = 0.30     # 置信度阈值（默认 0.30）
pose.postprocessor.nms_threshold = 0.45      # NMS IoU 阈值（默认 0.5）
pose.postprocessor.set_keypoints_num(17)     # 关键点数（默认 17，须与模型输出一致）
hand.set_keypoints_num(21)                   # 手部点数（构造默认 21）
vehicle.set_keypoints_num(4)                 # 车轮点数（构造默认 4，不同车型可覆盖）
# FaceLandmark 固定 106 点（InsightFace 2d106 仿射对齐），无可调参数；输入须为人脸裁剪图

# 3. 单图推理：返回 list[KeyPointsResult]
img = cv2.imread("test.jpg")
results = pose.predict(img)
for r in results:
    print(r.label_id, r.score, r.box.x, r.box.y, r.box.width, r.box.height)
    for kp in r.keypoints:
        print(kp.x, kp.y, kp.z)              # z 为关键点置信度
face_crop = cv2.imread("face_crop.jpg")
fr = face.predict(face_crop)                 # -> 单元素 list（box 为整图、score=1.0、106 点 z=0）

# 4. 批量推理：返回 list[list[KeyPointsResult]]（HandKeypoint / FaceLandmark 未绑定 batch_predict）
batch = pose.batch_predict([cv2.imread("a.jpg"), cv2.imread("b.jpg")])
vbatch = vehicle.batch_predict([cv2.imread("a.jpg"), cv2.imread("b.jpg")])

# 5. 可视化：md.vision.vis_keypoints（vis_pose / vis_hand 未绑定 Python）
#    签名：vis_keypoints(image, result, font_path="", font_size=14, landmark_radius=4, alpha=0.15, save_result=False, draw_lines=False)
vis = md.vision.vis_keypoints(img, results, font_path="msyh.ttc", font_size=14, landmark_radius=4, alpha=0.3)
cv2.imwrite("pose_vis.jpg", vis)

# 6. 多线程：clone() 深拷贝独立实例（HandKeypoint 未绑定 clone）
pose2 = pose.clone()
vehicle2 = vehicle.clone()
```

## 9. OBB（旋转框检测）（UltralyticsObb）

Ultralytics YOLO-OBB 模型（`yolo11n-obb.onnx` 等），输入默认 [1024, 1024]。单图返回 `list[ObbResult]`（字段 `rotated_box: RotatedRect`、`label_id: int`、`score: float`）；`rotated_box` 为旋转框（`xc/yc` 中心点 + `width/height` 边长 + `angle` 弧度角，原图像素坐标）。

```python
import cv2
import modeldeploy as md

# 1. 构造（option 配置见上节）
option = md.RuntimeOption()
option.use_ort_backend()
option.use_cpu()
obb = md.vision.UltralyticsObb("yolo11n-obb.onnx", option)

# 2. 预处理/后处理参数（属性直接赋值）
obb.preprocessor.size = [1024, 1024]       # letterbox 输入尺寸（默认 [1024, 1024]）
obb.preprocessor.padding_value = 114.0      # 填充灰值（单值标量）
obb.postprocessor.conf_threshold = 0.25    # 置信度阈值（默认 0.25）
obb.postprocessor.nms_threshold = 0.45     # NMS IoU 阈值（默认 0.5）

# 3. 单图推理：返回 list[ObbResult]
img = cv2.imread("test.jpg")
results = obb.predict(img)
for r in results:
    rb = r.rotated_box
    print(r.label_id, r.score, rb.xc, rb.yc, rb.width, rb.height, rb.angle)

# 4. 批量推理：返回 list[list[ObbResult]]（按图分组）
batch = obb.batch_predict([cv2.imread("a.jpg"), cv2.imread("b.jpg")])

# 5. 可视化：md.vision.vis_obb 返回 BGR ndarray
#    签名：vis_obb(image, result, threshold=0.5, font_path="", font_size=14, alpha=0.15, save_result=False)
vis = md.vision.vis_obb(img, results, threshold=0.5, font_path="msyh.ttc", font_size=14, alpha=0.3)
cv2.imwrite("obb_vis.jpg", vis)

# 6. 多线程：clone() 深拷贝独立实例（每线程持有一个，互不干扰）
obb2 = obb.clone()
```

## 10. 图像分类（Classification）

分类模型（`yolo11n-cls.onnx` 等，输入默认 [224, 224] + center crop）。单图返回**单个** `ClassifyResult`（字段 `label_ids: list[int]`、`scores: list[float]`，二者按序配对；Top-K 由模型后处理决定）。注意：Python 绑定的 `ClassificationPostprocessor` **只暴露 `set_multi_label`，无 `set_topk`**。

```python
import cv2
import modeldeploy as md

# 1. 构造（option 配置见上节）
cls = md.vision.Classification("yolo11n-cls.onnx", option)

# 2. 预处理/后处理参数（属性直接赋值）
cls.preprocessor.size = [224, 224]         # 输入尺寸（默认 [224, 224]）
cls.preprocessor.disable_center_crop()     # 关闭 center crop（默认开启）
cls.postprocessor.set_multi_label(True)    # 多标签模式（默认 False）

# 3. 单图推理：返回单个 ClassifyResult（label_ids 与 scores 逐位配对）
img = cv2.imread("test.jpg")
cr = cls.predict(img)
for lid, s in zip(cr.label_ids, cr.scores):
    print(lid, s)

# 4. 批量推理：返回 list[ClassifyResult]（每图一个）
batch = cls.batch_predict([cv2.imread("a.jpg"), cv2.imread("b.jpg")])

# 5. 可视化：md.vision.vis_cls；top_k / threshold 为必填位置参数
#    签名：vis_cls(image, result, top_k, threshold, font_path="", font_size=14, alpha=0.15, save_result=False)
vis = md.vision.vis_cls(img, cr, 5, 0.35, font_path="msyh.ttc", font_size=14, alpha=0.3)
cv2.imwrite("cls_vis.jpg", vis)

# 6. 多线程：clone() 深拷贝独立实例
cls2 = cls.clone()
```

## 11. OCR（PaddleOCR + 子模型）

PaddleOCR 三段流水线：文本检测（`DBDetector`）→ 方向分类（`Classifier`）→ 文本识别（`Recognizer`），构造时依次传 det/cls/rec 模型路径与字符字典 `dict_path`（`option` 为必填位置参数）。单图 `predict` 返回**单个** `OCRResult`：`text`（识别文本）、`boxes`（每行 4 点共 8 个整数，原图像素，按上下序排列）、`rec_scores`、`cls_labels`、`cls_scores` 五个列表**逐行配对**（det 未检出文本框时对整图直接识别，此时 `boxes` 为空）。

```python
import cv2
import modeldeploy as md

# 1. 构造（det/cls/rec/dict 四个路径 + option；option 配置见上节）
ocr = md.vision.PaddleOCR("det.onnx", "cls.onnx", "rec.onnx", "dict.txt", option)

# 2. 参数设置（主流水线 batch 属性 + 经 get_* 子模型链式设置；括号内为默认值）
ocr.cls_batch_size = 6                     # 方向分类子模型 batch（默认 6）
ocr.rec_batch_size = 8                     # 文本识别子模型 batch（默认 8）
det = ocr.get_detector()                   # -> DBDetector
det.preprocessor.max_side_len = 960        # 检测最长边（默认 960）
det.postprocessor.det_db_thresh = 0.3      # DB 二值化阈值（默认 0.3）
det.postprocessor.det_db_box_thresh = 0.6  # 框置信度阈值（默认 0.6）
det.postprocessor.det_db_unclip_ratio = 1.5   # 扩框比例（默认 1.5）
det.postprocessor.det_db_score_mode = "slow"  # 框得分模式（默认 "slow"）
det.postprocessor.use_dilation = 0         # 是否膨胀（int，默认 0）
ocr.get_recognizer().preprocessor.rec_image_shape = [3, 48, 320]  # 识别输入形状（默认 [3, 48, 320]）
ocr.get_classifier().postprocessor.cls_thresh = 0.9               # 方向分类阈值（默认 0.9）

# 3. 单图推理：返回单个 OCRResult（text/boxes/rec_scores/cls_labels 逐行配对）
img = cv2.imread("test.jpg")
r = ocr.predict(img)
for i in range(len(r.text)):
    print(r.text[i], r.rec_scores[i], r.cls_labels[i], r.boxes[i])  # boxes：4 点共 8 整数

# 4. 批量推理：返回 list[OCRResult]（每图一个）
batch = ocr.batch_predict([cv2.imread("a.jpg"), cv2.imread("b.jpg")])

# 5. 可视化：md.vision.vis_ocr 返回 BGR ndarray
#    签名：vis_ocr(image, result, font_path="", font_size=14, alpha=0.15, save_result=False)
vis = md.vision.vis_ocr(img, r, font_path="msyh.ttc", font_size=14, alpha=0.3)
cv2.imwrite("ocr_vis.jpg", vis)

# 6. 多线程：clone() 深拷贝独立实例（主流水线三个子模型一起深拷贝）
ocr2 = ocr.clone()

# 7. 子模型独立使用（也可不经 PaddleOCR 单独构造；predict 均返回 OCRResult）
db = md.vision.DBDetector("det.onnx", option)                # predict -> OCRResult.boxes
rec = md.vision.Recognizer("rec.onnx", "dict.txt", option)   # predict -> OCRResult.text/rec_scores
clr = md.vision.Classifier("cls.onnx", option)               # predict -> OCRResult.cls_labels/cls_scores
boxes = db.predict(img).boxes        # 文本框（4 点 8 整数列表）
text = rec.predict(img).text         # 识别文本（整图为 1 行）
label = clr.predict(img).cls_labels  # 方向标签（0°/180°，整图为 1 项）
```

## 12. OCR 进阶（版面 / 表格 / 公式 / 文档转 Markdown）

进阶四件套全部绑定 Python：版面分析 `StructureV2Layout`、表格 `StructureV2Table`/`PPStructureV2Table`、公式 `FormulaRecognizer`，以及把它们**组合**成整页 Markdown 的 `DocToMarkdown`（`set_layout/set_ocr/set_table/set_formula` → `predict` 返回 Markdown 字符串）。

```python
import cv2
import modeldeploy as md

option = md.RuntimeOption()
option.use_ort_backend()
option.use_cpu()

# 1. 版面分析：StructureV2Layout(model_file, option) -> list[DetectionResult]
#    CDLA 版面类别检测（picodet），返回每个版面区域框 + 类别
layout = md.vision.StructureV2Layout("layout.onnx", option)
layout.preprocessor.layout_image_shape = [3, 800, 608]  # 输入 c,h,w（默认 [3, 800, 608]）
layout.preprocessor.static_shape_infer = True           # 静态输入形状（默认 True）
layout.postprocessor.score_threshold = 0.4             # 置信度阈值（默认 0.4）
layout.postprocessor.nms_threshold = 0.5               # NMS IoU 阈值（默认 0.5）
layout.postprocessor.num_class = 5                     # 版面类别数（默认 5）
img = cv2.imread("doc.jpg")
boxes = layout.predict(img)                            # list[DetectionResult]
for b in boxes:
    print(b.label_id, b.score, b.box.x, b.box.y, b.box.width, b.box.height)

# 2. 表格结构识别：StructureV2Table(model, table_dict, option) -> OCRResult
#    仅表格结构（SLANet），输出 table_html / table_structure
table = md.vision.StructureV2Table("table.onnx", "table_dict.txt", option)
tr = table.predict(img)
print(tr.table_html)        # '<html><body><table>...' 完整表格 HTML
print(tr.table_structure)   # ['<td>', '</td>', ...] 结构 token 列表
print(tr.table_boxes)       # 单元框（4 点共 8 整数）

# 3. 端到端表格：PPStructureV2Table(det, rec, table, rec_dict, table_dict, option)
#    检测 + 识别 + 表结构串联，option 为必填位置参数（可带 keyword）
ppt = md.vision.PPStructureV2Table(
    "det.onnx", "rec.onnx", "table.onnx",
    "rec_dict.txt", "table_dict.txt", option=option)
ppt.rec_batch_size = 8                # 识别子模型 batch（默认 8）
pr = ppt.predict(img)
print(pr.table_html)
for i in range(len(pr.text)):
    print(pr.text[i], pr.rec_scores[i], pr.boxes[i])   # 单元格文字逐行

# 4. 公式识别：FormulaRecognizer(model, dict, option) -> str（LaTeX）
formula = md.vision.FormulaRecognizer("formula.onnx", "dict.txt", option)
tex = formula.predict(cv2.imread("equation.jpg"))     # -> 'x^2 + y^2 = r^2' 等 LaTeX
print(tex)

# 5. 文档转 Markdown：DocToMarkdown 组合器
#    先构造各子模型，再 set_* 注入；layout 为必须，ocr/table/formula 至少其一
ocr = md.vision.PaddleOCR("det.onnx", "cls.onnx", "rec.onnx", "dict.txt", option)
doc = md.vision.DocToMarkdown()       # 默认空构造
doc.set_layout(layout)                # 版面（必须）
doc.set_ocr(ocr)                      # 文本区 OCR
doc.set_table(ppt)                    # 表格区
doc.set_formula(formula)              # 公式区（可省）
print(doc.ready())                    # 是否已配置 layout + 至少一个内容识别器
markdown = doc.predict(img)           # -> Markdown 字符串
print(markdown)
```

> `DocToMarkdown` 为**单列自上而下**顺序排版（不做多栏重排）；公式以 `$...$`、表格以 HTML 呈现。`set_*` 在 Python 中通过 `keep_alive` 保证子模型寿命被托管，可安全复用已构造的 `layout/ocr/ppt/formula`。

## 13. 已绑定模块
- **核心**：`RuntimeOption`、`Runtime`、`Tensor`、`BaseModel`、`Device`、`Backend`
- **视觉模型**：`UltralyticsDet/Seg/Obb/Pose`、`UltralyticsSem/Depth`、`FastSam`、`HandKeypoint`、`landmark.VehicleKeypoint/FaceLandmark`、`Classification`、`Scrfd`、`SeetaFace*`、`LprPipeline`、`PaddleOCR`、`PedestrianAttribute` 等
- **结果结构**：`DetectionResult`、`InstanceSegResult`、`SemSegResult`、`DepthResult`、`OCRResult`、`KeyPointsResult` 等
- **可视化**：`vis_det`、`vis_iseg`、`vis_keypoints`、`vis_ocr` 等
- **音频**：`Kokoro`（TTS，`predict_stream` 返回 chunks 列表）、`SenseVoice` 等

## 14. 性能测试

```python
import time
results = None
for _ in range(loop_count):
    results = model.predict(image)
print(f"{loop_count / elapsed} FPS")
```
