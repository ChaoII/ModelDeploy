# C# 绑定（.NET）

C# 绑定封装 C API，命名空间 `ModelDeploy`（模型类在 `ModelDeploy.Models`）。解决方案见 `csharp/ModelDeploy.sln`。

## 1. 引入

```csharp
using ModelDeploy;
using ModelDeploy.Models;
```

## 2. RuntimeOption 与检测示例

C# 没有 `UseCpu()`；改 CPU 用 `SetDevice(Device.CPU)`。后端用 `UseOrt()`/`UseMnn()`/`UseTrt()`/`UseSophgo()`/`UseNcnn()`，设备用 `SetDevice(Device dev, int deviceId = 0)`：

```csharp
using ModelDeploy;
using ModelDeploy.Models;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);
option.SetCpuThreads(4);

// 设备（OPENCL/VULKAN 需显式 MNN 后端，否则 fail-closed）
// option.UseMnn().SetDevice(Device.OPENCL, 0);
// option.SetDevice(Device.VULKAN, 0);
// GPU：option.UseOrt().SetDevice(Device.GPU, 0);

var model = new DetectionModel("yolo11n.onnx", option);
model.SetInputSize(640, 640);

using var img = VisionImage.Read("test.jpg");
using var pred = model.Predict(img);
foreach (var r in pred) {
    Console.WriteLine($"{r.LabelId} {r.Score} {r.Box}");
}
```

## 3. 目标检测（`DetectionModel`）

检测模型类 `ModelDeploy.Models.DetectionModel`（对应 `MD_MODEL_DETECTION`）。结果类型 `ModelDeploy.Results.DetectionResult`（属性 `Box: RectF{X,Y,Width,Height}`、`LabelId: int`、`Score: float`）；`Predict` 返回 `Prediction<DetectionResult>`（实现 `IReadOnlyList<T>`，`using`/`Dispose` 释放底层结果句柄）。

```csharp
using System;
using System.Collections.Generic;
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var det = new DetectionModel("yolo11n.onnx", option);

// 预处理/后处理参数（类型化 setter，编译期检查；通用 setter 为 SetParam(name, value)，见下）
det.SetInputSize(640, 640);      // letterbox 输入尺寸
det.SetConfThreshold(0.25);      // 置信度阈值（默认 0.25）
det.SetNmsThreshold(0.45);       // NMS IoU 阈值（默认 0.5）
// 自省：本 kind 支持的参数名与类型（'I'/'D'/'B'/'S'）
Console.WriteLine(string.Join(", ", det.ParamNames()));

using var img = VisionImage.Read("test.jpg");

// 单图推理：Predict 返回 Prediction<DetectionResult>（可 foreach / 索引 / .Count）
using var pred = det.Predict(img);
foreach (var r in pred) {
    Console.WriteLine($"{r.LabelId} {r.Score:F3} ({r.Box.X},{r.Box.Y},{r.Box.Width},{r.Box.Height})");
}

// 可视化：pred.Draw 直达 C++ vis_det（句柄直达原始结果；绘制到任意 VisionImage 画布）
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions {
    Threshold = 0.25,
    LabelMap = new Dictionary<int, string> { { 0, "person" }, { 1, "bicycle" }, { 2, "car" } },
    FontSize = 14,
    Alpha = 0.3,
});
canvas.Save("det_vis.jpg");

// 批量推理：PredictBatch 按图返回 IReadOnlyList<DetectionResult[]>
using var img2 = VisionImage.Read("bus.jpg");
var batch = det.PredictBatch(new[] { img, img2 });
for (int g = 0; g < batch.Count; g++) {
    Console.WriteLine($"image {g}: {batch[g].Length} objects");
}

// 多线程：Clone() 深拷贝独立实例（每线程持有一个，互不干扰）
using var det2 = det.Clone();
```

## 4. 实例分割（`InstanceSegModel`）

实例分割模型类 `ModelDeploy.Models.InstanceSegModel`（对应 `MD_MODEL_INSTANCE_SEG`）。结果类型 `ModelDeploy.Results.InstanceSegResult`（属性 `Box: RectF`、`LabelId: int`、`Score: float`、`Mask: byte[]` + `MaskWidth`/`MaskHeight`；`Mask` 为 uint8 0/1，行主序 `MaskHeight * MaskWidth`）。

```csharp
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var seg = new InstanceSegModel("yolo11n-seg.onnx", option);
seg.SetInputSize(640, 640);      // letterbox 输入尺寸
seg.SetConfThreshold(0.25);      // 置信度阈值（默认 0.25）
seg.SetNmsThreshold(0.45);       // NMS IoU 阈值（默认 0.5）
seg.SetMaskThreshold(0.5);       // 掩码二值化阈值（默认 0.5）

using var img = VisionImage.Read("test.jpg");

// 单图推理：Prediction<InstanceSegResult>（foreach / 索引 / .Count）
using var pred = seg.Predict(img);
foreach (var r in pred) {
    Console.WriteLine($"{r.LabelId} {r.Score:F3} ({r.Box.X},{r.Box.Y},{r.Box.Width},{r.Box.Height})");
    // 掩码逐像元读取：r.Mask[y * r.MaskWidth + x]
    Console.WriteLine($"mask {r.MaskWidth}x{r.MaskHeight}, bytes={r.Mask.Length}");
}

// 批量推理：按图返回 IReadOnlyList<InstanceSegResult[]>
using var img2 = VisionImage.Read("bus.jpg");
var batch = seg.PredictBatch(new[] { img, img2 });
for (int g = 0; g < batch.Count; g++) {
    Console.WriteLine($"image {g}: {batch[g].Length} instances");
}

// 可视化：pred.Draw 直达 C++ vis_iseg
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions { Threshold = 0.5 });
canvas.Save("iseg_vis.jpg");

// 多线程：Clone() 深拷贝独立实例
using var seg2 = seg.Clone();
```

## 5. FastSAM（`FastSamModel`）

FastSAM 结果与实例分割同构（`InstanceSegResult`）。`PredictWithPrompts` 在全量结果上按提示过滤实例，**不重跑网络**；提示为空等价全图 `Predict`。

```csharp
using ModelDeploy;
using ModelDeploy.Models;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var sam = new FastSamModel("fastsam-s.onnx", option);
sam.SetInputSize(1024, 1024);    // 默认 640x640；官方 FastSAM-s 常配 1024x1024
sam.SetConfThreshold(0.30);      // 默认 conf 0.30 / nms 0.5 / mask 0.5
sam.SetNmsThreshold(0.40);
sam.SetMaskThreshold(0.5);

using var img = VisionImage.Read("test.jpg");

// 全图（Everything）分割：Prediction<InstanceSegResult>，用法同 InstanceSegModel
using var pred = sam.Predict(img);

// 提示过滤：bboxes 为 [x,y,w,h,...]（原图像素）、points 为 [x,y,...]、
// labels 逐点（1=前景保留, 0=背景剔除）。返回 IReadOnlyList<InstanceSegResult>
var prompted = sam.PredictWithPrompts(img,
    new float[] { 100, 80, 220, 180 },
    new float[] { 150, 130 },
    new int[] { 1 });
Console.WriteLine($"prompted: {prompted.Count}");
```

## 6. 语义分割（`SemSegModel`）

语义分割模型（`yolo26n-sem` 等，cityscapes 19 类）。结果类型 `ModelDeploy.Results.SemSegResult`（属性 `Labels: byte[]`、`Width`/`Height`、`NumClasses: int`；`Labels` 为每像素类别索引 `[0, NumClasses)`，行主序）。该模型无阈值 setter，也无 `PredictBatch`（C API 对 `MD_MODEL_SEM_SEG` 无参数、结果为整图单值）。

```csharp
using ModelDeploy;
using ModelDeploy.Models;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var sem = new SemSegModel("yolo26n-sem.onnx", option);
sem.SetInputSize(640, 640);      // 输入尺寸可调；无其它参数（后处理 argmax）

using var img = VisionImage.Read("test.jpg");
using var pred = sem.Predict(img);
var r = pred[0];
// 掩码/标签逐像元读取：r.Labels[y * r.Width + x]
Console.WriteLine($"sem {r.Width}x{r.Height} classes={r.NumClasses} labels={r.Labels.Length}");

// 可视化：pred.Draw 直达 C++ vis_sem（cityscapes 调色板叠加）
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions { Alpha = 0.5 });
canvas.Save("sem_vis.jpg");
```

## 7. 深度估计（`DepthModel`）

深度估计模型（`yolo26n-depth` 等）。结果类型 `ModelDeploy.Results.DepthResult`（属性 `Depth: float[]`、`Width`/`Height`；每像素深度单位**米**，log 输出已 `exp` 还原，行主序）。该模型无阈值 setter，也无 `PredictBatch`（同上，整图单值结果）。

```csharp
using System;
using ModelDeploy;
using ModelDeploy.Models;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var dep = new DepthModel("yolo26n-depth.onnx", option);
dep.SetInputSize(640, 640);

using var img = VisionImage.Read("test.jpg");
using var pred = dep.Predict(img);
var r = pred[0];
// 深度逐像元读取：r.Depth[y * r.Width + x]（米）
float near = float.MaxValue, far = float.MinValue;
foreach (var d in r.Depth) { near = Math.Min(near, d); far = Math.Max(far, d); }
Console.WriteLine($"depth {r.Width}x{r.Height} range=[{near:F2}, {far:F2}] m");

// 可视化：pred.Draw 直达 C++ vis_depth（JET 伪彩）
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions());
canvas.Save("depth_vis.jpg");
```

## 8. 姿态与关键点族（`PoseModel` / `HandModel` / `VehicleKeypointModel` / `FaceLandmarkModel`）

四个模型类（`ModelDeploy.Models`，分别对应 `MD_MODEL_POSE` / `MD_MODEL_HAND` / `MD_MODEL_VEHICLE_KEYPOINT` / `MD_MODEL_FACE_LANDMARK`）结果类型统一为 `ModelDeploy.Results.PoseResult`（属性 `Box: RectF`、`Score: float`、`KeyPoints: Point3F[]`，`Point3F` 含 `X/Y/Z`，`Z` 为关键点置信度；无 `LabelId`）。`PoseModel`（COCO 17 点人体骨架）、`HandModel`（21 点手部）、`VehicleKeypointModel`（4 车轮关键点）支持 `SetConfThreshold` / `SetNmsThreshold` / `SetKeypointsNum`；`FaceLandmarkModel`（InsightFace 2d106 面部 106 点，`Z` 恒为 0）无阈值 setter（输入须为人脸裁剪图）。

```csharp
using System;
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var pose = new PoseModel("yolo11n-pose.onnx", option);
using var hand = new HandModel("hand.onnx", option);
using var vehicle = new VehicleKeypointModel("vehicle.onnx", option);
using var face = new FaceLandmarkModel("face_landmark.onnx", option);

// 预处理/后处理参数（FaceLandmarkModel 无参数 setter；HandModel 构造默认 21 点、
// VehicleKeypointModel 构造默认 4 点，均可 SetKeypointsNum 覆盖）
pose.SetInputSize(640, 640);      // letterbox 输入尺寸
pose.SetConfThreshold(0.30);      // 置信度阈值（默认 0.30）
pose.SetNmsThreshold(0.45);       // NMS IoU 阈值（默认 0.5）
pose.SetKeypointsNum(17);         // 关键点数（默认 17，须与模型输出一致）
hand.SetKeypointsNum(21);
vehicle.SetKeypointsNum(4);

using var img = VisionImage.Read("test.jpg");

// 单图推理：Predict 返回 Prediction<PoseResult>（foreach / 索引 / .Count）
using var pred = pose.Predict(img);
foreach (var r in pred) {
    Console.WriteLine($"{r.Score:F3} ({r.Box.X},{r.Box.Y},{r.Box.Width},{r.Box.Height}) kps={r.KeyPoints.Length}");
    foreach (var kp in r.KeyPoints) {
        Console.WriteLine($"  ({kp.X},{kp.Y},{kp.Z})");
    }
}
// 面部 Landmark：输入人脸裁剪图 -> 单元素结果（106 点 Z=0，Box 为整图、Score=1.0）
using var crop = VisionImage.Read("face_crop.jpg");
using var fpred = face.Predict(crop);

// 批量推理：PredictBatch 按图返回 IReadOnlyList<PoseResult[]>
using var img2 = VisionImage.Read("bus.jpg");
var batch = pose.PredictBatch(new[] { img, img2 });
for (int g = 0; g < batch.Count; g++) {
    Console.WriteLine($"image {g}: {batch[g].Length} persons");
}

// 可视化：pred.Draw 直达 C++ vis_pose（COCO 骨架连线）
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions { Threshold = 0.30, Alpha = 0.3 });
canvas.Save("pose_vis.jpg");

// 多线程：Clone() 深拷贝独立实例
using var pose2 = pose.Clone();
```

## 9. OBB（旋转框检测）（`ObbModel`）

旋转框检测模型类 `ModelDeploy.Models.ObbModel`（对应 `MD_MODEL_OBB`）。结果类型 `ModelDeploy.Results.ObbResult`（属性 `Box: RotatedRectF{Cx, Cy, Width, Height, Angle}`、`LabelId: int`、`Score: float`）；`Cx/Cy` 为旋转框中心、`Angle` 为弧度角，坐标均为原图像素。

```csharp
using System;
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var obb = new ObbModel("yolo11n-obb.onnx", option);

// 预处理/后处理参数
obb.SetInputSize(1024, 1024);     // letterbox 输入尺寸（默认 1024x1024）
obb.SetConfThreshold(0.25);       // 置信度阈值（默认 0.25）
obb.SetNmsThreshold(0.45);        // NMS IoU 阈值（默认 0.5）

using var img = VisionImage.Read("test.jpg");

// 单图推理：Prediction<ObbResult>（foreach / 索引 / .Count）
using var pred = obb.Predict(img);
foreach (var r in pred) {
    Console.WriteLine($"{r.LabelId} {r.Score:F3} " +
                      $"(xc={r.Box.Cx}, yc={r.Box.Cy}, w={r.Box.Width}, h={r.Box.Height}, angle={r.Box.Angle})");
}

// 可视化：pred.Draw 直达 C++ vis_obb
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions { Threshold = 0.5, FontSize = 14, Alpha = 0.3 });
canvas.Save("obb_vis.jpg");

// 批量推理：PredictBatch 按图返回 IReadOnlyList<ObbResult[]>
using var img2 = VisionImage.Read("bus.jpg");
var batch = obb.PredictBatch(new[] { img, img2 });
for (int g = 0; g < batch.Count; g++) {
    Console.WriteLine($"image {g}: {batch[g].Length} rotated boxes");
}

// 多线程：Clone() 深拷贝独立实例
using var obb2 = obb.Clone();
```

## 10. 图像分类（`ClassificationModel`）

分类模型类 `ModelDeploy.Models.ClassificationModel`（对应 `MD_MODEL_CLASSIFICATION`）。结果类型 `ModelDeploy.Results.ClassificationResult`（属性 `LabelId: int`、`Score: float`，`Prediction` 中 Top-K 逐项）。参数 setter：`SetTopK`（默认 1）、`SetMultiLabel`（默认 false）。

```csharp
using System;
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

using var cls = new ClassificationModel("yolo11n-cls.onnx", option);

// 预处理/后处理参数
cls.SetInputSize(224, 224);       // 输入尺寸（默认 224x224）
cls.SetTopK(5);                   // Top-K 输出个数（默认 1）
cls.SetMultiLabel(false);         // 多标签模式（默认 false）

using var img = VisionImage.Read("test.jpg");

// 单图推理：Prediction<ClassificationResult>（LabelId 与 Score 逐位配对）
using var pred = cls.Predict(img);
foreach (var r in pred) {
    Console.WriteLine($"{r.LabelId} {r.Score:F3}");
}

// 可视化：pred.Draw 直达 C++ vis_cls（Threshold 即分数阈值）
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions { Threshold = 0.35, FontSize = 14, Alpha = 0.3 });
canvas.Save("cls_vis.jpg");

// 批量推理：PredictBatch 按图返回 IReadOnlyList<ClassificationResult[]>
using var img2 = VisionImage.Read("bus.jpg");
var batch = cls.PredictBatch(new[] { img, img2 });
for (int g = 0; g < batch.Count; g++) {
    Console.WriteLine($"image {g}: {batch[g].Length} labels");
}

// 多线程：Clone() 深拷贝独立实例
using var cls2 = cls.Clone();
```

## 11. OCR（`OcrModel` + 子模型）

主流水线类 `ModelDeploy.Models.OcrModel`（对应 `MD_MODEL_OCR`）：`modelPath` 用 `|` 串联 **det/cls/rec/dict 四段**（`"det.onnx|cls.onnx|rec.onnx|dict.txt"`）。`Predict` 返回 `Prediction<OcrResult>`，每行文本一项：`Quad`（4 点共 8 个 int，原图像素）、`Text`、`Score`（识别得分）、`ClsLabel`/`ClsScore`（方向分类，逐行配对）。注意：单图 `Predict` 的 `Prediction` 因底层单值包装**仅含首行**，完整逐行结果可经 `pred.Handle` 配合原生 `md_result_ocr` / `md_result_ocr_cls` 读取；OCR 模型类亦未提供 `PredictBatch`。

```csharp
using System;
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

// 1. 构造：det|cls|rec|dict 四段路径（'|' 分隔）
using var ocr = new OcrModel("det.onnx|cls.onnx|rec.onnx|dict.txt", option);

// 2. 参数设置（括号内为默认值）
ocr.SetDetDbThresh(0.3);          // DB 二值化阈值（默认 0.3）
ocr.SetDetDbBoxThresh(0.6);       // 框置信度阈值（默认 0.6）
ocr.SetDetDbUnclipRatio(1.5);     // 扩框比例（默认 1.5）
ocr.SetDetDbScoreMode("slow");    // 框得分模式（默认 "slow"）
ocr.SetUseDilation(false);        // 是否膨胀（默认 false）
ocr.SetClsThresh(0.9);            // 方向分类阈值（默认 0.9）
ocr.SetMaxSideLen(960);           // 检测最长边（默认 960）
ocr.SetClsBatchSize(6);           // 方向分类子模型 batch（默认 6）
ocr.SetRecBatchSize(8);           // 识别子模型 batch（默认 8）
ocr.SetRecImageShape(3, 48, 320); // 识别输入形状（默认 3x48x320）

// 3. 单图推理：Prediction<OcrResult>（Text/Quad/Score/ClsLabel/ClsScore 逐行配对）
using var img = VisionImage.Read("test.jpg");
using var pred = ocr.Predict(img);
foreach (var r in pred) {
    Console.WriteLine($"{r.Text} {r.Score:F3} cls={r.ClsLabel} box=({string.Join(',', r.Quad)})");
}

// 4. 可视化：pred.Draw 直达 C++ vis_ocr
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions { FontPath = "msyh.ttc", FontSize = 14, Alpha = 0.3 });
canvas.Save("ocr_vis.jpg");

// 5. 子模型独立使用（也可不经 OcrModel 单独构造；Predict 均返回 Prediction<OcrResult>）
using var db  = new DbDetectorModel("det.onnx", option);            // Quad（文本框）
using var rec = new RecognizerModel("rec.onnx|dict.txt", option);   // Text/Score（路径 '|' 两段）
using var clr = new OcrClassifierModel("cls.onnx", option);         // 需 SetClsThresh 时可设
using var predDet = db.Predict(img);
using var predRec = rec.Predict(img);
using var predCls = clr.Predict(img);

// 6. 多线程：Clone() 深拷贝独立实例
using var ocr2 = ocr.Clone();
```

## 12. OCR 进阶（版面 / 表格 / 公式 / 文档转 Markdown）

C# 仅绑定其中**公式识别**：`ModelDeploy.Models.FormulaRecognizerModel`，`Predict` 直接返回 LaTeX `string`。
**版面 `StructureV2Layout` / 表格 `StructureV2Table` / `PPStructureV2Table` / 文档转 Markdown `DocToMarkdown` 本语言未绑定**（C API 无对应 kind），如需请用 C++ / Python 绑定。

`FormulaRecognizerModel` 的 `modelPath` 用 `|` 串联 **model[|dict]** 两段：`"formula.onnx|dict.txt"`（dict 可省略为 `"formula.onnx"`）。

```csharp
using ModelDeploy;
using ModelDeploy.Models;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

// 公式识别：model|dict 两段路径（'|' 分隔，dict 可省）
using var formula = new FormulaRecognizerModel("formula.onnx|dict.txt", option);

// 单图推理：返回 LaTeX 字符串
using var img = VisionImage.Read("equation.jpg");
string latex = formula.Predict(img);
Console.WriteLine(latex);

// 多线程：Clone() 深拷贝独立实例
using var formula2 = formula.Clone();
```

## 13. 人脸（`FaceDetModel` / `FaceRecModel` / `FaceAgeModel` / `FaceGenderModel` / `FaceRecognizerPipelineModel`）

人脸模型类位于 `ModelDeploy.Models`，结果类型不同：`FaceDetModel` 返回 `FaceDetResult`（`Box: RectF` + `Score` + `KeyPoints: PointF[]`，5 关键点）、`FaceRecModel`/`FaceRecognizerPipelineModel` 返回 `FaceRecResult`（`Embedding: float[]`，512 维）、`FaceAgeModel`/`FaceGenderModel` 直接返回 `int`（gender `0`=女 / `1`=男）。**人脸防伪（SeetaFaceAs 一/二阶段与 AsPipeline）本语言未绑定**（C API 未暴露 `FACE_AS_PIPELINE` 于 C# 模型类），如需请用 C++ / Python。

```csharp
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

// 1. 人脸检测：FaceDetModel -> Prediction<FaceDetResult>（框 + 5 关键点）
using var det = new FaceDetModel("scrfd.onnx", option);
det.SetInputSize(640, 640);
det.SetConfThreshold(0.30);        // 置信度阈值（默认 0.25）
det.SetNmsThreshold(0.45);         // NMS IoU 阈值（默认 0.5）
det.SetLandmarksPerFace(5);        // 每人脸关键点（默认 5）

using var img = VisionImage.Read("test.jpg");
using var detPred = det.Predict(img);
foreach (var r in detPred) {
    Console.WriteLine($"{r.Score:F3} ({r.Box.X},{r.Box.Y},{r.Box.Width},{r.Box.Height}) kps={r.KeyPoints.Length}");
    foreach (var kp in r.KeyPoints) Console.WriteLine($"  ({kp.X},{kp.Y})");
}

// 2. 年龄 / 性别：FaceAgeModel / FaceGenderModel -> int
using var age = new FaceAgeModel("age.onnx", option);
using var gender = new FaceGenderModel("gender.onnx", option);
var a = age.Predict(img);
var g = gender.Predict(img);
Console.WriteLine($"age={a} gender={g} ({(g == 0 ? "女" : "男")})");

// 3. 人脸识别（特征）：FaceRecModel -> FaceRecResult.Embedding（512 维）
using var rec = new FaceRecModel("rec.onnx", option);
var emb = rec.Predict(img);
Console.WriteLine($"embedding dim={emb.Embedding.Length}");

// 4. 识别流水线：FaceRecognizerPipelineModel（检测 + 特征一体化）
//    modelPath 用 '|' 两段，或直接用 (detModelPath, recModelPath, opt) 构造
using var pipe = new FaceRecognizerPipelineModel("det.onnx", "rec.onnx", option);
pipe.SetConfThreshold(0.30);
pipe.SetLandmarksPerFace(5);
using var pipePred = pipe.Predict(img);      // Prediction<FaceRecResult>
foreach (var r in pipePred) Console.WriteLine($"embedding dim={r.Embedding.Length}");

// 批量推理：det.PredictBatch(new[] { img, img2 }) 按图返回 IReadOnlyList<FaceDetResult[]>
// 多线程：Clone() 深拷贝独立实例
using var det2 = det.Clone();
```

## 14. InsightFace 全流程（`InsightFaceModel` + `InsightFaceDetModel`）

InsightFace Buffalo 全家桶（det + 2d106 + 3d68 + recognition）经一次 `Predict` 综合输出检测框/5 关键点/姿态/特征/性别年龄。

```csharp
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

// 1. 全流程分析：InsightFaceModel。modelPath 用 '|' 串联最多 5 段子模型路径
//    （det_10g.onnx|w600k_r50.onnx|2d106det.onnx|1k3d68.onnx|genderage.onnx，第 5 段可省）
using var ia = new InsightFaceModel(
    "det_10g.onnx|w600k_r50.onnx|2d106det.onnx|1k3d68.onnx|genderage.onnx", option);
ia.SetDetThresh(0.5);            // 检测阈值（默认 0.5）

using var img = VisionImage.Read("test.jpg");
using var pred = ia.Predict(img);     // Prediction<InsightFaceResult>
foreach (var r in pred)
{
    Console.WriteLine($"{r.Score:F3} ({r.Box.X},{r.Box.Y},{r.Box.Width},{r.Box.Height}) " +
                      $"kps={r.KeyPoints.Length} emb={r.Embedding.Length} pose={r.Pose.Length} " +
                      $"gender={r.Gender} age={r.Age}");
    // r.Box / r.Score：检测框 + 置信度
    // r.KeyPoints：5 关键点（PointF[]）；r.Embedding：512 维特征（float[]）
    // r.Pose：3 姿态角（float[]）；r.Gender / r.Age：0/1 整数、年龄（未启用 genderage 时为 -1）
    // 注意：C# 未暴露 106/68 关键点（C++/Python 才有）
}

// 批量：ia.PredictBatch(new[] { img, img2 }) -> IReadOnlyList<InsightFaceResult[]>
// 多线程：ia.Clone()；可视化：pred.Draw(canvas, new DrawOptions { Alpha = 0.3 })

// 2. 子模型：InsightFaceDetModel（仅检测，det_10g.onnx -> Prediction<FaceDetResult>）
using var det = new InsightFaceDetModel("det_10g.onnx", option);
using var detPred = det.Predict(img);
foreach (var f in detPred)
{
    Console.WriteLine($"{f.Score:F3} ({f.Box.X},{f.Box.Y},{f.Box.Width},{f.Box.Height}) " +
                      $"kps={f.KeyPoints.Length}");
}
```

## 15. 车牌 LPR（`LprModel` / `LprDetectionModel` / `LprRecognizerModel`）

车牌模型类位于 `ModelDeploy.Models`：主流水线 `LprModel`（对应 `MD_MODEL_LPR_PIPELINE`）、子模型 `LprDetectionModel`（`MD_MODEL_LPR_DET`，仅框 + 置信度 → `LprDetResult`）与 `LprRecognizerModel`（`MD_MODEL_LPR_REC`，字符/颜色 → `LprResult`）。主流水线/识别结果类型 `ModelDeploy.Results.LprResult`（属性 `Box: RectF`、`Plate: string`、`Color: string`、`Score: float`、`KeyPoints: PointF[]`（车牌 4 角点））。

```csharp
using System;
using System.Collections.Generic;
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

// 1. 主流水线：两参重载 (det, rec)，或单串 "det.onnx|rec.onnx"
using var lpr = new LprModel("det.onnx", "rec.onnx", option);

using var img = VisionImage.Read("test.jpg");

// 2. 单图推理：Prediction<LprResult>
using var pred = lpr.Predict(img);
foreach (var r in pred)
{
    Console.WriteLine($"'{r.Plate}' {r.Color} {r.Score:F3} " +
                      $"({r.Box.X},{r.Box.Y},{r.Box.Width},{r.Box.Height}) kps={r.KeyPoints.Length}");
    foreach (var kp in r.KeyPoints) Console.WriteLine($"  ({kp.X},{kp.Y})");
}

// 3. 可视化：pred.Draw 直达 C++ vis_lpr（车牌框 + 字符 + 4 角点连线）
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions { FontPath = "msyh.ttc", FontSize = 14, Alpha = 0.3 });
canvas.Save("lpr_vis.jpg");

// 4. 批量推理：PredictBatch 按图返回 IReadOnlyList<LprResult[]>
using var img2 = VisionImage.Read("bus.jpg");
var batch = lpr.PredictBatch(new[] { img, img2 });
for (int g = 0; g < batch.Count; g++)
    Console.WriteLine($"image {g}: {batch[g].Length} plates");

// 5. 多线程：Clone() 深拷贝独立实例
using var lpr2 = lpr.Clone();

// 6. 子模型独立使用
using var det = new LprDetectionModel("det.onnx", option);
det.SetInputSize(640, 640);
det.SetConfThreshold(0.25);    // 置信度阈值（默认 0.25）
det.SetNmsThreshold(0.45);     // NMS IoU 阈值（默认 0.5）
det.SetLandmarksPerCard(4);    // 每车牌角点数（默认 4）
using var detPred = det.Predict(img);        // Prediction<LprDetResult>（仅框 + 置信度）
foreach (var r in detPred)
    Console.WriteLine($"{r.Score:F3} ({r.Box.X},{r.Box.Y},{r.Box.Width},{r.Box.Height})");

using var rec = new LprRecognizerModel("rec.onnx", option);
using var crop = VisionImage.Read("plate_crop.jpg");
using var recPred = rec.Predict(crop);       // Prediction<LprResult>（输入车牌裁剪图）
foreach (var r in recPred)
    Console.WriteLine($"'{r.Plate}' {r.Color} {r.Score:F3}");
```

## 16. 行人属性（`PedestrianAttributeModel`）+ 行人 ReID（`ReIdModel`）

行人属性模型类 `ModelDeploy.Models.PedestrianAttributeModel`（对应 `MD_MODEL_PED_ATTR`）。构造可用两参重载 `(detModelPath, clsModelPath)`（内部拼成 C API 的 `"det|cls"`）或单串 `"det.onnx|cls.onnx"`。结果类型 `ModelDeploy.Results.AttributeResult`（属性 `Box: RectF`、`BoxLabelId: int`、`BoxScore: float`、`AttrScores: float[]`）；`Predict` 返回 **`Prediction<AttributeResult>`**。参数 setter：`SetInputSize`（检测子模型输入尺寸）、`SetClsInputSize`（分类子模型输入尺寸）、`SetClsBatchSize`（>0 固定 / -1 自动）、`SetDetThreshold`。

行人 ReID 模型类 `ModelDeploy.Models.ReIdModel`（对应 `MD_MODEL_REID`），`Predict` 返回**裸 `ReIdResult`**（属性 `Embedding: float[]`，L2 归一化 512 维），**不是** `Prediction<ReIdResult>`。**C# 未绑定 ReIdGallery**（C API 无对应句柄），行人库检索请用 C++/Python。

```csharp
using System;
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

// 1. 行人属性：两参重载 (det, cls) 或单串 "det.onnx|cls.onnx"
using var attr = new PedestrianAttributeModel("det.onnx", "cls.onnx", option);
attr.SetInputSize(1280, 1280);        // 检测子模型输入尺寸
attr.SetClsInputSize(192, 256);       // 分类子模型输入尺寸
attr.SetClsBatchSize(8);              // 分类子模型 batch（>0 固定 / -1 自动；Sophgo batch=1 静态）
attr.SetDetThreshold(0.5);            // 检测阈值（默认 0.5）

using var img = VisionImage.Read("test.jpg");
using var pred = attr.Predict(img);   // Prediction<AttributeResult>
foreach (var r in pred)
    Console.WriteLine($"{r.Box} label={r.BoxLabelId} score={r.BoxScore:F3} attrs=[{string.Join(",", r.AttrScores)}]");

// 可视化：pred.Draw 直达 C++ vis_attr（框 + 属性文本）
using var canvas = img.Clone();
pred.Draw(canvas, new DrawOptions { FontPath = "msyh.ttc", FontSize = 14, Alpha = 0.3 });
canvas.Save("attr_vis.jpg");

// 批量：attr.PredictBatch(new[] { img, img2 }) -> IReadOnlyList<AttributeResult[]>
// 多线程：using var attr2 = attr.Clone();

// 2. 行人 ReID：ReIdModel 单模型，Predict 返回裸 ReIdResult（非 Prediction）
using var reid = new ReIdModel("osnet.onnx", option);
using var crop = VisionImage.Read("person_crop.jpg");
ReIdResult r = reid.Predict(crop);    // 裸 ReIdResult，非 Prediction
Console.WriteLine($"embedding dim={r.Embedding.Length}");

// 注意：C# 未绑定 ReIdGallery；行人库检索请用 C++/Python
```

## 17. 条码 / 二维码（`BarcodeDetector`）

命名空间 `ModelDeploy`，`BarcodeDetector` 封装 C API `md_barcode_*`，是**纯 CV**（基于 ZXing）、无模型依赖，CPU 上即可解码条码 / 二维码。默认解码全部格式，可用 `SetFormats` 限定 `FMT_*` 位或子集。

```csharp
using System;
using ModelDeploy;

// 1. 构造（无模型路径，无需 RuntimeOption）
var det = new BarcodeDetector();
det.SetFormats((1u << 0) | (1u << 4));   // FMT_QR_CODE | FMT_EAN_13（位 0 和位 4）

// 2. 检测：Detect(VisionImage) -> BarcodeResult[]
using var img = VisionImage.Read("qrcode.jpg");
foreach (var r in det.Detect(img))
{
    // r.Format: string（如 "QR_CODE"/"EAN_13"）；r.Text: 解码文本/URL
    // r.Score: 可信度 [0,1]；r.IsQr: 是否二维码；r.Quad: PointF[] 长度 4（角点，左上起顺时针）
    Console.WriteLine($"{r.Format} [{r.Text}] score={r.Score:F3} is_qr={r.IsQr}");
    foreach (var p in r.Quad) Console.WriteLine($"  corner=({p.X},{p.Y})");
}
```

## 18. 多目标跟踪（`ModelDeploy.Tracking.Tracker`）

命名空间 `ModelDeploy.Tracking`，`Tracker` 封装 C API `md_tracker_*`，构造时选 `TrackerKind`（`ByteTrack` / `BotSort` / `StrongSort`），**纯 CPU**、无模型依赖，跟踪 ID 跨帧稳定，`Reset()` 归零。因 C API 用命名参数，C# 用 `SetParam(name, value)` 设参（支持 `track_thresh` / `high_thresh` / `low_thresh` / `max_age` / `min_hits` / `iou_threshold` / `match_thresh` / `ema_alpha` / `fuse_score_weight` / `appearance_priority` / `with_cmc`，按 kind 忽略不适用项）；`Update(boxes, scores, labelIds)` 返回 `TrackItem[]`（`TrackId` 跨帧关联同一目标；`State` 为 `MDTrackState`：New=0 / Tracked=1 / Lost=2 / Removed=3）。

```csharp
using System;
using ModelDeploy;
using ModelDeploy.Tracking;
using ModelDeploy.Models;
using ModelDeploy.Results;

var option = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

// 1. 构造与命名参数
var tracker = new Tracker(TrackerKind.ByteTrack);
tracker.SetParam("track_thresh", 0.5);
tracker.SetParam("max_age", 30.0);
tracker.SetParam("iou_threshold", 0.3);
// BoT-SORT / StrongSORT 另可：SetParam("match_thresh", 0.8)、SetParam("ema_alpha", 0.9) 等

using var det = new DetectionModel("yolo11n.onnx", option);
for (int f = 0; f < 100; f++)   // 假定逐帧读取视频，这里用帧索引示意
{
    using var frame = VisionImage.Read("frame_%03d.jpg");
    var rs = det.Predict(frame);   // IReadOnlyList<DetectionResult>

    // 2. 每帧：检测框 -> RectF[]/scores/labelIds
    int n = rs.Count;
    var boxes = new RectF[n];
    var scores = new float[n];
    var labels = new int[n];
    for (int i = 0; i < n; i++)
    {
        boxes[i] = rs[i].Box;
        scores[i] = rs[i].Score;
        labels[i] = rs[i].LabelId;
    }

    // 3. 推进一帧：Update(boxes, scores, labelIds) -> TrackItem[]
    foreach (var t in tracker.Update(boxes, scores, labels))
        // t.TrackId: 跨帧稳定 ID；t.State: New/Tracked/Lost/Removed；其余：X/Y/Width/Height/Score
        Console.WriteLine($"id={t.TrackId} state={t.State} score={t.Score:F3}");
}
tracker.Reset();   // 清空内部状态，ID 重新从 0 计
```

## 19. CV 解决方案 + 工具（`ModelDeploy.Solutions`）

`ModelDeploy.Solutions` 命名空间封装 C API `md_solution_*`，提供解决方案类与静态工具类 `Tool`。所有解决方案均 `IDisposable`（`using` 自动释放底层句柄），框为扁平 `float[]`（`[x,y,w,h,...]`）、多边形为扁平 `[x0,y0,x1,y1,...]`。

**已绑定清单**：`ObjectCounter` / `Heatmap` / `RegionCounter` / `QueueManager` / `TrackZone` + `Tool.Iou`。
**未绑定**：`SpeedEstimator`（测速）、`ParkingManager`（停车）、`FallDetector`、`WorkoutMonitor`、`DistanceEstimator` 本语言**无类封装**（C API 枚举可创建但无对应 C# 查询接口），如需请用 C++ / Python。

```csharp
using System;
using ModelDeploy;
using ModelDeploy.Solutions;

// 1. 人流统计：跨线进出 + 区域/类别统计
using var cnt = new ObjectCounter();
cnt.SetLine(0, 0, 100, 100);                    // 计数线两点
cnt.Update(boxes, labelIds, trackIds);          // boxes=[x,y,w,h,...]
var (inCnt, outCnt) = cnt.HLine();

// 2. 热力图：SetSize 低分辨率栅格，Update 累加，Peak 峰值
using var hm = new Heatmap();
hm.SetSize(320, 240);
hm.Update(boxes, frameW: 1920, frameH: 1080);
var peak = hm.Peak();                           // (X, Y)

// 3. 多区域逐帧计数：AddRegion 命名区域 -> Count(name)
using var rc = new RegionCounter();
rc.AddRegion("doorA", new[] { 0f, 0, 80, 0, 80, 240, 0, 240 });   // 扁平多边形
rc.Update(boxes, ids, labels);
Console.WriteLine(rc.Count("doorA"));

// 4. 排队：单区域当前帧排队长度
using var qm = new QueueManager();
qm.SetRegion(new[] { 100f, 0, 160, 0, 160, 240, 100, 240 });
qm.Update(boxes, ids, labels);
Console.WriteLine(qm.Count());

// 5. 追踪区域：只保留区域内目标并计数
using var tz = new TrackZone();
tz.SetRegion(new[] { 100f, 0, 160, 0, 160, 240, 100, 240 });
tz.Update(boxes, ids, labels);
Console.WriteLine(tz.Count());

// 6. 工具：两个矩形 (x,y,w,h) 的 IoU（对应 C++ vision::tool::iou）
float iou = Tool.Iou(0, 0, 100, 100, 20, 20, 100, 100);
Console.WriteLine(iou);
```

> C API `MDSolutionKind` 另有 `SPEED`/`DISTANCE`/`WORKOUT`/`PARKING`/`FALL_DETECT` 五个枚举项，但 C# 与 C API 均无对应逐帧查询接口（仅 `md_solution_create`/`destroy`），故不提供类封装；底层 C++ 实现见 [solutions.md](../solutions.md)。

## 20. 音频模型（SenseVoice / Kokoro / SpeakerVerify）

`ModelDeploy.Models` 命名空间提供 ASR `SenseVoiceModel`、TTS `KokoroModel` 与声纹 `SpeakerVerifyModel`；`SpeakerGallery`（注册/比对库）**未绑定**，用 `ModelDeploy.Audio.SpeakerSearch` 替代（封装 C API `md_audio_speaker_search_*`）。模型路径为 **`|` 拼接的多文件路径**。

```csharp
using System;
using ModelDeploy;
using ModelDeploy.Models;

var opt = new RuntimeOption().UseOrt().SetDevice(Device.CPU);

// ── ASR（SenseVoice，16k）─ 路径格式: model.onnx|tokens.txt
var asr = new SenseVoiceModel("sense_voice.onnx|tokens.txt", opt);
var r1  = asr.Predict(wav16k, 16000);            // AsrResult.Text：纯净文本
var r2  = asr.PredictWav("in.wav");
var rs  = asr.PredictStructured(wav16k, 16000);  // AsrResult{Text,Language,Emotion,Event,Task,Itn,NoSpeech}

// ── TTS（Kokoro，24k）─ 路径格式: model.onnx|tokens.txt|lex_en.txt|lex_zh.txt|voices.bin|jieba_dir|norm_dir
var kokoro = new KokoroModel("kokoro.onnx|tokens.txt|lex_en.txt|lex_zh.txt|voices.bin|jieba_dir|norm_dir", opt);
var tts  = kokoro.Predict("你好，世界。", "zf_001");   // TtsResult{ Audio(float[]), SampleRate }
kokoro.SaveWav(tts, "kokoro.wav");

// ── 声纹（SpeakerVerify，提取 192-d embedding）与声纹库 ─
var spv = new SpeakerVerifyModel("ecapa.onnx", opt);
float[] emb = spv.Predict(wav16k);

// SpeakerGallery 未绑定 → 用 SpeakerSearch（纯内存声纹库）
using var gal = new ModelDeploy.Audio.SpeakerSearch();
gal.Enroll("alice", emb);
string label = gal.Match(emb);                  // 余弦 top-1，返回最相似 label
```

> `SenseVoiceModel` 另提供 `PredictWavStructured`/`PredictStructured`（结构化 `AsrResult`，字段 `Text/Language/Emotion/Event/Task/Itn/NoSpeech`）；`KokoroModel.Predict` 与 `SpeakerVerifyModel.Predict` 的 PCM 输入为 16kHz 单声道（约 1s 足够），Kokoro 输出 24kHz。`SpeakerSearch` 仅暴露 `Enroll`/`Match`（top-1），如需 top-k 与多说话人请用 C++ / Python。

## 21. 音频解决方案 + 工具（`Audio.SpeakerSearch` / `Audio.Tools`）

`ModelDeploy.Audio` 命名空间提供说话人检索 `SpeakerSearch`（封装 C API `md_audio_speaker_search_*`）与静态工具类 `Tools`（封装 `md_audio_resample`）。**TTS 批处理（`TTSBatcher`）与逆文本归一化（`ITN`：`InverseTextNormalizer`/`ItnEngine`）本语言未绑定**（C API 未暴露其 enqueue/dequeue 与 normalize 接口），如需请用 C++ / Python。

```csharp
using ModelDeploy;
using ModelDeploy.Audio;

// 1. 说话人检索：SpeakerSearch（纯内存声纹库，对应 C API md_audio_speaker_search_*）
using var ss = new SpeakerSearch();
ss.Enroll("alice", emb);             // label + embedding（float[]）
string label = ss.Match(emb);        // 余弦 top-1，返回最相似 label

// 2. 工具：重采样（Audio.Tools.Resample，封装 C API md_audio_resample）
float[] out16k = Audio.Tools.Resample(pcm48k, 48000, 16000);
```

> `SpeakerSearch` 仅暴露 `Enroll`/`Match`（top-1，见 §20）；`Audio.Tools.Resample(float[], inSr, outSr)` 在任意采样率间转换 float PCM。C# 无声纹 `SpeakerGallery`/`TTSBatcher`/`ITN` 封装。

## 运行示例

```bash
cd csharp
dotnet build ModelDeployExample/ModelDeployExample.csproj -c Debug
cd ModelDeployExample/bin/Debug/net9.0
./ModelDeployExample.exe
```

`Program.cs` 的 `Main` 依次运行：检测 → 图像工具 → 分类 → 姿态 → OCR → InsightFace → ASR(SenseVoice) → TTS(Kokoro)，各结果打印到控制台，检测/图像落盘，TTS 落盘 `output.wav`。

## 主要项目

- `ModelDeploy` — C# 绑定库
- `ModelDeployExample` — 示例
- `ModelDeployUnitTest` — 单元测试

## 图像原始字节（`VisionImage`）

`VisionImage` 提供原生格式的宿主字节读取（`Type`/`Width`/`Height` 见 `VisionImage` 属性）：

```csharp
byte[] raw = img.ToNativeBytes();     // 整幅原生连续字节；GPU 设备帧自动回读主机
byte[] y   = img.GetPlaneBytes(0);    // 第 0 平面（NV12 的 Y）
byte[] uv  = img.GetPlaneBytes(1);    // 第 1 平面（NV12 的 UV）
```

- `ToNativeBytes()` 布局随 `Type` 而定：BGR24=`Width*Height*3`；NV12=`w*h + w*h/2`；I420=`w*h + w*h/4 + w*h/4`。
- 平面越界抛 `InvalidOperationException`；不支持的类型/设备（如 TPU）抛 `NotSupportedException`。
- 两方法立即将原生缓冲拷贝为托管 `byte[]`（指针在下次调用即失效）。
- 与既有 `ToByteArray()`（编码为 **BMP** 文件字节，含 54 字节头）不同，上面拿到的是**原始像素**，可直接交给其它编解码接口。

## 视频编解码（`Video`）

`ModelDeploy` 命名空间下的视频封装等价于 C++ `modeldeploy::video`（解码+编码全功能，经 C API 承载），类均实现 `IDisposable`，析构自动释放底层句柄。

| 类型 | 说明 |
|------|------|
| `VideoConfig` | 解码/编码配置（`set` 属性，可用对象初始化器） |
| `VideoCapabilities` | 能力探测（FFmpeg/GStreamer/硬解硬编列表） |
| `VideoDecoder` | 解码器：同步 `ReadFrame` + 异步回调 + 状态/统计 |
| `VideoEncoder` | 编码器：`Encode` / `EncodeAsync` / `StartAsync` |
| `VideoStats` | 编解码统计 |

枚举：`VideoCodecBackend{Auto,FFmpeg,GStreamer}`、`VideoHwAccel{Auto,None,Cuda,Vaapi,Sophgo}`、`VideoBackpressure{Block,Drop,OverwriteOldest}`、`VideoState{Idle,Opening,Running,Reconnecting,Eof,Error,Closed}`。

```csharp
using ModelDeploy;

// 能力探测
var cap = new VideoCapabilities();
Console.WriteLine($"ffmpeg={cap.FfmpegAvailable} hw_encode={string.Join(",", cap.HwEncoders)}");

// 配置 + 解码
using var dec = new VideoDecoder(new VideoConfig {
    Backend = VideoCodecBackend.FFmpeg,
    HwAccel = VideoHwAccel.Cuda,
    DeviceOnly = false,               // true = GPU 设备内存直通
});
dec.Open("demo.mp4");                 // 或 rtsp://...
var (image, pts) = dec.ReadFrame();   // 返回 (VisionImage NV12帧, 时间戳毫秒)；空帧表示 EOF
while (image != null) {
    Console.WriteLine($"{pts}ms {image.Width}x{image.Height}");
    (image, pts) = dec.ReadFrame();
}
var st = dec.Stats;                   // VideoStats
```

**异步解码（回调推送）**：`dec.SetCallback((image, pts) => { ... })` 注册交付回调（帧为自有 `VisionImage`，用后自动释放），再 `dec.Start()` / `dec.Stop()`。

**编码**：
```csharp
using var enc = new VideoEncoder(new VideoConfig {
    Codec = "libx264", BitrateKbps = 2000, Gop = 30, Fps = 25,
});
enc.Open("out.mp4", 1280, 720, 25);
enc.Encode(img, ptsMs);               // img 生命周期须覆盖本次调用
enc.Close();                          // 必须 Close，mp4 尾部索引(moov)在此写盘
```

设备显存（`DeviceOnly` 解码 / `GpuDirectInput` 编码）零拷贝直通。详见 [视频接口总览](../video/api.md)。
