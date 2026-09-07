# Rust 绑定

Rust 通过 FFI 封装 C API。目录 `rust/modeldeploy/`：

## 1. 引入

```toml
[dependencies]
modeldeploy = { path = "path/to/ModelDeploy/rust/modeldeploy" }
```

常用类型在 crate 顶层 re-export：`RuntimeOption`、`Image`、各模型类（如 `UltralyticsDet`）与 `MdError`；设备枚举 `MDDevice` 位于 `ffi` 模块（`modeldeploy::ffi::MDDevice`）。

## 2. RuntimeOption（后端/设备/精度）

```rust
use modeldeploy::{Image, RuntimeOption, UltralyticsDet};
use modeldeploy::ffi::MDDevice;

fn main() -> Result<(), modeldeploy::MdError> {
    let mut opt = RuntimeOption::new()?;          // 返回 Result
    opt.use_ort().set_device(MDDevice::CPU, 0)?;  // 设备
    opt.set_cpu_threads(4)?;                      // CPU 线程数

    // 其它后端（&mut self -> &mut Self，可链式）：
    // opt.use_mnn() / opt.use_trt() / opt.use_sophgo() / opt.use_ncnn()
    // 精度：opt.set_fp16(true)?
    // GPU：opt.use_ort().set_device(MDDevice::GPU, 0)?
    // OPENCL/VULKAN 需显式 MNN 后端，否则 fail-closed：
    // opt.use_mnn().set_device(MDDevice::OPENCL, 0)?;

    let model = UltralyticsDet::new("yolo11n.onnx", &opt)?;
    let img = Image::read("test.jpg")?;
    let dets = model.predict(&img)?;
    Ok(())
}
```

- `RuntimeOption::new()` 返回 `Result`；setter 返回 `Result<&mut Self, MdError>`，用 `?` 链式调用。
- 后端方法：`use_ort` / `use_mnn` / `use_trt` / `use_sophgo` / `use_ncnn`。
- 设备：`set_device(MDDevice::CPU, 0)?`；线程数 `set_cpu_threads(n)?`；精度 `set_fp16(bool)?`。

## 3. 目标检测（`UltralyticsDet`）

检测模型 `UltralyticsDet::new(path, &opt)?` 加载。结果类型 `Detection`（字段 `rect: Rect{x, y, width, height}`、`label_id: i32`、`score: f32`）；`predict` 返回 `Vec<Detection>`，`predict_batch` 返回 `Vec<Vec<Detection>>`（按图分组）。

```rust
use modeldeploy::{DrawOptions, Image, RuntimeOption, UltralyticsDet};
use modeldeploy::ffi::MDDevice;

fn main() -> Result<(), modeldeploy::MdError> {
    // 1. 运行时选项（详见上节）
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(MDDevice::CPU, 0)?.set_cpu_threads(4)?;

    // 2. 构造模型
    let model = UltralyticsDet::new("yolo11n.onnx", &opt)?;

    // 3. 预处理/后处理参数（均为 Result<(), MdError>）
    model.set_input_size(640, 640)?;      // letterbox 输入尺寸
    model.set_conf_threshold(0.25)?;      // 置信度阈值（默认 0.25）
    model.set_nms_threshold(0.45)?;       // NMS IoU 阈值（默认 0.5）

    // 4. 单图推理：predict(&Image) -> Vec<Detection>
    let img = Image::read("test.jpg")?;
    let dets = model.predict(&img)?;
    for d in &dets {
        println!("label={} score={:.3} rect=({:.0},{:.0},{:.0},{:.0})",
                 d.label_id, d.score, d.rect.x, d.rect.y, d.rect.width, d.rect.height);
    }

    // 5. 批量推理：predict_batch(&[&Image]) -> Vec<Vec<Detection>>
    let img2 = Image::read("bus.jpg")?;
    let batch = model.predict_batch(&[&img, &img2])?;
    for (i, dets) in batch.iter().enumerate() {
        println!("image {}: {} objects", i, dets.len());
    }

    // 6. 可视化：predict_and_draw 句柄直达 C++ vis_det，把结果绘制到 canvas（并返回检测结果）
    let canvas = img.clone()?;
    model.predict_and_draw(
        &img,
        &canvas,
        &DrawOptions::new()
            .with_threshold(0.25)
            .with_label_map(vec![(0, "person".into()), (1, "bicycle".into()), (2, "car".into())])
            .with_alpha(0.3),
    )?;
    canvas.save("det_vis.png")?;

    // 7. 多线程：clone() 深拷贝独立实例（返回 Result<Self, MdError>，每线程持有一个）
    let model2 = model.clone()?;
    Ok(())
}
```

## 4. 实例分割（`UltralyticsSeg`）

实例分割模型 `UltralyticsSeg::new(path, &opt)?` 加载。结果类型 `InstanceSeg`（字段 `rect: Rect`、`label_id: i32`、`score: f32`）。

> **mask 说明**：本绑定的 `InstanceSeg` 结构**未封装掩码读取**（C API 的 `md_result_mask` 未桥接）；需要掩码数据请用 C++ / C API / Python / C# 绑定，或按 `mask` 尺寸自行经 C API 扩展。可视化（`predict_and_draw`）不受影响，底层 `vis_iseg` 会完整绘制掩码。

```rust
use modeldeploy::{DrawOptions, Image, RuntimeOption, UltralyticsSeg};
use modeldeploy::ffi::MDDevice;

fn main() -> Result<(), modeldeploy::MdError> {
    // 1. 运行时选项 + 构造（详见上节）
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(MDDevice::CPU, 0)?.set_cpu_threads(4)?;
    let model = UltralyticsSeg::new("yolo11n-seg.onnx", &opt)?;

    // 2. 预处理/后处理参数（均为 Result<(), MdError>）
    model.set_input_size(640, 640)?;      // letterbox 输入尺寸
    model.set_conf_threshold(0.25)?;      // 置信度阈值（默认 0.25）
    model.set_nms_threshold(0.45)?;       // NMS IoU 阈值（默认 0.5）
    model.set_mask_threshold(0.5)?;       // 掩码二值化阈值（默认 0.5）

    // 3. 单图推理：predict(&Image) -> Vec<InstanceSeg>
    let img = Image::read("test.jpg")?;
    let instances = model.predict(&img)?;
    for r in &instances {
        println!("label={} score={:.3} rect=({:.0},{:.0},{:.0},{:.0})",
                 r.label_id, r.score, r.rect.x, r.rect.y, r.rect.width, r.rect.height);
    }

    // 4. 批量推理：predict_batch(&[&Image]) -> Vec<Vec<InstanceSeg>>（按图分组）
    let img2 = Image::read("bus.jpg")?;
    let batch = model.predict_batch(&[&img, &img2])?;
    for (i, rs) in batch.iter().enumerate() {
        println!("image {}: {} instances", i, rs.len());
    }

    // 5. 可视化：predict_and_draw 句柄直达 C++ vis_iseg，把结果绘制到 canvas
    let canvas = img.clone()?;
    model.predict_and_draw(&img, &canvas, &DrawOptions::new().with_threshold(0.5))?;
    canvas.save("iseg_vis.jpg")?;

    // 6. 多线程：clone() 深拷贝独立实例（返回 Result<Self, MdError>）
    let model2 = model.clone()?;
    Ok(())
}
```

## 5. FastSAM（`FastSam`）

FastSAM 结果与实例分割同构（`Vec<InstanceSeg>`，同样不含 mask，见上节说明）。
`predict_with_prompts` 在全量结果上按提示过滤实例，**不重跑网络**；空切片提示等价全图 `predict`。

```rust
use modeldeploy::{FastSam, Image, RuntimeOption};
use modeldeploy::ffi::MDDevice;

fn main() -> Result<(), modeldeploy::MdError> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(MDDevice::CPU, 0)?;
    let model = FastSam::new("fastsam-s.onnx", &opt)?;

    // 1. 参数：默认输入 640x640、conf 0.30 / nms 0.5 / mask 0.5（官方 FastSAM-s 常配 1024x1024）
    model.set_input_size(1024, 1024)?;
    model.set_conf_threshold(0.30)?;
    model.set_nms_threshold(0.40)?;
    model.set_mask_threshold(0.5)?;

    let img = Image::read("test.jpg")?;

    // 2. 全图（Everything）分割：predict -> Vec<InstanceSeg>
    let all = model.predict(&img)?;

    // 3. 提示过滤：bboxes 为 [x,y,w,h,...]（原图像素，每个框取 IoU 最大实例）、
    //    points 为 [x,y,...]、labels 逐点（1=前景保留, 0=背景剔除）
    let prompted = model.predict_with_prompts(
        &img,
        &[100.0, 80.0, 220.0, 180.0],
        &[150.0, 130.0],
        &[1],
    )?;
    println!("all={} prompted={}", all.len(), prompted.len());

    // 4. 可视化同实例分割：predict_and_draw（底层 vis_iseg）
    Ok(())
}
```

## 6. 语义分割（`UltralyticsSem`）

语义分割模型（`yolo26n-sem` 等，cityscapes 19 类）。`predict` 返回 `Vec<SemSeg>`（单图单结果，取 `[0]`），`SemSeg`（字段 `labels: Vec<u8>`、`height`/`width: usize`、`num_classes: i32`）；`labels` 为每像素类别索引 `[0, num_classes)`（行主序）。该模型无阈值 setter（后处理 argmax，运行时无参数）。

```rust
use modeldeploy::{Image, RuntimeOption, UltralyticsSem};
use modeldeploy::ffi::MDDevice;

fn main() -> Result<(), modeldeploy::MdError> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(MDDevice::CPU, 0)?;
    let model = UltralyticsSem::new("yolo26n-sem.onnx", &opt)?;
    model.set_input_size(640, 640)?;      // 输入尺寸可调；无其它参数

    let img = Image::read("test.jpg")?;
    let sems = model.predict(&img)?;
    let sem = &sems[0];
    // 逐像元读取：sem.labels[y * sem.width + x]
    println!("sem {}x{} classes={} labels={}",
             sem.width, sem.height, sem.num_classes, sem.labels.len());

    // 批量推理：predict_batch(&[&Image]) -> Vec<Vec<SemSeg>>（按图分组）
    let img2 = Image::read("bus.jpg")?;
    let batch = model.predict_batch(&[&img, &img2])?;

    // 可视化：predict_and_draw（底层 vis_sem，cityscapes 调色板叠加）
    let canvas = img.clone()?;
    model.predict_and_draw(&img, &canvas, &modeldeploy::DrawOptions::new())?;
    canvas.save("sem_vis.jpg")?;
    Ok(())
}
```

## 7. 深度估计（`UltralyticsDepth`）

深度估计模型（`yolo26n-depth` 等）。`predict` 返回 `Vec<Depth>`（单图单结果，取 `[0]`），`Depth`（字段 `depth: Vec<f32>`、`height`/`width: usize`）；每像素深度单位**米**（log 输出已 `exp` 还原，行主序）。该模型无阈值 setter（后处理无参数）。

```rust
use modeldeploy::{Image, RuntimeOption, UltralyticsDepth};
use modeldeploy::ffi::MDDevice;

fn main() -> Result<(), modeldeploy::MdError> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(MDDevice::CPU, 0)?;
    let model = UltralyticsDepth::new("yolo26n-depth.onnx", &opt)?;
    model.set_input_size(640, 640)?;

    let img = Image::read("test.jpg")?;
    let depths = model.predict(&img)?;
    let dep = &depths[0];
    // 逐像元读取：dep.depth[y * dep.width + x]（米）
    let mut near = f32::MAX;
    let mut far = f32::MIN;
    for &d in &dep.depth { near = near.min(d); far = far.max(d); }
    println!("depth {}x{} range=[{:.2}, {:.2}] m", dep.width, dep.height, near, far);

    // 批量推理：predict_batch(&[&Image]) -> Vec<Vec<Depth>>（按图分组）

    // 可视化：predict_and_draw（底层 vis_depth JET 伪彩）
    let canvas = img.clone()?;
    model.predict_and_draw(&img, &canvas, &modeldeploy::DrawOptions::new())?;
    canvas.save("depth_vis.jpg")?;
    Ok(())
}
```

## 8. 姿态与关键点族（`UltralyticsPose` / `HandKeypoint` / `VehicleKeypoint` / `FaceLandmark`）

姿态与关键点模型族结果类型统一为 `Pose`（字段 `rect: Rect`、`score: f32`、`keypoints: Vec<Point3>`，`Point3` 含 `x/y/z`，`z` 为关键点置信度；无 `label_id`）。`predict` 返回 `Vec<Pose>`，`predict_batch` 返回 `Vec<Vec<Pose>>`（按图分组）。类对应：`UltralyticsPose`（COCO 17 点人体骨架）、`HandKeypoint`（21 点手部）、`VehicleKeypoint`（4 车轮关键点）、`FaceLandmark`（InsightFace 2d106 面部 106 点，`z` 恒为 0，输入须为人脸裁剪图）。

```rust
use modeldeploy::{Image, RuntimeOption, UltralyticsPose};
use modeldeploy::ffi::MDDevice;

fn main() -> Result<(), modeldeploy::MdError> {
    // 1. 运行时选项（详见上节）
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(MDDevice::CPU, 0)?.set_cpu_threads(4)?;

    // 2. 构造：HandKeypoint / VehicleKeypoint / FaceLandmark 同为 ::new(path, &opt)?
    let model = UltralyticsPose::new("yolo11n-pose.onnx", &opt)?;
    let hand = modeldeploy::HandKeypoint::new("hand.onnx", &opt)?;
    let vehicle = modeldeploy::VehicleKeypoint::new("vehicle.onnx", &opt)?;
    let face = modeldeploy::FaceLandmark::new("face_landmark.onnx", &opt)?;

    // 3. 预处理/后处理参数（UltralyticsPose/HandKeypoint/VehicleKeypoint 共用；
    //    FaceLandmark 无参数表，调阈值 setter 会返回 Err）
    model.set_input_size(640, 640)?;      // letterbox 输入尺寸
    model.set_conf_threshold(0.30)?;      // 置信度阈值（默认 0.30）
    model.set_nms_threshold(0.45)?;       // NMS IoU 阈值（默认 0.5）
    model.set_keypoints_num(17)?;         // 关键点数（默认 17，须与模型输出一致）
    hand.set_keypoints_num(21)?;          // 手部 21 点（构造默认 21）
    vehicle.set_keypoints_num(4)?;        // 车轮 4 点（构造默认 4，不同车型可覆盖）

    // 4. 单图推理：predict(&Image) -> Vec<Pose>
    let img = Image::read("test.jpg")?;
    let poses = model.predict(&img)?;
    for p in &poses {
        println!("score={:.3} rect=({:.0},{:.0},{:.0},{:.0}) kps={}",
                 p.score, p.rect.x, p.rect.y, p.rect.width, p.rect.height, p.keypoints.len());
        for kp in &p.keypoints {
            println!("  kp=({:.1}, {:.1}, {:.2})", kp.x, kp.y, kp.z);   // z 为关键点置信度
        }
    }
    // 面部 Landmark：输入人脸裁剪图 -> 单元素 Vec（106 点 z=0，rect 为整图、score=1.0）
    let crop = Image::read("face_crop.jpg")?;
    let _flms = face.predict(&crop)?;

    // 5. 批量推理：predict_batch(&[&Image]) -> Vec<Vec<Pose>>（按图分组）
    let img2 = Image::read("bus.jpg")?;
    let batch = model.predict_batch(&[&img, &img2])?;
    for (i, ps) in batch.iter().enumerate() {
        println!("image {}: {} persons", i, ps.len());
    }

    // 6. 可视化：predict_and_draw 句柄直达 C++ vis_pose（COCO 骨架连线）
    let canvas = img.clone()?;
    model.predict_and_draw(&img, &canvas, &modeldeploy::DrawOptions::new().with_alpha(0.3))?;
    canvas.save("pose_vis.jpg")?;

    // 7. 多线程：clone() 深拷贝独立实例（返回 Result<Self, MdError>）
    let model2 = model.clone()?;
    Ok(())
}
```

## 9. OBB（旋转框检测）（`UltralyticsObb`）

旋转框检测模型 `UltralyticsObb::new(path, &opt)?` 加载（输入默认 1024x1024）。结果类型 `Obb`（字段 `rotated_box: RotatedBox{cx, cy, width, height, angle}`、`label_id: i32`、`score: f32`）；`cx/cy` 为旋转框中心、`angle` 为弧度角，坐标均为原图像素。

```rust
use modeldeploy::{DrawOptions, Image, RuntimeOption, UltralyticsObb};
use modeldeploy::ffi::MDDevice;

fn main() -> Result<(), modeldeploy::MdError> {
    // 1. 运行时选项 + 构造（详见上节）
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(MDDevice::CPU, 0)?.set_cpu_threads(4)?;
    let model = UltralyticsObb::new("yolo11n-obb.onnx", &opt)?;

    // 2. 预处理/后处理参数（均为 Result<(), MdError>）
    model.set_input_size(1024, 1024)?;    // letterbox 输入尺寸（默认 1024x1024）
    model.set_conf_threshold(0.25)?;      // 置信度阈值（默认 0.25）
    model.set_nms_threshold(0.45)?;       // NMS IoU 阈值（默认 0.5）

    // 3. 单图推理：predict(&Image) -> Vec<Obb>
    let img = Image::read("test.jpg")?;
    let obbs = model.predict(&img)?;
    for r in &obbs {
        let rb = &r.rotated_box;
        println!("label={} score={:.3} obb=(xc={:.1}, yc={:.1}, w={:.1}, h={:.1}, angle={:.3})",
                 r.label_id, r.score, rb.cx, rb.cy, rb.width, rb.height, rb.angle);
    }

    // 4. 批量推理：predict_batch(&[&Image]) -> Vec<Vec<Obb>>（按图分组）
    let img2 = Image::read("bus.jpg")?;
    let batch = model.predict_batch(&[&img, &img2])?;
    for (i, rs) in batch.iter().enumerate() {
        println!("image {}: {} rotated boxes", i, rs.len());
    }

    // 5. 可视化：predict_and_draw 句柄直达 C++ vis_obb，把结果绘制到 canvas
    let canvas = img.clone()?;
    model.predict_and_draw(&img, &canvas, &DrawOptions::new().with_threshold(0.5).with_alpha(0.3))?;
    canvas.save("obb_vis.jpg")?;

    // 6. 多线程：clone() 深拷贝独立实例（返回 Result<Self, MdError>）
    let model2 = model.clone()?;
    Ok(())
}
```

## 10. 图像分类（`Classification`）

分类模型 `Classification::new(path, &opt)?` 加载（输入默认 224x224）。单图返回 `Vec<ClassificationResult>`（字段 `label_id: i32`、`score: f32`，Top-K 逐项）。参数：`set_top_k`（默认 1）、`set_multi_label`（默认 false）。

```rust
use modeldeploy::{Classification, DrawOptions, Image, RuntimeOption};
use modeldeploy::ffi::MDDevice;

fn main() -> Result<(), modeldeploy::MdError> {
    // 1. 运行时选项 + 构造（详见上节）
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(MDDevice::CPU, 0)?.set_cpu_threads(4)?;
    let model = Classification::new("yolo11n-cls.onnx", &opt)?;

    // 2. 预处理/后处理参数
    model.set_input_size(224, 224)?;      // 输入尺寸（默认 224x224）
    model.set_top_k(5)?;                  // Top-K 输出个数（默认 1）
    model.set_multi_label(false)?;        // 多标签模式（默认 false）

    // 3. 单图推理：predict(&Image) -> Vec<ClassificationResult>
    let img = Image::read("test.jpg")?;
    let cls = model.predict(&img)?;
    for c in &cls {
        println!("label={} score={:.3}", c.label_id, c.score);
    }

    // 4. 批量推理：predict_batch(&[&Image]) -> Vec<Vec<ClassificationResult>>（按图分组）
    let img2 = Image::read("bus.jpg")?;
    let batch = model.predict_batch(&[&img, &img2])?;
    for (i, cs) in batch.iter().enumerate() {
        println!("image {}: {} labels", i, cs.len());
    }

    // 5. 可视化：predict_and_draw 句柄直达 C++ vis_cls
    let canvas = img.clone()?;
    model.predict_and_draw(&img, &canvas, &DrawOptions::new().with_threshold(0.35).with_alpha(0.3))?;
    canvas.save("cls_vis.jpg")?;

    // 6. 多线程：clone() 深拷贝独立实例（返回 Result<Self, MdError>）
    let model2 = model.clone()?;
    Ok(())
}
```

## 11. OCR（`PaddleOCR` + 子模型）

主流水线 `PaddleOCR::new(model_path, &opt)?`：`model_path` 用 `|` 串联 **det/cls/rec/dict 四段**（`"det.onnx|cls.onnx|rec.onnx|dict.txt"`）。单图返回 `Vec<OcrLine>`（字段 `quad: [i32; 8]`（4 点，原图像素）、`text: String`、`score: f32`、`cls_label: i32`、`cls_score: f32`，逐行配对）。注意：单图 `predict` 经 C API 单值包装**仅返回首行**，完整逐行结果请用 `predict_batch(&[&img])`（逐图逐行完整读取）。

```rust
use modeldeploy::{DbDetectorModel, DrawOptions, Image, OcrClassifierModel, PaddleOCR,
                  RecognizerModel, RuntimeOption};
use modeldeploy::ffi::MDDevice;

fn main() -> Result<(), modeldeploy::MdError> {
    // 1. 运行时选项 + 构造（det|cls|rec|dict 四段路径，详见上节）
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(MDDevice::CPU, 0)?.set_cpu_threads(4)?;
    let model = PaddleOCR::new("det.onnx|cls.onnx|rec.onnx|dict.txt", &opt)?;

    // 2. 参数设置（括号内为默认值）
    model.set_det_db_thresh(0.3)?;        // DB 二值化阈值（默认 0.3）
    model.set_det_db_box_thresh(0.6)?;    // 框置信度阈值（默认 0.6）
    model.set_det_db_unclip_ratio(1.5)?;  // 扩框比例（默认 1.5）
    model.set_det_db_score_mode("slow")?; // 框得分模式（默认 "slow"）
    model.set_use_dilation(false)?;       // 是否膨胀（默认 false）
    model.set_cls_thresh(0.9)?;           // 方向分类阈值（默认 0.9）
    model.set_max_side_len(960)?;         // 检测最长边（默认 960）
    model.set_cls_batch_size(6)?;         // 方向分类子模型 batch（默认 6）
    model.set_rec_batch_size(8)?;         // 识别子模型 batch（默认 8）
    model.set_rec_image_shape(3, 48, 320)?; // 识别输入形状（默认 3x48x320）

    // 3. 单图推理：Vec<OcrLine>（text/quad/score/cls_label/cls_score 逐行配对；仅首行，见上）
    let img = Image::read("test.jpg")?;
    let lines = model.predict(&img)?;
    for line in &lines {
        println!("{} {:.3} cls={} box={:?}", line.text, line.score, line.cls_label, line.quad);
    }

    // 4. 批量推理：predict_batch(&[&Image]) -> Vec<Vec<OcrLine>>（按图分组，逐行完整）
    let img2 = Image::read("bus.jpg")?;
    let batch = model.predict_batch(&[&img, &img2])?;
    for (i, ls) in batch.iter().enumerate() {
        println!("image {}: {} lines", i, ls.len());
    }

    // 5. 可视化：predict_and_draw 句柄直达 C++ vis_ocr
    let canvas = img.clone()?;
    model.predict_and_draw(&img, &canvas,
        &DrawOptions::new().with_font("msyh.ttc", 14).with_alpha(0.3))?;
    canvas.save("ocr_vis.jpg")?;

    // 6. 子模型独立使用（也可不经 PaddleOCR 单独构造；predict 均返回 Vec<OcrLine>）
    let db  = DbDetectorModel::new("det.onnx", &opt)?;           // quad（文本框）
    let rec = RecognizerModel::new("rec.onnx|dict.txt", &opt)?;  // text/score（路径 '|' 两段）
    let clr = OcrClassifierModel::new("cls.onnx", &opt)?;        // 方向分类
    let _det_lines = db.predict(&img)?;
    let _rec_lines = rec.predict(&img)?;
    let _cls_lines = clr.predict(&img)?;

    // 7. 多线程：clone() 深拷贝独立实例（返回 Result<Self, MdError>）
    let model2 = model.clone()?;
    Ok(())
}
```

## 主要模块文件

| 文件 | 说明 |
|------|------|
| `src/runtime.rs` | `RuntimeOption` 封装 |
| `src/ffi.rs` | FFI 声明（`MD_BACKEND_*` 常量等） |
| `src/model.rs` | 模型加载/推理 |
| `src/audio.rs` | 音频能力封装 |
| `src/barcode.rs` | 条码/二维码封装 |
| `src/nlp.rs` | NLP 能力封装 |
| `src/solution.rs` | 解决方案封装 |
| `src/tracker.rs` | 跟踪能力封装 |
| `src/video.rs` | 视频编解码封装（解码/编码全功能） |
| `src/image.rs` | 图像处理 |
| `src/types.rs` / `src/error.rs` | 类型与错误 |

## 示例（运行于 `rust/modeldeploy/examples/`）

内置 12 个示例：`classification` / `depth` / `detection` / `face_age` / `face_detection` / `face_gender` / `face_rec` / `obb` / `pose` / `seg` / `sem` / `video_decode`。

```bash
cd rust/modeldeploy
cargo run --example detection
cargo run --example video_decode -- demo.mp4 100
```

## 图像原始字节（`Image`）

`Image` 提供原生格式的宿主字节读取（`format()` / `width()` / `height()` / `plane_count()` 见 `Image`）：

```rust
let raw: Vec<u8> = img.to_native_bytes()?;   // 整幅原生连续字节；GPU 设备帧自动回读
let y:  Vec<u8>  = img.plane_bytes(0)?;      // 第 0 平面（NV12 的 Y）
let uv: Vec<u8>  = img.plane_bytes(1)?;      // 第 1 平面（NV12 的 UV）
```

- `to_native_bytes()` 布局随 `format()` 而定：BGR24=`w*h*3`；NV12=`w*h + w*h/2`；I420=`w*h + w*h/4 + w*h/4`。
- 平面越界 / 不支持类型返回 `Err`；方法立即把缓冲拷贝进自有 `Vec<u8>`。

## 视频编解码（`video`）

`video` 模块等价于 C++ `modeldeploy::video`（解码+编码全功能，经 C API `md_video_*` 承载）。主要公开类型：

| 类型 | 说明 |
|------|------|
| `VideoConfig` | 解码/编码配置（构建器，setter 返回 `&mut Self`） |
| `VideoCapabilities` | 能力探测 |
| `VideoDecoder` | 解码器：同步 `read_frame` + 异步回调 + 状态/统计 |
| `VideoEncoder` | 编码器：`encode` / `encode_async` / `start_async` |
| `VideoFrame { image, pts_ms }` | 统一帧（`Image` 本绑定自有） |
| `VideoStats` | 编解码统计 |

枚举：`CodecBackend{Auto,FFmpeg,GStreamer}`、`HwAccel{Auto,None,Cuda,Vaapi,Sophgo}`、`Backpressure{Block,Drop,OverwriteOldest}`、`VideoState{Idle,Opening,Running,Reconnecting,Eof,Error,Closed}`。

```rust
use modeldeploy::{Backpressure, CodecBackend, VideoConfig, VideoDecoder, VideoFrame};

let mut cfg = VideoConfig::new()?;
cfg.backend(CodecBackend::FFmpeg)?
    .backpressure(Backpressure::Block)?
    .pooling(true)?;

let dec = VideoDecoder::create(Some(&cfg))?;
dec.open("demo.mp4")?;                       // 或 rtsp://...
let (w, h, f) = dec.size()?;
println!("{}x{} @ {}fps", w, h, f);

while let Ok(VideoFrame { image, pts_ms }) = dec.read_frame() {
    // image 为本绑定自有 Image，Drop 自动释放底层 md_image_destroy
    println!("{}ms {}x{}", pts_ms, image.width(), image.height());
}
let st = dec.stats()?;                       // VideoStats
```

**异步解码（回调推送）**：`dec.set_callback(|frame| { ... })` 注册交付回调（帧为自有 `VideoFrame`，用后自动释放；回调自后台线程投递），再 `dec.start()` / `dec.stop()`。

**编码**：
```rust
let mut ecfg = VideoConfig::new()?;
ecfg.codec("h264")?.bitrate_kbps(2000)?.gop(30)?.fps(25)?;
let enc = modeldeploy::VideoEncoder::create(Some(&ecfg))?;
enc.open("out.mp4", 1280, 720, 25)?;
enc.encode(&img, pts_ms)?;                    // img 生命周期须覆盖本次调用
enc.close();
```

设备显存（`device_only` 解码 / `gpu_direct_input` 编码）零拷贝直通。详见 [视频接口总览](../video/api.md)。
