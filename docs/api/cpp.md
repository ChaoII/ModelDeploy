# C++ 绑定

首选，完整功能，支持全部模型与后端。核心逻辑全在 C++ SDK。编译链接见 [快速开始](../quickstart.md#3-编写第一个检测程序)。

## 1. 安装/引入

引入视觉与音频头文件：

```cpp
#include "modeldeploy/vision.h"
#include "modeldeploy/audio.h"
```

CMake 链接 SDK（经 `find_package` 安装后）：

```cmake
find_package(ModelDeploySDK)
target_link_libraries(your_target PRIVATE ModelDeploySDK)
```

## 2. RuntimeOption（后端/设备/精度）

`RuntimeOption` 是运行时配置结构，控制后端、设备、精度、线程数等，所有模型类通过它初始化。完整字段见 [配置详解](../runtime_option.md)。

```cpp
#include "modeldeploy/vision.h"

modeldeploy::RuntimeOption option;

// 后端：五选一
option.use_ort_backend();       // OnnxRuntime（.onnx，最通用）
// option.use_mnn_backend();    // MNN（.mnn，移动端/边缘）
// option.use_trt_backend();    // TensorRT（.engine，GPU 最高性能）
// option.use_sophgo_backend(); // Sophgo（.bmodel，算能 TPU）
// option.use_ncnn_backend();   // ncnn（.param/.bin，CPU/Vulkan，YOLO 全系）

// 设备
option.use_cpu();                                   // CPU（默认）
// option.use_gpu(device_id);                       // NVIDIA GPU
// option.set_device(modeldeploy::Device::OPENCL, 0); // 需显式 use_mnn_backend()
// option.set_device(modeldeploy::Device::VULKAN, 0); // 需显式 use_mnn_backend()
// option.set_device(modeldeploy::Device::GPU, 0);
// option.set_device(modeldeploy::Device::TPU, 0);

// CPU 线程数
option.set_cpu_thread_num(4);

// 精度（公有字段，直接赋值）——enable_trt 仅对 ORT 后端生效（启用 TRT EP）
option.enable_fp16 = true;
option.enable_trt = true;

// 模型路径（可选，加密模型可带密码）
option.set_model_path("m.onnx");

auto det = modeldeploy::vision::detection::UltralyticsDet("yolo11n.onnx", option);
```

> **注意**：OPENCL/VULKAN 需显式 `use_mnn_backend()`，否则 fail-closed。其余后端与设备的组合见 [后端详解](../backends.md)。

## 3. 目标检测（UltralyticsDet）

Ultralytics YOLO 检测模型。结果类型 `vision::DetectionResult`（字段 `box: Rect2f{x,y,width,height}`、`label_id: int32_t`、`score: float`），`predict`/`batch_predict` 可选传入 `TimerArray*` 做分段计时。

```cpp
#include "modeldeploy/vision.h"

int main() {
    // 1. 运行时选项（详见上节）
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    opt.set_cpu_thread_num(4);

    // 2. 构造模型
    auto det = std::make_unique<modeldeploy::vision::detection::UltralyticsDet>("yolo11n.onnx", opt);
    if (!det->is_initialized()) return 1;

    // 3. 预处理/后处理参数
    det->get_preprocessor().set_size({640, 640});       // letterbox 输入尺寸（默认 {640, 640}）
    det->get_preprocessor().set_padding_value({114.f, 114.f, 114.f});
    det->get_postprocessor().set_conf_threshold(0.25f); // 置信度阈值（默认 0.25）
    det->get_postprocessor().set_nms_threshold(0.5f);   // NMS IoU 阈值（默认 0.5）

    // 4. 单图推理：predict(image, &res, timers = nullptr)
    auto im = modeldeploy::vision::ImageData::imread("test.jpg");
    std::vector<modeldeploy::vision::DetectionResult> res;
    if (!det->predict(im, &res)) return 1;
    for (const auto& r : res) {
        std::printf("label=%d score=%.3f box=(%.0f, %.0f, %.0f, %.0f)\n",
                    r.label_id, r.score, r.box.x, r.box.y, r.box.width, r.box.height);
    }

    // 5. 可视化：vis_det 返回绘制后的 ImageData；label_map 从模型元数据读取（ultralytics 导出的 onnx 键一般为 "names"）
    const auto label_map = det->get_label_map("names");
    auto vis = modeldeploy::vision::vis_det(im, res, 0.5, label_map, "msyh.ttc", 14, 0.3, false);
    vis.imwrite("det_vis.jpg");
    // 或就地绘制到图像帧（GPU 帧按设备分发）：
    // det->draw_result(im, res, 0.5);

    // 6. 批量推理：一次喂多图（各图统一 letterbox 到 set_size 尺寸后拼 batch），返回按图分组
    std::vector<modeldeploy::vision::ImageData> images = {
        modeldeploy::vision::ImageData::imread("a.jpg"),
        modeldeploy::vision::ImageData::imread("b.jpg"),
    };
    std::vector<std::vector<modeldeploy::vision::DetectionResult>> ress;
    det->batch_predict(images, &ress);
    for (size_t i = 0; i < ress.size(); ++i) {
        std::printf("image %zu: %zu objects\n", i, ress[i].size());
    }

    // 7. 多线程：clone() 深拷贝独立实例（每线程持有一个，互不干扰）
    std::vector<decltype(det->clone())> models;
    for (int i = 0; i < 4; ++i) {
        models.emplace_back(std::move(det->clone()));
    }
    // 各线程用 models[i].get() 调 predict
    return 0;
}
```

## 4. 实例分割（UltralyticsSeg）

Ultralytics YOLO 分割模型（`yolo11n-seg.onnx` 等）。结果类型 `vision::InstanceSegResult`（字段 `box: Rect2f`、`mask: Mask`、`label_id: int32_t`、`score: float`）；`mask` 为 `Mask{buffer, shape}`，`buffer` 是 uint8 0/1 的 H*W 平铺向量（行主序），`shape` 为 `{H, W}`。

```cpp
#include "modeldeploy/vision.h"

int main() {
    // 1. 运行时选项 + 构造（详见上节）
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    opt.set_cpu_thread_num(4);
    auto seg = std::make_unique<modeldeploy::vision::detection::UltralyticsSeg>("yolo11n-seg.onnx", opt);
    if (!seg->is_initialized()) return 1;

    // 2. 预处理/后处理参数
    seg->get_preprocessor().set_size({640, 640});       // letterbox 输入尺寸（默认 {640, 640}）
    seg->get_preprocessor().set_padding_value({114.f, 114.f, 114.f});
    seg->get_postprocessor().set_conf_threshold(0.25f); // 置信度阈值（默认 0.25）
    seg->get_postprocessor().set_nms_threshold(0.5f);   // NMS IoU 阈值（默认 0.5）
    seg->get_postprocessor().set_mask_threshold(0.5f);  // 掩码二值化阈值（默认 0.5）

    // 3. 单图推理
    auto im = modeldeploy::vision::ImageData::imread("test.jpg");
    std::vector<modeldeploy::vision::InstanceSegResult> res;
    if (!seg->predict(im, &res)) return 1;
    for (const auto& r : res) {
        std::printf("label=%d score=%.3f box=(%.0f, %.0f, %.0f, %.0f) mask=%dx%d\n",
                    r.label_id, r.score, r.box.x, r.box.y, r.box.width, r.box.height,
                    (int)r.mask.shape[0], (int)r.mask.shape[1]);
        // 逐像元读取掩码：r.mask.buffer[y * r.mask.shape[1] + x]
    }

    // 4. 可视化：vis_iseg；或 seg->draw_result(im, res, 0.5) 就地绘制（GPU 帧按设备分发）
    auto vis = modeldeploy::vision::vis_iseg(im, res, 0.5, "msyh.ttc", 14, 0.3, false);
    vis.imwrite("iseg_vis.jpg");

    // 5. 批量推理：各图统一 letterbox 后拼 batch，返回按图分组
    std::vector<modeldeploy::vision::ImageData> images = {
        modeldeploy::vision::ImageData::imread("a.jpg"),
        modeldeploy::vision::ImageData::imread("b.jpg"),
    };
    std::vector<std::vector<modeldeploy::vision::InstanceSegResult>> ress;
    seg->batch_predict(images, &ress);

    // 6. 类别表与多线程
    const auto label_map = seg->get_label_map("names");
    auto seg2 = seg->clone();
    return 0;
}
```

## 5. 轻量分割一切（FastSAM）

`FastSam`（命名空间 `vision::seg`，`fastsam-s.onnx` 等）一次性输出 box + mask，结果复用 `InstanceSegResult`（消费方式同实例分割）。
`predict_with_prompts` 在全量结果上按提示过滤实例，**不重跑网络**；提示为空等价全图 `predict`：

- `bboxes`（`Rect2f` x/y/width/height，原图像素）：每个框取 IoU 最大的实例。
- `points` + `point_labels`（等长，`1`=前景保留 / `0`=背景剔除）：按掩码是否命中该点保留/剔除实例。

```cpp
#include "modeldeploy/vision.h"

int main() {
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    auto sam = std::make_unique<modeldeploy::vision::seg::FastSam>("fastsam-s.onnx", opt);
    if (!sam->is_initialized()) return 1;

    // 1. 参数：默认输入 640x640、conf 0.30 / nms 0.5 / mask 0.5（官方 FastSAM-s 常配 1024x1024）
    sam->get_preprocessor().set_size({1024, 1024});
    sam->get_postprocessor().set_conf_threshold(0.30f);
    sam->get_postprocessor().set_nms_threshold(0.40f);
    sam->get_postprocessor().set_mask_threshold(0.5f);

    // 2. 全图（Everything）分割
    auto im = modeldeploy::vision::ImageData::imread("test.jpg");
    std::vector<modeldeploy::vision::InstanceSegResult> all;
    sam->predict(im, &all);

    // 3. 提示过滤
    modeldeploy::vision::seg::FastSamPrompts prompts;
    prompts.bboxes.push_back(modeldeploy::vision::Rect2f(100.f, 80.f, 220.f, 180.f));
    prompts.points.push_back(modeldeploy::vision::Point2f(150.f, 130.f));
    prompts.point_labels.push_back(1);
    std::vector<modeldeploy::vision::InstanceSegResult> prompted;
    sam->predict_with_prompts(im, prompts, &prompted);

    // 4. 可视化（vis_iseg 同实例分割）
    auto vis = modeldeploy::vision::vis_iseg(im, prompted, 0.3, "msyh.ttc", 14, 0.5, false);
    vis.imwrite("fastsam_vis.jpg");
    return 0;
}
```

## 6. 语义分割（UltralyticsSem）

语义分割模型（`yolo26n-sem` 等，cityscapes 19 类）。结果类型 `vision::SemSegResult`（字段 `labels: vector<uint8_t>`、`shape: {H, W}`、`num_classes: int32_t`）；`labels` 为每像素类别索引 `[0, num_classes)`（行主序 H*W），后处理 argmax 并去除 letterbox 边（无参数可调）。

```cpp
#include "modeldeploy/vision.h"

int main() {
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    auto sem = std::make_unique<modeldeploy::vision::detection::UltralyticsSem>("yolo26n-sem.onnx", opt);
    if (!sem->is_initialized()) return 1;

    sem->get_preprocessor().set_size({640, 640});   // 后处理无参数

    auto im = modeldeploy::vision::ImageData::imread("test_sem_540.jpg");
    modeldeploy::vision::SemSegResult res;
    if (!sem->predict(im, &res)) return 1;
    const int h = (int)res.shape[0], w = (int)res.shape[1];
    const uint8_t* lab = res.labels.data();         // lab[y * w + x]
    std::printf("sem %dx%d classes=%d\n", h, w, res.num_classes);

    // 批量推理：std::vector<SemSegResult> ress; sem->batch_predict(images, &ress);

    // 可视化：vis_sem（cityscapes 调色板叠加；label_map 从模型元数据读取）
    const auto label_map = sem->get_label_map("names");
    auto vis = modeldeploy::vision::vis_sem(im, res, label_map, 0.5, false);
    vis.imwrite("sem_vis.jpg");
    return 0;
}
```

## 7. 深度估计（UltralyticsDepth）

深度估计模型（`yolo26n-depth` 等）。结果类型 `vision::DepthResult`（字段 `depth: vector<float>`、`shape: {H, W}`）；log 空间深度已 `exp` 还原为**米**（后处理无参数可调）。

```cpp
#include "modeldeploy/vision.h"

int main() {
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    auto dep = std::make_unique<modeldeploy::vision::detection::UltralyticsDepth>("yolo26n-depth.onnx", opt);
    if (!dep->is_initialized()) return 1;

    dep->get_preprocessor().set_size({640, 640});

    auto im = modeldeploy::vision::ImageData::imread("test_depth_540.jpg");
    modeldeploy::vision::DepthResult res;
    if (!dep->predict(im, &res)) return 1;
    const int h = (int)res.shape[0], w = (int)res.shape[1];
    const float* d = res.depth.data();              // d[y * w + x]，单位米
    std::printf("depth %dx%d\n", h, w);

    // 批量推理：std::vector<DepthResult> ress; dep->batch_predict(images, &ress);

    // 可视化：vis_depth（JET 伪彩；colorize=false 输出灰度深度）
    auto vis = modeldeploy::vision::vis_depth(im, res, true, false);
    vis.imwrite("depth_vis.jpg");
    return 0;
}
```

## 8. 姿态与关键点族（UltralyticsPose / HandKeypoint / VehicleKeypoint / FaceLandmark）

姿态与关键点模型族结果类型统一为 `vision::KeyPointsResult`（字段 `box: Rect2f`、`keypoints: vector<Point3f>`（`x/y/z`，`z` 为关键点置信度）、`label_id: int32_t`、`score: float`）。类分布：`detection::UltralyticsPose`（COCO 17 点人体骨架）、`hand::HandKeypoint`（21 点手部）、`landmark::VehicleKeypoint`（4 车轮关键点）与 `landmark::FaceLandmark`（InsightFace 2d106 面部 106 点）。

```cpp
#include "modeldeploy/vision.h"

int main() {
    // 1. 运行时选项（详见上节）
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();

    // 2. 构造：UltralyticsPose（detection）/ HandKeypoint（hand）/ VehicleKeypoint、FaceLandmark（landmark）
    auto pose = std::make_unique<modeldeploy::vision::detection::UltralyticsPose>("yolo11n-pose.onnx", opt);
    auto hand = std::make_unique<modeldeploy::vision::hand::HandKeypoint>("hand.onnx", opt);
    auto vehicle = std::make_unique<modeldeploy::vision::landmark::VehicleKeypoint>("vehicle.onnx", opt);
    auto face = std::make_unique<modeldeploy::vision::landmark::FaceLandmark>("face_landmark.onnx", opt);
    if (!pose->is_initialized()) return 1;

    // 3. 预处理/后处理参数（HandKeypoint 构造默认 21 点、VehicleKeypoint 默认 4 点；
    //    FaceLandmark 无 get_preprocessor/get_postprocessor，参数不可调）
    pose->get_preprocessor().set_size({640, 640});       // letterbox 输入尺寸（默认 {640, 640}）
    pose->get_postprocessor().set_conf_threshold(0.30f); // 置信度阈值（默认 0.30）
    pose->get_postprocessor().set_nms_threshold(0.45f);  // NMS IoU 阈值（默认 0.5）
    pose->get_postprocessor().set_keypoints_num(17);     // 关键点数（默认 17，须与模型输出一致）
    hand->get_postprocessor().set_nms_threshold(0.45f);  // HandKeypoint/VehicleKeypoint 经 get_postprocessor 透传同款参数
    vehicle->get_postprocessor().set_keypoints_num(4);   // 不同车型模型可覆盖点数

    // 4. 单图推理
    auto im = modeldeploy::vision::ImageData::imread("test.jpg");
    std::vector<modeldeploy::vision::KeyPointsResult> res;
    if (!pose->predict(im, &res)) return 1;
    for (const auto& r : res) {
        std::printf("label=%d score=%.3f box=(%.0f, %.0f, %.0f, %.0f) kps=%zu\n",
                    r.label_id, r.score, r.box.x, r.box.y, r.box.width, r.box.height, r.keypoints.size());
        for (const auto& kp : r.keypoints) {
            std::printf("  kp=(%.1f, %.1f, %.2f)\n", kp.x, kp.y, kp.z);   // z 为关键点置信度
        }
    }
    // FaceLandmark：输入人脸裁剪图 -> 单元素结果（106 点 z=0，box 为整图、score=1.0）
    auto crop = modeldeploy::vision::ImageData::imread("face_crop.jpg");
    std::vector<modeldeploy::vision::KeyPointsResult> fres;
    if (!face->predict(crop, &fres)) return 1;

    // 5. 可视化：vis_pose（COCO 骨架连线）/ vis_hand（手部连线）/ vis_keypoints（仅关键点+框，车辆/面部用）
    auto vis = modeldeploy::vision::vis_pose(im, res, "msyh.ttc", 14, 4, 0.3, false);
    vis.imwrite("pose_vis.jpg");
    // 或就地绘制到图像帧（GPU 帧按设备分发）：
    // hand->draw_result(im, hand_res, 0.5); vehicle->draw_result(im, vres, 0.5);

    // 6. 批量推理：各图统一 letterbox 后拼 batch，返回按图分组（FaceLandmark 无 batch_predict）
    std::vector<modeldeploy::vision::ImageData> images = {
        modeldeploy::vision::ImageData::imread("a.jpg"),
        modeldeploy::vision::ImageData::imread("b.jpg"),
    };
    std::vector<std::vector<modeldeploy::vision::KeyPointsResult>> ress;
    pose->batch_predict(images, &ress);
    hand->batch_predict(images, &ress);

    // 7. 多线程：clone() 深拷贝独立实例（每线程持有一个，互不干扰）
    auto pose2 = pose->clone();
    return 0;
}
```

## 9. OBB（旋转框检测）（UltralyticsObb）

Ultralytics YOLO-OBB 模型（命名空间 `vision::detection`，`yolo11n-obb.onnx` 等）。结果类型 `vision::ObbResult`（字段 `rotated_box: RotatedRect{xc, yc, width, height, angle}`、`label_id: int32_t`、`score: float`）；`xc/yc` 为旋转框中心、`angle` 为弧度角，坐标均为原图像素。

```cpp
#include "modeldeploy/vision.h"

int main() {
    // 1. 运行时选项 + 构造（详见上节）
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    opt.set_cpu_thread_num(4);
    auto obb = std::make_unique<modeldeploy::vision::detection::UltralyticsObb>("yolo11n-obb.onnx", opt);
    if (!obb->is_initialized()) return 1;

    // 2. 预处理/后处理参数
    obb->get_preprocessor().set_size({1024, 1024});     // letterbox 输入尺寸（默认 {1024, 1024}）
    obb->get_preprocessor().set_padding_value(114.0f);  // 填充灰值（单值标量）
    obb->get_postprocessor().set_conf_threshold(0.25f); // 置信度阈值（默认 0.25）
    obb->get_postprocessor().set_nms_threshold(0.45f);  // NMS IoU 阈值（默认 0.5）

    // 3. 单图推理
    auto im = modeldeploy::vision::ImageData::imread("test.jpg");
    std::vector<modeldeploy::vision::ObbResult> res;
    if (!obb->predict(im, &res)) return 1;
    for (const auto& r : res) {
        const auto& rb = r.rotated_box;
        std::printf("label=%d score=%.3f obb=(xc=%.1f, yc=%.1f, w=%.1f, h=%.1f, angle=%.3f)\n",
                    r.label_id, r.score, rb.xc, rb.yc, rb.width, rb.height, rb.angle);
    }

    // 4. 可视化：vis_obb；或 obb->draw_result(im, res, 0.5) 就地绘制（GPU 帧按设备分发）
    auto vis = modeldeploy::vision::vis_obb(im, res, 0.5, "msyh.ttc", 14, 0.3, false);
    vis.imwrite("obb_vis.jpg");
    const auto label_map = obb->get_label_map("names");

    // 5. 批量推理：各图统一 letterbox 后拼 batch，返回按图分组
    std::vector<modeldeploy::vision::ImageData> images = {
        modeldeploy::vision::ImageData::imread("a.jpg"),
        modeldeploy::vision::ImageData::imread("b.jpg"),
    };
    std::vector<std::vector<modeldeploy::vision::ObbResult>> ress;
    obb->batch_predict(images, &ress);

    // 6. 多线程：clone() 深拷贝独立实例
    auto obb2 = obb->clone();
    return 0;
}
```

## 10. 图像分类（classification::Classification）

分类模型（命名空间 `vision::classification`，`yolo11n-cls.onnx` 等，输入默认 {224, 224} + center crop）。单图返回**单个** `vision::ClassifyResult`（字段 `label_ids: std::vector<int32_t>`、`scores: std::vector<float>`，二者按序配对）。后处理经 `set_top_k`（默认 1）控制输出个数、`set_multi_label`（默认 false）切换逐维独立概率的多标签模式；另有 `set_multi_label_auto(true)` 按全类概率和自动判别单/多标签（显式 `set_multi_label` 优先）。

```cpp
#include "modeldeploy/vision.h"

int main() {
    // 1. 运行时选项 + 构造（详见上节）
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    opt.set_cpu_thread_num(4);
    auto cls = std::make_unique<modeldeploy::vision::classification::Classification>(
        "yolo11n-cls.onnx", opt);
    if (!cls->is_initialized()) return 1;

    // 2. 预处理/后处理参数
    cls->get_preprocessor().set_size({224, 224});            // 输入尺寸（默认 {224, 224}）
    cls->get_preprocessor().disable_center_crop();           // 关闭 center crop（默认开启）
    cls->get_postprocessor().set_top_k(5);                   // Top-K 输出个数（默认 1）
    cls->get_postprocessor().set_multi_label(false);         // 多标签模式（默认 false）

    // 3. 单图推理：输出单个 ClassifyResult（label_ids 与 scores 逐位配对）
    auto im = modeldeploy::vision::ImageData::imread("test.jpg");
    modeldeploy::vision::ClassifyResult res;
    if (!cls->predict(im, &res)) return 1;
    for (size_t i = 0; i < res.label_ids.size() && i < res.scores.size(); ++i) {
        std::printf("label=%d score=%.3f\n", res.label_ids[i], res.scores[i]);
    }

    // 4. 可视化：vis_cls(image, result, top_k, threshold, font_path, font_size, alpha, save_result)
    auto vis = modeldeploy::vision::vis_cls(im, res, 5, 0.35, "msyh.ttc", 14, 0.3, false);
    vis.imwrite("cls_vis.jpg");

    // 5. 批量推理：每图一个 ClassifyResult
    std::vector<modeldeploy::vision::ImageData> images = {
        modeldeploy::vision::ImageData::imread("a.jpg"),
        modeldeploy::vision::ImageData::imread("b.jpg"),
    };
    std::vector<modeldeploy::vision::ClassifyResult> ress;
    cls->batch_predict(images, &ress);

    // 6. 多线程：clone() 深拷贝独立实例
    auto cls2 = cls->clone();
    return 0;
}
```

## 11. OCR（vision::ocr::PaddleOCR + 子模型）

OCR 主流水线（命名空间 `vision::ocr`）：`PaddleOCR(det, cls, rec, dict, opt)` 串联文本检测 `DBDetector` → 方向分类 `Classifier` → 文本识别 `Recognizer`（`cls_model_path` 可传空串跳过方向分类）。单图 `predict(im, &res)` 输出**单个** `vision::OCRResult`：`boxes`（`std::vector<std::array<int, 8>>`，每行 4 点共 8 个 int，原图像素，按上下序排列）、`text`、`rec_scores`、`cls_labels`、`cls_scores` 逐行配对（det 未检出文本框时对整图直接识别，此时 `boxes` 为空）。

```cpp
#include "modeldeploy/vision.h"

int main() {
    namespace ocr_ns = modeldeploy::vision::ocr;
    // 1. 运行时选项 + 构造（det/cls/rec/dict 四路径，详见上节）
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    opt.set_cpu_thread_num(4);
    auto ocr = std::make_unique<ocr_ns::PaddleOCR>(
        "det.onnx", "cls.onnx", "rec.onnx", "dict.txt", opt);
    if (!ocr->is_initialized()) return 1;

    // 2. 参数设置（主流水线 batch + 经 get_* 子模型指针设置；括号内为默认值）
    ocr->set_cls_batch_size(6);                 // 方向分类子模型 batch（默认 6）
    ocr->set_rec_batch_size(8);                 // 识别子模型 batch（默认 8）
    ocr->get_detector()->get_preprocessor().set_max_side_len(960);      // 检测最长边（默认 960）
    ocr->get_detector()->get_postprocessor().set_det_db_thresh(0.3);        // DB 二值化阈值（默认 0.3）
    ocr->get_detector()->get_postprocessor().set_det_db_box_thresh(0.6);    // 框置信度阈值（默认 0.6）
    ocr->get_detector()->get_postprocessor().set_det_db_unclip_ratio(1.5);  // 扩框比例（默认 1.5）
    ocr->get_detector()->get_postprocessor().set_det_db_score_mode("slow"); // 框得分模式（默认 "slow"）
    ocr->get_detector()->get_postprocessor().set_use_dilation(0);           // 是否膨胀（默认 0）
    ocr->get_recognizer()->get_preprocessor().set_rec_image_shape({3, 48, 320});  // 识别输入形状（默认 {3, 48, 320}）
    ocr->get_classifier()->get_postprocessor().set_cls_thresh(0.9f);        // 方向分类阈值（默认 0.9）

    // 3. 单图推理：输出单个 OCRResult（text/boxes/rec_scores/cls_labels 逐行配对）
    auto im = modeldeploy::vision::ImageData::imread("test.jpg");
    modeldeploy::vision::OCRResult res;
    if (!ocr->predict(im, &res)) return 1;
    for (size_t i = 0; i < res.text.size(); ++i) {
        const auto& box = res.boxes[i];   // 4 点共 8 个 int
        std::printf("%s %.3f cls=%d box=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
                    res.text[i].c_str(), res.rec_scores[i], res.cls_labels[i],
                    box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]);
    }

    // 4. 可视化：vis_ocr(image, result, font_path, font_size=14, alpha=0.15, save_result=false)
    auto vis = modeldeploy::vision::vis_ocr(im, res, "msyh.ttc", 14, 0.3, false);
    vis.imwrite("ocr_vis.jpg");

    // 5. 批量推理：每图一个 OCRResult
    std::vector<modeldeploy::vision::ImageData> images = {
        modeldeploy::vision::ImageData::imread("a.jpg"),
        modeldeploy::vision::ImageData::imread("b.jpg"),
    };
    std::vector<modeldeploy::vision::OCRResult> ress;
    ocr->batch_predict(images, &ress);

    // 6. 子模型独立使用（也可不经 PaddleOCR 单独构造）
    ocr_ns::DBDetector db("det.onnx", opt);
    ocr_ns::Recognizer rec("rec.onnx", "dict.txt", opt);
    ocr_ns::Classifier clr("cls.onnx", opt);
    modeldeploy::vision::OCRResult det_res;
    db.predict(im, &det_res);                    // 文本框：det_res.boxes
    std::string text; float rec_score = 0.f;
    rec.predict(im, &text, &rec_score);          // 整行识别（text + score）
    int32_t cls_label = 0; float cls_score = 0.f;
    clr.predict(im, &cls_label, &cls_score);     // 方向分类（0°/180°）

    // 7. 多线程：clone() 深拷贝独立实例（主流水线三个子模型一起深拷贝）
    auto ocr2 = ocr->clone();
    return 0;
}
```

## 12. OCR 进阶（版面 / 表格 / 公式 / 文档转 Markdown）

命名空间 `vision::ocr` 下四件套，见 [models.md §8.3/§8.4](../models.md#8-ocr文字识别) 与 [models.md §17 文档理解](../models.md#17-文档理解document-understanding--markdown):版面分析 `StructureV2Layout`、表格结构 `StructureV2Table` / 表格流水线 `PPStructureV2Table`、公式 `FormulaRecognizer`，以及组合器 `DocToMarkdown`（`set_layout/set_ocr/set_table/set_formula` → `predict` 输出 Markdown 字符串）。

```cpp
#include "modeldeploy/vision.h"

int main() {
    namespace ocr_ns = modeldeploy::vision::ocr;
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    opt.set_cpu_thread_num(4);

    // 1. 版面分析：StructureV2Layout(model, opt)
    ocr_ns::StructureV2Layout layout("layout.onnx", opt);
    auto& layout_pre = layout.get_preprocessor();
    layout_pre.set_layout_image_shape({3, 800, 608}); // 输入 c,h,w（默认 {3, 800, 608}）
    layout_pre.set_static_shape_infer(true);          // 静态输入形状（默认 true）
    auto& layout_post = layout.get_postprocessor();
    layout_post.set_score_threshold(0.4f);            // 置信度阈值（默认 0.4）
    layout_post.set_nms_threshold(0.5f);              // NMS IoU 阈值（默认 0.5）
    auto im = modeldeploy::vision::ImageData::imread("doc.jpg");

    std::vector<modeldeploy::vision::DetectionResult> layout_res;
    layout.predict(im, &layout_res);                  // -> CDLA 版面区域
    for (const auto& b : layout_res)
        std::printf("label=%d score=%.3f box=(%.0f %.0f %.0f %.0f)\n",
                    b.label_id, b.score, b.box.x, b.box.y, b.box.width, b.box.height);

    // 2. 表格结构：StructureV2Table(model, table_dict, opt) -> OCRResult(table_html/structure)
    ocr_ns::StructureV2Table table("table.onnx", "table_dict.txt", opt);
    modeldeploy::vision::OCRResult table_res;
    table.predict(im, &table_res);                    // 输出 table_html / table_structure
    std::printf("%s\n", table_res.table_html.c_str());

    // 3. 端到端表格：PPStructureV2Table(det, rec, table, rec_dict, table_dict, ...)
    ocr_ns::PPStructureV2Table ppt(
        "det.onnx", "rec.onnx", "table.onnx",
        "rec_dict.txt", "table_dict.txt", 960, 0.3, 0.6, 1.5, "slow", false, 8, opt);
    ppt.set_rec_batch_size(8);                        // 识别子模型 batch（默认 8）
    modeldeploy::vision::OCRResult ocr_res;
    ppt.predict(im, &ocr_res);                        // text + table_html 单元格内容
    for (size_t i = 0; i < ocr_res.text.size(); ++i)
        std::printf("%s %.3f\n", ocr_res.text[i].c_str(), ocr_res.rec_scores[i]);

    // 4. 公式识别：FormulaRecognizer(model, dict, opt) -> std::string（LaTeX）
    ocr_ns::FormulaRecognizer formula("formula.onnx", "dict.txt", opt);
    auto eq = modeldeploy::vision::ImageData::imread("equation.jpg");
    std::string latex;
    if (formula.predict(eq, &latex)) std::printf("%s\n", latex.c_str());

    // 5. 文档转 Markdown：DocToMarkdown 组合器（set_* 为借用指针，调用方须保持子模型存活）
    ocr_ns::PaddleOCR ocr("det.onnx", "cls.onnx", "rec.onnx", "dict.txt", opt);
    ocr_ns::DocToMarkdown doc;
    doc.set_layout(&layout);                          // 版面（必须）
    doc.set_ocr(&ocr);                                // 文本区 OCR
    doc.set_table(&ppt);                              // 表格区
    doc.set_formula(&formula);                        // 公式区（可省）
    std::string markdown;
    if (doc.ready() && doc.predict(im, &markdown))
        std::printf("%s\n", markdown.c_str());        // 公式 $...$、表格 HTML
    return 0;
}
```

> `DocToMarkdown` 为**单列自上而下**顺序排版（不做多栏重排）；`set_*` 同时提供 `std::unique_ptr` 所有权重载与 `T*` 借用重载（本例为借用，须保证子模型生命周期覆盖 `doc` 使用期间）。

## 13. 人脸（face::Scrfd / SeetaFace 族 / FaceRecognizerPipeline）

人脸模块在命名空间 `modeldeploy::vision::face`：`Scrfd`（检测，框+5 关键点）、`SeetaFaceID`（特征）、`SeetaFaceAge`（年龄）、`SeetaFaceGender`（性别）、`SeetaFaceAsFirst`/`SeetaFaceAsSecond`（防伪一/二阶段）、`SeetaFaceAsPipeline`（防伪串联）、`FaceRecognizerPipeline`（检测+特征一体化）。完整说明见 [models.md §6](../models.md#6-人脸face)。

**注意**：`Scrfd` 与姿态族同用 `ScrfdPreprocessor`+`ScrfdPostprocessor`（`set_size`/`set_padding_value`/`set_scale_up`/`set_mini_pad`/`set_stride`；`set_conf_threshold`/`set_nms_threshold`/`set_landmarks_per_face`），**不是**检测族的 preprocessor；`SeetaFaceID/Age/Gender` 的 preprocessor 仅 `set_size`、postprocessor **无参数**（不同构于 det）。

```cpp
#include "modeldeploy/vision.h"

int main() {
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    opt.set_cpu_thread_num(4);

    // 1. 人脸检测：face::Scrfd -> std::vector<KeyPointsResult>（框 + 5 关键点）
    auto det = std::make_unique<modeldeploy::vision::face::Scrfd>("scrfd.onnx", opt);
    if (!det->is_initialized()) return 1;
    det->get_preprocessor().set_size({640, 640});      // letterbox 输入尺寸（默认 {640, 640}）
    det->get_postprocessor().set_conf_threshold(0.30f); // 置信度阈值（默认 0.25）
    det->get_postprocessor().set_nms_threshold(0.45f);  // NMS IoU 阈值（默认 0.5）
    det->get_postprocessor().set_landmarks_per_face(5); // 每人脸关键点（默认 5）

    auto im = modeldeploy::vision::ImageData::imread("test.jpg");
    std::vector<modeldeploy::vision::KeyPointsResult> faces;
    if (!det->predict(im, &faces)) return 1;
    for (const auto& r : faces) {
        std::printf("score=%.3f box=(%.0f, %.0f, %.0f, %.0f) kps=%zu\n",
                    r.score, r.box.x, r.box.y, r.box.width, r.box.height, r.keypoints.size());
        for (const auto& kp : r.keypoints)
            std::printf("  kp=(%.1f, %.1f)\n", kp.x, kp.y);
    }

    // 2. 年龄 / 性别（输入对齐后的人脸裁剪图）：predict(image, &v) 输出 int
    modeldeploy::vision::face::SeetaFaceAge age("age.onnx", opt);
    modeldeploy::vision::face::SeetaFaceGender gender("gender.onnx", opt);
    int a = 0, g = 0;
    if (age.predict(im, &a) && gender.predict(im, &g))
        std::printf("age=%d gender=%d (%s)\n", a, g, g == 0 ? "女" : "男");

    // 3. 人脸识别（特征）：face::SeetaFaceID -> FaceRecognitionResult{embedding}
    modeldeploy::vision::face::SeetaFaceID rec("rec.onnx", opt);
    modeldeploy::vision::FaceRecognitionResult emb;
    rec.predict(im, &emb);
    std::printf("embedding dim=%zu\n", emb.embedding.size());

    // 4. 防伪一 / 二阶段
    modeldeploy::vision::face::SeetaFaceAsFirst af("first.onnx", opt);
    float score = 0.f;
    af.predict(im, &score);                 // 活体得分
    modeldeploy::vision::face::SeetaFaceAsSecond as("second.onnx", opt);
    std::vector<std::tuple<int, float>> probs;
    as.predict(im, &probs);                 // label + 概率

    // 5. 防伪流水线：det|first|second 三模型，predict(im, &res, fuse=0.8, clarity=0.3)
    modeldeploy::vision::face::SeetaFaceAsPipeline as_pipe(
        "det.onnx", "first.onnx", "second.onnx", opt);
    std::vector<modeldeploy::vision::FaceAntiSpoofResult> spoofs;
    as_pipe.predict(im, &spoofs, 0.8f, 0.3f);   // 枚举 REAL / FUZZY / SPOOF

    // 6. 识别流水线：det|rec 两模型，输出 vector<FaceRecognitionResult>
    modeldeploy::vision::face::FaceRecognizerPipeline pipe("det.onnx", "rec.onnx", opt);
    pipe.get_detector()->get_postprocessor().set_conf_threshold(0.30f);  // 调检测子模型阈值
    std::vector<modeldeploy::vision::FaceRecognitionResult> embeddings;
    pipe.predict(im, &embeddings);
    // 只取主脸：modeldeploy::vision::FaceRecognitionResult maxface;
    // pipe.predict_max_face(im, &maxface, &face_count);

    // 7. 多线程：clone() 深拷贝独立实例（各模型均支持）
    auto det2 = det->clone();
    return 0;
}
```

## 14. InsightFace 全流程（face::InsightFaceAnalysis + 子模型）

InsightFace Buffalo 全家桶（det + 2d106 + 3d68 + recognition）组合成一次 `analyze`，输出检测框/5 关键点/106/68 点/姿态/特征，适合人脸注册与比对。详细语义见 [models.md §6.1](../models.md#61-insightfacebuffalo-系列全流程)。

```cpp
#include "modeldeploy/vision.h"

int main() {
    // 1. 从 buffalo_l 模型目录加载全流水线
    //    （create_from_dir 自动拼出 det_10g / w600k_r50 / 2d106det / 1k3d68 / genderage）
    auto analysis = modeldeploy::vision::face::InsightFaceAnalysis::create_from_dir(
        "test_data/test_models/onnx/insightface/buffalo_l");
    if (!analysis || !analysis->is_initialized()) return 1;

    // 或显式构造：InsightFaceAnalysis(det, rec, lmk2d, lmk3d, option, genderage="")
    // auto analysis = std::make_unique<modeldeploy::vision::face::InsightFaceAnalysis>(
    //     "det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx", "1k3d68.onnx",
    //     modeldeploy::RuntimeOption(), "genderage.onnx");

    analysis->set_det_thresh(0.5f);   // 检测阈值（默认 0.5）

    auto im = modeldeploy::vision::ImageData::imread("test.jpg");
    std::vector<modeldeploy::vision::face::InsightFaceResult> res;
    if (!analysis->analyze(im, &res)) return 1;
    for (const auto& r : res) {
        std::printf("det=%.3f bbox=(%.0f,%.0f,%.0f,%.0f)\n",
                    r.det_score, r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3]);
        // r.bbox：[x1,y1,x2,y2]（原图坐标）；r.det_score：检测置信度
        // r.kps：5 关键点；r.landmark_2d_106：106 个 2D 点；r.landmark_3d_68：68 个 3D 点
        // r.pose：[pitch,yaw,roll]；r.embedding：512 维特征；r.gender / r.age（-1 表示未启用 genderage）
        std::printf("kps=%zu l2d=%zu l3d=%zu emb=%zu gender=%d age=%d\n",
                    r.kps.size(), r.landmark_2d_106.size(),
                    r.landmark_3d_68.size(), r.embedding.size(), r.gender, r.age);
    }
    // analyze(image, &res, with_2d106=true, with_3d68=true, with_recognition=true,
    //         with_genderage=true, max_face_only=false)
    // 只取主脸：analysis->analyze_max_face(im, &one, ...)

    // 2. 仅检测：输出 vector<InsightFaceBox>（bbox/kps/score）
    std::vector<modeldeploy::vision::face::InsightFaceBox> boxes;
    analysis->detect(im, &boxes);

    // 3. 子模型独立使用（均位于 modeldeploy::vision::face，继承 BaseModel）
    modeldeploy::vision::face::InsightFaceDet det("det_10g.onnx");
    std::vector<modeldeploy::vision::face::InsightFaceBox> det_res;
    det.predict(im, &det_res);

    modeldeploy::vision::face::InsightFaceLandmark lmk("2d106det.onnx");
    std::vector<std::array<float, 2>> pts_2d;
    lmk.predict_2d106(im, det_res[0].bbox, &pts_2d);               // 106 个 2D 点
    std::vector<std::array<float, 3>> pts_3d;
    std::array<float, 3> pose{};
    lmk.predict_3d68(im, det_res[0].bbox, &pts_3d, &pose);         // 68 个 3D 点 + 姿态

    modeldeploy::vision::face::InsightFaceRecognition rec("w600k_r50.onnx");
    std::vector<float> emb;
    rec.predict(im, det_res[0].kps, &emb);                         // 512 维特征（需 5 关键点）

    modeldeploy::vision::face::InsightFaceGenderAge ga("genderage.onnx");
    modeldeploy::vision::face::GenderAgeResult ga_res;
    ga.predict_gender_age(im, det_res[0].bbox, &ga_res);           // ga_res.gender / ga_res.age
    return 0;
}
```

> 注意：C++（与 Python）是仅有的两类能拿到 `landmark_2d_106` / `landmark_3d_68` 的绑定；C API / C# / Rust 仅暴露 bbox/kps/embedding/pose/gender/age。

## 15. 车牌 LPR（vision::lpr::LprPipeline / LprDetection / LprRecognizer）

车牌识别在命名空间 `modeldeploy::vision::lpr`：主流水线 `LprPipeline`（检测 det + 识别 rec 一体化），子模型 `LprDetection`（Locate 车牌位置，输出框 + 4 角点）与 `LprRecognizer`（识别车牌字符/颜色）。主流水线结果类型 `vision::LprResult`（字段 `box: Rect2f`、`keypoints: vector<Point3f>`（车牌 4 角点，`z` 恒为 0）、`label_id: int32_t`、`score: float`、`car_plate_str: string`、`car_plate_color: string`）。注意：`LprResult` 的关键点字段名为 `keypoints`（Python 绑定为 `landmarks`）。

```cpp
#include "modeldeploy/vision.h"

int main() {
    namespace lpr_ns = modeldeploy::vision::lpr;
    // 1. 运行时选项 + 构造（详见上节）
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    opt.set_cpu_thread_num(4);

    // 2. 主流水线：LprPipeline(det, rec, opt) 检测 + 识别一体化
    lpr_ns::LprPipeline lpr("det.onnx", "rec.onnx", opt);
    if (!lpr.is_initialized()) return 1;

    auto im = modeldeploy::vision::ImageData::imread("test.jpg");
    std::vector<modeldeploy::vision::LprResult> results;
    if (!lpr.predict(im, &results)) return 1;
    for (const auto& r : results) {
        std::printf("%s %s score=%.3f label=%d box=(%.0f, %.0f, %.0f, %.0f) kps=%zu\n",
                    r.car_plate_str.c_str(), r.car_plate_color.c_str(), r.score, r.label_id,
                    r.box.x, r.box.y, r.box.width, r.box.height, r.keypoints.size());
        for (const auto& kp : r.keypoints)     // 车牌 4 角点（z 恒为 0）
            std::printf("  kp=(%.1f, %.1f)\n", kp.x, kp.y);
    }

    // 3. 可视化：vis_lpr(image, result, font_path, font_size, landmark_radius, alpha, save)
    //    （绘制车牌框 + 字符 + 4 角点连线）
    auto vis = modeldeploy::vision::vis_lpr(im, results, "msyh.ttc", 14, 4, 0.3, false);
    vis.imwrite("lpr_vis.jpg");

    // 4. 多线程：clone() 深拷贝独立实例（每线程持有一个，互不干扰）
    auto lpr2 = lpr.clone();

    // 5. 子模型独立使用（也可不经 LprPipeline 单独构造）
    lpr_ns::LprDetection det("det.onnx", opt);
    if (!det.is_initialized()) return 1;
    det.get_preprocessor().set_size({640, 640});       // letterbox 输入尺寸（默认 {640, 640}）
    det.get_postprocessor().set_conf_threshold(0.25f); // 置信度阈值（默认 0.25）
    det.get_postprocessor().set_nms_threshold(0.45f);  // NMS IoU 阈值（默认 0.5）
    det.get_postprocessor().set_landmarks_per_card(4); // 每车牌角点数（默认 4）
    std::vector<modeldeploy::vision::KeyPointsResult> boxes;
    if (!det.predict(im, &boxes)) return 1;            //  输出框 + 4 角点
    for (const auto& r : boxes)
        std::printf("score=%.3f box=(%.0f, %.0f, %.0f, %.0f) kps=%zu\n",
                    r.score, r.box.x, r.box.y, r.box.width, r.box.height, r.keypoints.size());

    lpr_ns::LprRecognizer rec("rec.onnx", opt);        // 输入车牌裁剪图，输出单个 LprResult
    if (!rec.is_initialized()) return 1;
    auto crop = modeldeploy::vision::ImageData::imread("plate_crop.jpg");
    modeldeploy::vision::LprResult plate;
    if (!rec.predict(crop, &plate)) return 1;
    std::printf("rec: %s %s %.3f\n", plate.car_plate_str.c_str(),
                plate.car_plate_color.c_str(), plate.score);
    return 0;
}
```

> 与 `Scrfd`/姿态族类似，`LprDetection` 的 `markers` 后处理参数名是 `landmarks_per_card`（每车牌角点数），**不是** `landmarks_per_face`；`LprRecPreprocessor` 仅 `set_size`（默认 `{168, 48}`），`LprRecPostprocessor` **无参数**（字符/颜色内联解码，字符表含 78 类）。

## 16. 行人属性 + 行人 ReID（vision::PedestrianAttribute / reid::ReID / reid::ReIdGallery）

行人属性 `modeldeploy::vision::PedestrianAttribute`（`vision.h`）把检测 + 多标签属性分类串联成流水线，`predict` 输出每个人的 `AttributeResult`（字段 `box: Rect2f`、`box_label_id: int32_t`、`box_score: float`、`attr_scores: vector<float>`）；行人 ReID 在 `modeldeploy::vision::reid` 命名空间：`reid::ReID`（OSNet）提取 L2 归一化 512-d 特征，配合内存 `reid::ReIdGallery` 做注册与 top-k 余弦匹配。

```cpp
#include "modeldeploy/vision.h"
#include "vision/reid/reid.h"
#include "vision/reid/gallery.h"

int main() {
    // 1. 运行时选项（详见上节）
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    opt.set_cpu_thread_num(4);

    auto im = modeldeploy::vision::ImageData::imread("test.jpg");
    auto crop = modeldeploy::vision::ImageData::imread("person_crop.jpg");

    // 2. 行人属性：PedestrianAttribute(det, cls, opt) 检测 + 多标签属性分类串联
    modeldeploy::vision::PedestrianAttribute attr("det.onnx", "cls.onnx", opt);
    if (!attr.is_initialized()) return 1;
    attr.set_det_threshold(0.5f);            // 检测阈值（默认 0.5）
    attr.set_det_input_size({1280, 1280});   // 检测子模型输入尺寸（默认 {640, 640}）
    attr.set_cls_input_size({192, 256});     // 分类子模型输入尺寸（默认 {192, 256}）
    attr.set_cls_batch_size(8);              // 分类子模型 batch（默认 8），>0 固定 / -1 自动
    // 也可经 get_detector() / get_classifier() 链式细调
    std::vector<modeldeploy::vision::AttributeResult> attrs;
    if (!attr.predict(im, &attrs)) return 1;
    for (const auto& r : attrs) {
        std::printf("box=(%.0f, %.0f, %.0f, %.0f) label=%d score=%.3f attrs=%zu\n",
                    r.box.x, r.box.y, r.box.width, r.box.height,
                    r.box_label_id, r.box_score, r.attr_scores.size());
        for (float s : r.attr_scores) std::printf("  %.3f\n", s);
    }
    // 批量：attr.batch_predict({im, ...}, &batch)；多线程：attr.clone()
    // 可视化：vis_attr(image, result, threshold, label_map, font_path, font_size, alpha,
    //                  save, abnormal_ids, show_attr)
    auto vis = modeldeploy::vision::vis_attr(im, attrs, 0.5, {}, "msyh.ttc", 14, 0.3, false, {}, true);
    vis.imwrite("attr_vis.jpg");

    // 3. 行人 ReID：reid::ReID(model, opt) 提取 L2 归一化 512-d 特征（输入行人裁剪图）
    modeldeploy::vision::reid::ReID reid("osnet.onnx", opt);
    if (!reid.is_initialized()) return 1;
    std::vector<modeldeploy::vision::ReIdResult> res;
    if (!reid.predict(crop, &res) || res.empty() || res[0].embedding.empty()) return 1;
    const auto& emb = res[0].embedding;      // 已 L2 归一化

    // 4. reid::ReIdGallery：内存行人库，注册 + 余弦 top-k 匹配（同 label 覆盖）
    modeldeploy::vision::reid::ReIdGallery gallery;
    gallery.enroll("a", emb);                // label + embedding
    gallery.enroll("b", emb);
    std::cout << "gallery size=" << gallery.size() << "\n";
    for (auto& [label, score] : gallery.match(emb, 1))   // -> vector<pair<label, score>> 降序
        std::printf("match -> label=%s score=%.3f\n", label.c_str(), score);
    gallery.remove("b");                     // -> vector<bool>；gallery.clear() 清空
    return 0;
}
```

> `ReID::predict` 输出即已 L2 归一化；`ReIdGallery::match` 直接对其做点积求余弦（`enroll`/`match` 的 embedding 均要求已归一化）。

## 17. 条码 / 二维码（vision::barcode::BarcodeDetector）

条码识别器 `modeldeploy::vision::barcode::BarcodeDetector`（`vision/barcode/barcode.h`）是**纯 CV**（基于 ZXing）、无模型依赖，CPU 上即可解码条码 / 二维码。默认解码全部格式（`FMT_ALL`），可用 `set_formats` 限定 `FMT_*` 位或子集。

```cpp
#include "modeldeploy/vision.h"
#include "vision/barcode/barcode.h"

int main() {
    modeldeploy::vision::barcode::BarcodeDetector det;
    det.set_formats(modeldeploy::vision::barcode::FMT_QR_CODE |
                    modeldeploy::vision::barcode::FMT_EAN_13);   // 限定 QR + EAN-13

    auto im = modeldeploy::vision::ImageData::imread("qrcode.jpg");
    auto codes = det.detect(im);   // std::vector<barcode::BarcodeResult>
    for (const auto& r : codes) {
        // r.text: 解码文本/URL；r.format: "QR_CODE"/"EAN_13"/"CODE_128"/...
        // r.score: 可信度 [0,1]；r.is_qr: 是否二维码；r.quad: std::array<Point2f,4>
        std::printf("%s [%s] score=%.3f is_qr=%d\n", r.text.c_str(), r.format.c_str(), r.score, r.is_qr);
        std::printf("quad: ");
        for (const auto& p : r.quad) std::printf("(%.0f,%.0f) ", p.x, p.y);
        std::printf("\n");
    }
    det.set_formats(modeldeploy::vision::barcode::FMT_ALL);   // 恢复全部格式
    return 0;
}
```

## 18. 多目标跟踪（tracking::ByteTracker / BotSortTracker / StrongSortTracker）

三类 MOT 跟踪器均在 `modeldeploy::vision::tracking` 命名空间，**纯 CPU**、无模型依赖，跟踪 ID 跨帧稳定，`reset()` 归零。用法一致：每帧把检测框转成 `tracking::Detection`（`box: Rect2f`、`score: float`、`label_id: int`、`feature: vector<float>`），再 `update` 推进一帧，返回 `vector<tracking::TrackResult>`（`track_id` 跨帧关联同一目标；`state` 为 `TrackState` 枚举 `New=0 / Tracked=1 / Lost=2 / Removed=3`）。

```cpp
#include "modeldeploy/vision.h"
#include "vision/detection/ultralytics_det.h"
#include "vision/tracking/bytetrack.h"      // ByteTracker
#include "vision/tracking/botsort.h"        // BotSortTracker
#include "vision/tracking/strongsort.h"     // StrongSortTracker

int main() {
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend(); opt.use_cpu();

    // 1. 构造（无参）与参数（各 tracker 的 set_params 签名不同，见下）
    modeldeploy::vision::tracking::ByteTracker tracker;
    tracker.set_params(0.5f, 0.5f, 0.1f, 30, 3, 0.3f);

    // BoT-SORT 额外参数：match_thresh=0.8, fuse_score_weight=0.5, ema_alpha=0.9, with_cmc=true
    // modeldeploy::vision::tracking::BotSortTracker bs;
    // bs.set_params(0.5f, 0.5f, 0.1f, 30, 3, 0.3f, 0.8f, 0.5f, 0.9f, true);

    // StrongSORT 额外参数：match_thresh=0.8, ema_alpha=0.9, appearance_priority=0.7, with_cmc=true
    // modeldeploy::vision::tracking::StrongSortTracker ss;
    // ss.set_params(0.5f, 0.5f, 0.1f, 30, 3, 0.3f, 0.8f, 0.9f, 0.7f, true);

    modeldeploy::vision::detection::UltralyticsDet det("yolo11n.onnx", opt);

    for (int f = 0; f < 100; ++f) {   // 假定逐帧读取视频，这里用帧索引示意
        auto frame = modeldeploy::vision::ImageData::imread("frame_%03d.jpg");
        std::vector<modeldeploy::vision::DetectionResult> boxes;
        if (!det.predict(frame, &boxes)) break;

        // 2. 每帧：检测框 -> tracking::Detection
        std::vector<modeldeploy::vision::tracking::Detection> dets;
        for (const auto& r : boxes) {
            modeldeploy::vision::tracking::Detection d;
            d.box = r.box; d.score = r.score; d.label_id = r.label_id;
            // d.feature = {...};   // 可选：BoT-SORT / StrongSORT 外观特征（ReID）
            dets.push_back(d);
        }

        // 3. 推进一帧：update(detections, frame=nullptr, timestamp=-1)
        auto tracks = tracker.update(dets, nullptr, -1.0);
        for (const auto& t : tracks)
            // t.track_id: 跨帧稳定 ID；t.state: TrackState；其余同 Detection
            std::printf("id=%d state=%d box=(%.0f,%.0f,%.0f,%.0f) score=%.3f\n",
                        t.track_id, t.state, t.box.x, t.box.y, t.box.width, t.box.height, t.score);
    }
    tracker.reset();   // 清空内部状态，ID 重新从 0 计
    return 0;
}
```

## 19. CV 解决方案 + 工具（`vision::solution` / `vision::tool`）

解决方案层（`modeldeploy::vision::solution`）把**检测 + 跟踪**组合成可交付的业务功能，均为纯后处理 / 统计逻辑，逐帧喂入 `std::vector<tracking::TrackResult>` 即得业务结果；工具层（`modeldeploy::vision::tool`）提供检测容器 `Detections`、区域判断 `LineZone`/`PolygonZone`、评估与 IoU/NMS 等公共底座。完整清单见 [solutions.md](../solutions.md) 与 [tools.md](../tools.md)。

```cpp
#include "modeldeploy/vision.h"
#include "vision/solutions/object_counter.h"
#include "vision/solutions/heatmap.h"
#include "vision/solutions/region_counter.h"
#include "vision/solutions/queue_manager.h"
#include "vision/solutions/track_zone.h"
#include "vision/solutions/speed_estimator.h"
#include "vision/solutions/parking_manager.h"
#include "vision/solutions/fall_detector.h"
#include "vision/solutions/workout_monitor.h"
#include "vision/solutions/distance_estimator.h"

using namespace modeldeploy;
namespace sol = modeldeploy::vision::solution;

int main() {
    // 1. 人流统计：跨线/区域进出 + 类别统计
    sol::ObjectCounter cnt;
    cnt.set_line({0.f, 0.f}, {100.f, 100.f});                 // 计数线两点
    cnt.set_region({Point2f(0,0), Point2f(100,0), Point2f(100,240), Point2f(0,240)});
    cnt.set_classes({0});                                     // 类别白名单（空=全部）
    cnt.update(tracks);                                       // 每帧喂入 TrackResult
    auto s = cnt.stats();    // s.line_in / s.line_out / s.class_count
    int n = cnt.region_count();

    // 2. 热力图：set_size 低分辨率栅格，peak 峰值，heat_at 取栅格值
    sol::Heatmap hm;
    hm.set_size(320, 240);
    hm.update(tracks, frame_w, frame_h);
    auto peak = hm.peak();    float v = hm.heat_at(peak.first, peak.second);

    // 3. 多区域逐帧计数：add_region 命名区域 -> region_counts()
    sol::RegionCounter rc;
    rc.add_region("doorA", {Point2f(0,0), Point2f(80,0), Point2f(80,240), Point2f(0,240)});
    rc.update(tracks);
    for (auto& [name, c] : rc.region_counts()) printf("%s=%d\n", name.c_str(), c);
    size_t total = rc.total_regions();

    // 4. 排队：单区域当前帧排队长度
    sol::QueueManager q;
    q.set_region({Point2f(100,0), Point2f(160,0), Point2f(160,240), Point2f(100,240)});
    q.update(tracks);    int qlen = q.queue_count();

    // 5. 追踪区域：只保留区域内目标并计数
    sol::TrackZone tz;
    tz.set_region({Point2f(100,0), Point2f(160,0), Point2f(160,240), Point2f(100,240)});
    tz.update(tracks);    auto inside = tz.inside_tracks();   int in = tz.inside_count();

    // 6. 测速：像素->米比例需标定；update 带毫秒时间戳
    sol::SpeedEstimator sp;
    sp.set_meter_per_pixel(0.05f);
    sp.update(tracks, ts_ms);    auto mps = sp.speeds_m_s();   // track_id -> 米/秒

    // 7. 停车：set_slots 车位多边形列表 -> occupancy()
    sol::ParkingManager pk;
    pk.set_slots({{Point2f(0,0),Point2f(40,0),Point2f(40,40),Point2f(0,40)},
                  {Point2f(50,0),Point2f(90,0),Point2f(90,40),Point2f(50,40)}});
    pk.update(tracks);    auto occ = pk.occupancy();          // [bool, ...]

    // 8. 跌倒：update 喂入关键点结果 -> FallResult{state, confidence}
    sol::FallDetector fd;
    auto fr = fd.update(persons);   // state: Standing=0/PreFall=1/Fallen=2

    // 9. 锻炼计数：夹角阈值状态机（min_deg/max_deg）；angle 为三点二维夹角(度)
    sol::WorkoutMonitor wm(70.0f, 160.0f);
    float deg = sol::WorkoutMonitor::angle(a, b, c);
    wm.update(deg);    int reps = wm.reps();

    // 10. 距离估计：两两质心距离（米，需标定 mpp）
    sol::DistanceEstimator de;
    de.set_meter_per_pixel(0.05f);
    auto dm = de.pair_distances_m(tracks);   // { {id_a,id_b}, 距离 }

    // 工具层：modeldeploy::vision::tool
    namespace tool = modeldeploy::vision::tool;
    tool::Detections d = tool::from_track(tracks);            // 跟踪 -> 检测容器
    tool::nms(d, 0.5f);                                       // 就地 NMS
    float iou = tool::iou(d.boxes[0], d.boxes[1]);
    tool::LineZone lz(Point2f(160,0), Point2f(160,240));      // 跨线触发
    if (lz.trigger(Point2f(cx, cy))) ++crossed;
    tool::PolygonZone zone({Point2f(0,0), Point2f(320,0), Point2f(320,240), Point2f(0,240)});
    bool in_zone = zone.contains(Point2f(100,100));
    return 0;
}
```

> C++ 解决方案/工具命名空间为 `modeldeploy::vision::solution` 与 `modeldeploy::vision::tool`（注意 `solution` 为单数）。以上方法与 Python `md.vision.solutions`/`md.vision.tools`、C API `md_solution_*` 对应；`FallDetector`/`WorkoutMonitor`/`DistanceEstimator`/`SpeedEstimator`/`ParkingManager` 为 C++ 完整实现，其部分绑定（C API/C#/Rust）仅创建、无逐帧查询接口，见各语言章节标注。

## 20. 更多模型（均使用同一 `RuntimeOption`）

| 能力 | 类 | 用法 |
|------|----|------|
| 目标检测 | `vision::detection::UltralyticsDet` | 见上文 §3 |
| 实例分割 | `vision::detection::UltralyticsSeg` | 见上文 §4 |
| 轻量分割一切 | `vision::seg::FastSam` | 见上文 §5 |
| 语义分割 | `vision::detection::UltralyticsSem` | 见上文 §6 |
| 深度估计 | `vision::detection::UltralyticsDepth` | 见上文 §7 |
| 姿态 / 关键点 | `vision::detection::UltralyticsPose` / `vision::hand::HandKeypoint` / `vision::landmark::VehicleKeypoint` / `vision::landmark::FaceLandmark` | 见上文 §8 |
| 旋转框 | `vision::detection::UltralyticsObb` | 见 [models-旋转框](../models.md#4-旋转框检测oriented-bounding-box) |
| 分类 | `vision::Classification` | 见 [models-分类](../models.md#5-图像分类classification) |
| OCR | `vision::ocr::PaddleOCR` | 见上文 §11 |
| 人脸 | `vision::face::Scrfd` / `InsightFaceAnalysis` | 见 [models-人脸](../models.md#6-人脸face) |
| 车牌 | `vision::lpr::LprPipeline` | 见上文 §15 |
| 行人属性 | `vision::PedestrianAttribute` | 见上文 §16 |
| 行人 ReID | `vision::reid::ReID` + `reid::ReIdGallery` | 见上文 §16 |
| 条码 / 二维码 | `vision::barcode::BarcodeDetector` | 见上文 §17 |
| 多目标跟踪 | `vision::tracking::ByteTracker` / `BotSortTracker` / `StrongSortTracker` | 见上文 §18 |
| ASR | `audio::asr::SenseVoice` | 见下文 §21 与 [models-语音](../models.md#10-语音识别asr) |
| TTS（Kokoro） | `audio::tts::Kokoro` | 见下文 §21 与 [models-TTS](../models.md#11-语音合成tts) |

> 各类模型完整 API 见 [models.md](../models.md)；后端/设备切换见 [RuntimeOption](../runtime_option.md) 与 [后端详解](../backends.md)。

## 21. 音频模型（`audio::tts::Kokoro` / `audio::asr::SenseVoice` / `audio::speaker`）

`modeldeploy::audio` 命名空间提供语音合成（`tts::Kokoro`，24kHz）、语音识别（`asr::SenseVoice`，16k PCM）与说话人验证/声纹库（`speaker_verify::SpeakerVerify` + `SpeakerGallery`）。完整示例见 `examples/demo_audio/` 与 `examples/demo_speaker/`。

### TTS：`audio::tts::Kokoro`（24kHz）

Kokoro 继承 `audio::tts::ITtsModel`，提供 `predict` / `predict_stream` / `get_sample_rate`。`predict_stream` 的 `cb` 签名 `bool(const float*, int, float progress)`，返回 `false` 中止；`chunk_frames == 0` 等价一次性合成。`chunk_frames` 单位为 **UTF-8 字符数**（>120 字触发多块），上层应传相对小的值以观察多次音频回调（如 120）。

```cpp
modeldeploy::RuntimeOption option;
option.use_ort_backend(); option.use_cpu();

// Kokoro：24kHz；voice 来自 {model_dir}/voices/
modeldeploy::audio::tts::Kokoro kokoro("kokoro.onnx", "tokens.txt",
    {"lexicon-us-en.txt", "lexicon-zh.txt"}, "voices.bin", "dict/", "", option);
std::vector<float> wav;
kokoro.predict("你好，世界。", "zf_001", 1.0f, &wav);

// 统一流式合成
kokoro.predict_stream("你好，世界。", "zf_001", 1.0f, 120,
    [](const float* samples, int n, float progress) -> bool { return true; });
```

### ASR：`audio::asr::SenseVoice`（16k）

```cpp
modeldeploy::audio::asr::SenseVoice sv("sense_voice.onnx", "tokens.txt", option);

std::string text;                    // 纯净文本
sv.predict(pcm16k, &text);

modeldeploy::audio::asr::SenseVoiceResult r;   // 结构化：language/emotion/event/task/itn/nospeech
sv.predict(pcm16k, &r);
std::printf("%s | %s | %s | %s | %s | itn=%d nospeech=%d\n",
            r.text.c_str(), r.language.c_str(), r.emotion.c_str(),
            r.event.c_str(), r.task.c_str(), r.itn, r.nospeech);
```

### 说话人：`audio::speaker_verify::SpeakerVerify` + `audio::SpeakerGallery`

```cpp
using modeldeploy::audio::speaker_verify::SpeakerVerify;
SpeakerVerify sp("ecapa.onnx", option);          // 192-d 声纹 embedding
std::vector<float> emb;
sp.predict(pcm16k, &emb);

modeldeploy::audio::SpeakerGallery gal;          // 纯内存声纹库（label -> l2 归一化 embedding）
gal.enroll("alice", emb);
for (auto& [label, score] : gal.match(emb, 1))   // top-k 余弦匹配
    std::printf("%s %.4f\n", label.c_str(), score);
gal.size(); gal.remove("alice"); gal.clear();
```

## 22. 音频解决方案 + 工具（`audio::solution` / `audio::tool`）

`modeldeploy::audio::solution` 提供说话人检索 `SpeakerSearch`、流式识别 `StreamingSTT` 与 TTS 批处理 `TTSBatcher`；`modeldeploy::audio::tool` 提供逆文本归一化 `InverseTextNormalizer`/`ItnEngine`、特征 `Fbank`/`Spectrum`/`Waveform` 与重采样 `Resampler` 等纯音频工具。方案清单与算法说明见 [solutions.md](../solutions.md) 与 [models.md §20/§27](../models.md)；可运行示例见 `examples/demo_audio_solutions/`（`demo_diarization` / `demo_stream_stt` / `demo_tts_batch`）。

```cpp
#include "audio/solutions/speaker_search.h"
#include "audio/solutions/streaming_stt.h"
#include "audio/solutions/tts_batcher.h"
#include "audio/tools/itn.h"
#include "audio/tools/itn_engine.h"
#include "audio/tools/fbank.h"
#include "audio/tools/resampler.h"
#include "audio/tools/waveform.h"

using namespace modeldeploy;
namespace sol = modeldeploy::audio::solution;
namespace tool = modeldeploy::audio::tool;

// 1. 说话人检索：SpeakerSearch（纯内存声纹库，enroll/match top-k）
sol::SpeakerSearch ss;
ss.enroll("alice", emb);
for (auto& [label, score] : ss.match(emb, 1))   // -> vector<pair<label,score>> 降序
    std::printf("%s %.3f\n", label.c_str(), score);

// 2. TTS 批处理：TTSBatcher（set_synth 注入合成回调；可包真实 Kokoro）
modeldeploy::audio::tts::Kokoro kokoro("kokoro.onnx", "tokens.txt", {"lexicon-us-en.txt"},
                                       "voices.bin", "dict/", "", option);
sol::TTSBatcher batcher(sol::TTSBatcher::kokoro_synth(kokoro, "zf_001", 1.0f));
batcher.enqueue("锄禾日当午，汗滴禾下土。");      // 长文本自动按标点分块
auto batches = batcher.dequeue_all();             // vector<vector<float>>（每段 PCM）

// 3. 流式识别：StreamingSTT（分块 push + VAD 分段，回调交付文字）
sol::StreamingSTT stt([](const std::string& text) { std::printf("[STT] %s\n", text.c_str()); });
stt.set_transcribe(sol::StreamingSTT::sense_voice(sv));   // 可选：注入 SenseVoice 转写
stt.push(pcm_chunk, 16000); stt.run_once();               // ... 逐块喂入
stt.finish();                                             // 流结束，转写末尾语音段

// 4. 逆文本归一化：口读 -> 书面
tool::InverseTextNormalizer itn;
std::string out = itn.normalize("二零二四年三月五日");     // "2024年5月9日" 等
tool::ItnEngine eng(tool::ItnBackend::Lightweight);        // Lightweight / WeText
std::string out2 = eng.normalize("百分之五");               // "5%"
bool is_we = eng.backend() == tool::ItnBackend::WeText;

// 5. Fbank / Spectrum / Waveform / Resampler
tool::Fbank fbank(16000, 80);                              // sample_rate, num_bins
auto feats = fbank.compute(samples);                       // vector<vector<float>>（帧 x bins）
tool::Spectrum sp(1024);                                   // fft_n=1024
auto mags = sp.magnitudes(samples);                        // vector<float>
auto down = tool::Waveform::downsample(samples, 256);
auto resampled = tool::Resampler::resample(samples, 48000, 16000);  // 静态重采样
```

> C++ 音频解决方案/工具命名空间为 `modeldeploy::audio::solution`（单数）与 `modeldeploy::audio::tool`。`TTSBatcher::enqueue` 同时提供单文本与 `vector<string>` 批量重载；`StreamingSTT::set_transcribe` 可注入 `StreamingSTT::sense_voice(AsrModel&)` 转写器，缺省仅做 VAD 分段。

## 设备与设备帧

`RuntimeOption::set_device(Device::OPENCL/VULKAN)`(需显式 `use_mnn_backend()`,否则 fail-closed）:

```cpp
modeldeploy::RuntimeOption opt;
opt.use_mnn_backend();
opt.set_device(modeldeploy::Device::OPENCL, 0);   // == OK
opt.set_device(modeldeploy::Device::VULKAN, 0);   // == OK
```

设备帧 NV12：`ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, device)`(device 取 `Device::CPU/GPU/OPENCL/VULKAN/TPU`）——Python `ImageData.from_device_nv12(y, uv, w, h, dev=...)` 与 C/C#/Rust 均对齐此语义。

## 23. 工程配置

```cmake
CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(test C CXX)
set(CMAKE_CXX_STANDARD 17)
if (MSVC) add_compile_options(/utf-8) endif ()
include_directories("E:/.../install/include")
link_directories("E:/.../install/lib")
add_executable(test main.cpp)
target_link_libraries(test ModelDeploySDK)
```
