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
    obb->get_preprocessor().set_padding_value({114.f, 114.f, 114.f});
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

## 11. 更多模型（均使用同一 `RuntimeOption`）

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
| OCR | `vision::ocr::PaddleOCR` | 见 [models-OCR](../models.md#8-ocr文字识别) |
| 人脸 | `vision::face::Scrfd` / `InsightFaceAnalysis` | 见 [models-人脸](../models.md#6-人脸face) |
| 车牌 | `vision::lpr::LprPipeline` | 见 [models-车牌](../models.md#7-车牌识别license-plate) |
| ASR | `audio::asr::SenseVoice` | 见 [models-语音](../models.md#10-语音识别asr) |
| TTS（Kokoro） | `audio::tts::Kokoro` | 见 [models-TTS](../models.md#11-语音合成tts) |

> 各类模型完整 API 见 [models.md](../models.md)；后端/设备切换见 [RuntimeOption](../runtime_option.md) 与 [后端详解](../backends.md)。

### TTS 类签名

Kokoro 继承 `audio::tts::ITtsModel`，提供 `predict` / `predict_stream` / `get_sample_rate`（`predict_stream` 的 `cb` 签名 `bool(const float*, int, float progress)`，返回 `false` 中止；`chunk_frames == 0` 等价一次性合成）。

> `chunk_frames` 单位为 **UTF-8 字符数**（Kokoro，>120 字触发多块）。上层应传相对小的值以观察多次音频回调（如 120）。

- `audio::tts::Kokoro(model_onnx, tokens, lexicons, voices_bin, jieba_dir, norm_dir, opt)` —— 24kHz，`predict(text, voice, speed, &audio)`。

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

## 设备与设备帧

`RuntimeOption::set_device(Device::OPENCL/VULKAN)`(需显式 `use_mnn_backend()`,否则 fail-closed）:

```cpp
modeldeploy::RuntimeOption opt;
opt.use_mnn_backend();
opt.set_device(modeldeploy::Device::OPENCL, 0);   // == OK
opt.set_device(modeldeploy::Device::VULKAN, 0);   // == OK
```

设备帧 NV12：`ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, device)`(device 取 `Device::CPU/GPU/OPENCL/VULKAN/TPU`）——Python `ImageData.from_device_nv12(y, uv, w, h, dev=...)` 与 C/C#/Rust 均对齐此语义。

## 12. 工程配置

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
