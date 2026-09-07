# C API（`md_*` 前缀）

面向 C/C++ 嵌入式、其它语言 FFI 桥接。编译需 `BUILD_CAPI=ON`。引入头文件：

```c
#include "modeldeploy/md_capi.h"     // 常量/句柄/枚举 + 统一运行时/模型/结果接口
```

## 1. 统一分发点与生命周期

C API 是**统一分发点**设计：没有 per-model 的 create/get_preprocessor，所有模型族经一个 `md_model_create` 按 `MDModelKind` 分发，结果统一用 `md_result_*` 读取。生命周期：

```c
MDOptionHandle opt;
md_option_create(&opt);                              // 1. 建选项
// ... md_option_set_* 配置 ...
MDModelHandle m;
md_model_create(&m, MD_MODEL_DETECTION, "yolo11n.onnx", opt);  // 2. 建模型
MDImageHandle img;
md_image_from_file(&img, "test.jpg");                // 3. 读图
MDResultHandle res;
md_model_predict(m, img, &res);                      // 4. 推理
// ... md_result_* 读结果 ...
md_result_destroy(res);                              // 5. 释放
md_model_destroy(m);
md_image_destroy(img);
md_option_destroy(opt);
```

**错误处理**：所有函数返回 `MDStatus`（`MD_OK == 0` 表示成功），失败时可用 `md_get_last_error()` 取线程安全的错误信息字符串。

## 2. RuntimeOption（后端/设备/精度）

选项用 `md_option_create` 创建后，用下面的 setter 配置（均可链式校验返回值）：

```c
MDOptionHandle opt;
md_option_create(&opt);

md_option_set_backend(opt, MD_BK_ORT);       // 见下方 MD_BK_* 枚举
md_option_set_device(opt, MD_DEV_CPU, 0);    // 见下方 MD_DEV_* 枚举
md_option_set_cpu_threads(opt, 4);
md_option_set_fp16(opt, 1);                  // 0/1 表示关闭/开启
md_option_set_model_path(opt, "m.onnx", ""); // path + 可选加密密码 pwd
```

### 枚举取值

设备 `MDDevice`（`MD_DEV_*`，数值与 C#/Rust 对齐）：

| 枚举 | 值 | 说明 |
|------|----|------|
| `MD_DEV_CPU` | 0 | CPU（默认） |
| `MD_DEV_GPU` | 1 | NVIDIA GPU |
| `MD_DEV_TPU` | 2 | 算能 TPU |
| `MD_DEV_OPENCL` | 3 | OpenCL（需先 `MD_BK_MNN`，否则 fail-closed） |
| `MD_DEV_VULKAN` | 4 | Vulkan（需先 `MD_BK_MNN`，否则 fail-closed） |

后端 `MDBackend`（`MD_BK_*`）：

| 枚举 | 值 | 说明 |
|------|----|------|
| `MD_BK_ORT` | 0 | OnnxRuntime（`.onnx`，最通用） |
| `MD_BK_MNN` | 1 | MNN（`.mnn`） |
| `MD_BK_TRT` | 2 | TensorRT（`.engine`） |
| `MD_BK_SOPHGO` | 3 | Sophgo（`.bmodel`） |
| `MD_BK_NCNN` | 4 | ncnn（`.param`/`.bin`） |

### 骨架示例

生命周期总览（create→configure→predict→read→destroy）见上文 §1；完整检测示例见下文 §3 目标检测（结果项 `MDDetectionItem` 为扁平结构，直接访问 `items[i].x/y/w/h/score/label_id`）。

## 3. 目标检测（`MD_MODEL_DETECTION`）

C API 无 per-model 类，检测模型由 `md_model_create(kind=MD_MODEL_DETECTION, ...)` 创建。检测结果经 `md_result_detection` 以 `MDDetectionItem{ x, y, w, h, score, label_id }` 数组返回（内存归结果句柄所有，无需逐项释放）。

```c
#include <stdio.h>
#include "modeldeploy/md_capi.h"

int main(void) {
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle m = NULL;
    if (md_model_create(&m, MD_MODEL_DETECTION, "yolo11n.onnx", opt) != MD_OK) {
        fprintf(stderr, "create failed: %s\n", md_get_last_error());
        return 1;
    }

    /* 预处理/后处理参数：输入尺寸 + 阈值。
     * 参数名以运行时内省为准（md_model_param_names 返回 "conf_threshold|nms_threshold"） */
    md_model_set_input_size(m, 640, 640);
    const char* param_names = NULL;
    md_model_param_names(MD_MODEL_DETECTION, &param_names);  /* 自省本 kind 支持的参数名 */
    md_model_set_param_d(m, "conf_threshold", 0.25);
    md_model_set_param_d(m, "nms_threshold", 0.45);

    MDImageHandle img = NULL;
    md_image_from_file(&img, "test.jpg");

    MDResultHandle res = NULL;
    if (md_model_predict(m, img, &res) != MD_OK) {
        fprintf(stderr, "predict failed: %s\n", md_get_last_error());
        return 1;
    }

    /* 结果遍历 */
    const MDDetectionItem* items = NULL;
    size_t n = 0;
    md_result_detection(res, &items, &n);
    for (size_t i = 0; i < n; i++) {
        printf("[%zu] label=%d score=%.3f box=(%.0f, %.0f, %.0f, %.0f)\n",
               i, items[i].label_id, items[i].score,
               items[i].x, items[i].y, items[i].w, items[i].h);
    }
    /* 批量推理：结果按图存储，用 md_result_detection_batch 逐图读取 */
    MDImageHandle img_b = NULL;
    md_image_from_file(&img_b, "bus.jpg");
    MDImageHandle imgs[2] = {img, img_b};
    MDResultHandle bres = NULL;
    if (md_model_predict_batch(m, imgs, 2, &bres) == MD_OK) {
        for (size_t g = 0; g < 2; ++g) {
            const MDDetectionItem* bitems = NULL;
            size_t bn = 0;
            if (md_result_detection_batch(bres, g, &bitems, &bn) == MD_OK) {
                printf("image %zu: %zu objects\n", g, bn);
            }
        }
        md_result_destroy(bres);
    }
    md_image_destroy(img_b);

    /* 多线程：md_model_clone 深拷贝独立句柄（每线程持有一个，互不干扰） */
    MDModelHandle m2 = NULL;
    md_model_clone(m, &m2);

    /* 可视化：md_draw_result 就地绘制到画布（threshold/label_map/字体/alpha 可控，传 NULL 用默认值）；
     * 需在结果句柄存活期间调用 */
    MDLabelItem label_map[] = {{0, "person"}, {1, "bicycle"}, {2, "car"}};
    MDDrawOptions dopt = {0};
    dopt.threshold = 0.25;
    dopt.label_map = label_map;
    dopt.label_map_size = 3;
    dopt.font_size = 14;
    dopt.alpha = 0.3;
    MDImageHandle canvas = NULL;
    md_image_clone(img, &canvas);
    md_draw_result(canvas, res, &dopt);
    md_image_save(canvas, "det_vis.jpg");
    md_image_destroy(canvas);

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(m2);
    md_model_destroy(m);
    md_option_destroy(opt);
    return 0;
}
```

## 4. 实例分割（`MD_MODEL_INSTANCE_SEG`）

实例分割模型经 `md_model_create(kind=MD_MODEL_INSTANCE_SEG, ...)` 创建。实例框经 `md_result_instance_seg` 以 `MDIsegItem{ x, y, w, h, score, label_id }` 数组返回；掩码单独经 `md_result_mask(res, i, ...)` 按实例序号读取（uint8 0/1，行主序 H*W，内存归结果句柄所有）。

```c
#include <stdio.h>
#include "modeldeploy/md_capi.h"

int main(void) {
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);

    MDModelHandle m = NULL;
    if (md_model_create(&m, MD_MODEL_INSTANCE_SEG, "yolo11n-seg.onnx", opt) != MD_OK) {
        fprintf(stderr, "create failed: %s\n", md_get_last_error());
        return 1;
    }
    md_model_set_input_size(m, 640, 640);
    /* 参数自省：INSTANCE_SEG 返回 "conf_threshold|nms_threshold|mask_threshold"（类型均 'D'） */
    md_model_set_param_d(m, "conf_threshold", 0.25);
    md_model_set_param_d(m, "nms_threshold", 0.45);
    md_model_set_param_d(m, "mask_threshold", 0.5);   /* 掩码二值化阈值（默认 0.5） */

    MDImageHandle img = NULL;
    md_image_from_file(&img, "test.jpg");
    MDResultHandle res = NULL;
    md_model_predict(m, img, &res);

    const MDIsegItem* items = NULL;
    size_t n = 0;
    md_result_instance_seg(res, &items, &n);
    for (size_t i = 0; i < n; i++) {
        const unsigned char* mask = NULL;
        size_t mh = 0, mw = 0;
        md_result_mask(res, i, &mask, &mh, &mw);   /* mask[y * mw + x]，uint8 0/1 */
        printf("[%zu] label=%d score=%.3f box=(%.0f, %.0f, %.0f, %.0f) mask=%zux%zu\n",
               i, items[i].label_id, items[i].score,
               items[i].x, items[i].y, items[i].w, items[i].h, mh, mw);
    }

    /* 批量推理：md_result_instance_seg_batch(bres, g, &items, &n) 取第 g 图项数组，
     * 掩码用 md_result_mask_batch(bres, g, j, &mask, &mh, &mw) 按 (图,项) 读 */

    /* 可视化：md_draw_result 支持 MD_RES_INSTANCE_SEG（底层 vis_iseg），用法同 §3 */

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(m);
    md_option_destroy(opt);
    return 0;
}
```

## 5. FastSAM（`MD_MODEL_FASTSAM`）

FastSAM 结果与实例分割完全同构（`md_result_instance_seg` + `md_result_mask`）。`md_fastsam_predict_with_prompts` 在全量结果上按提示过滤实例，**不重跑网络**；空提示（`nb==0 && np==0`）等价全图 `md_model_predict`。

```c
MDModelHandle m = NULL;
md_model_create(&m, MD_MODEL_FASTSAM, "fastsam-s.onnx", opt);
md_model_set_input_size(m, 1024, 1024);   /* 默认 640x640；官方 FastSAM-s 常配 1024x1024 */
/* 参数自省：FASTSAM 返回 "conf_threshold|nms_threshold|mask_threshold"；
 * 默认 conf 0.30 / nms 0.5 / mask 0.5 */
md_model_set_param_d(m, "conf_threshold", 0.30);
md_model_set_param_d(m, "nms_threshold", 0.40);

MDImageHandle img = NULL;
md_image_from_file(&img, "test.jpg");

/* 全图（Everything）分割：普通 predict，读取同实例分割 */
MDResultHandle res = NULL;
md_model_predict(m, img, &res);
md_result_destroy(res);

/* 提示过滤：
 * - bboxes: float[nb*4] = (x,y,w,h) 原图像素，每个框取 IoU 最大实例
 * - points: float[np*2]；labels: int[np]（1=前景保留, 0=背景剔除；NULL = 全前景）
 */
float bboxes[4] = {100.f, 80.f, 220.f, 180.f};
float points[2] = {150.f, 130.f};
int labels[1] = {1};
MDResultHandle pres = NULL;
md_fastsam_predict_with_prompts(m, img, bboxes, 1, points, labels, 1, &pres);
/* 读取同 INSTANCE_SEG：md_result_instance_seg(pres, &items, &n) + md_result_mask(pres, i, ...) */
md_result_destroy(pres);
```

## 6. 语义分割（`MD_MODEL_SEM_SEG`）

语义分割模型（`yolo26n-sem` 等，cityscapes 19 类）。结果经 `md_result_sem_seg` 返回每像素类别索引（uint8，`[0, num_classes)`，行主序 H*W）；该 kind 无 `md_model_set_param_*` 参数（后处理 argmax + 去除 letterbox 边，返回空参数表）。

```c
MDModelHandle m = NULL;
md_model_create(&m, MD_MODEL_SEM_SEG, "yolo26n-sem.onnx", opt);
md_model_set_input_size(m, 640, 640);

MDImageHandle img = NULL;
md_image_from_file(&img, "test_sem_540.jpg");
MDResultHandle res = NULL;
md_model_predict(m, img, &res);

const unsigned char* labels = NULL;
size_t h = 0, w = 0;
int num_classes = 0;
md_result_sem_seg(res, &labels, &h, &w, &num_classes);   /* labels[y * w + x] */
printf("sem %zux%zu classes=%d\n", h, w, num_classes);

/* 批量推理：md_result_sem_seg_batch(bres, g, &labels, &h, &w, &num_classes) 按图读 */

/* 可视化：md_draw_result 支持 MD_RES_SEM_SEG（底层 vis_sem，label_map 经 MDDrawOptions） */
md_result_destroy(res);
```

## 7. 深度估计（`MD_MODEL_DEPTH`）

深度估计模型（`yolo26n-depth` 等）。结果经 `md_result_depth` 返回每像素深度（float，单位**米**，log 输出已 `exp` 还原，行主序 H*W）；该 kind 无 `md_model_set_param_*` 参数（返回空参数表）。

```c
MDModelHandle m = NULL;
md_model_create(&m, MD_MODEL_DEPTH, "yolo26n-depth.onnx", opt);
md_model_set_input_size(m, 640, 640);

MDImageHandle img = NULL;
md_image_from_file(&img, "test_depth_540.jpg");
MDResultHandle res = NULL;
md_model_predict(m, img, &res);

const float* depth = NULL;
size_t h = 0, w = 0;
md_result_depth(res, &depth, &h, &w);   /* depth[y * w + x]，单位米 */
printf("depth %zux%zu, center=%.2f m\n", h, w, depth[(h / 2) * w + (w / 2)]);

/* 批量推理：md_result_depth_batch(bres, g, &depth, &h, &w) 按图读 */

/* 可视化：md_draw_result 支持 MD_RES_DEPTH（底层 vis_depth JET 伪彩） */
md_result_destroy(res);
```

## 8. 姿态与关键点族（`MD_MODEL_POSE` / `MD_MODEL_HAND` / `MD_MODEL_VEHICLE_KEYPOINT` / `MD_MODEL_FACE_LANDMARK`）

四个 kind 均经 `md_model_create` 创建，结果统一为 `MD_RES_POSE` 种类：`md_result_pose` 返回 `MDPoseItem{ x, y, w, h, score }` 数组，关键点经 `md_result_keypoints(res, i, ...)` 按实例序号读取 `MDPoint3{ x, y, z }` 数组（`z` 为关键点置信度）。参数自省：POSE/HAND/VEHICLE_KEYPOINT 返回 `"conf_threshold|nms_threshold|keypoints_num"`（类型 'D'/'D'/'I'，默认 conf 0.30 / nms 0.5 / 17 点）；FACE_LANDMARK 无参数（固定 106 点，InsightFace 2d106，输入须为人脸裁剪图）。

```c
#include <stdio.h>
#include "modeldeploy/md_capi.h"

int main(void) {
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);

    MDModelHandle m = NULL;
    if (md_model_create(&m, MD_MODEL_POSE, "yolo11n-pose.onnx", opt) != MD_OK) {
        fprintf(stderr, "create failed: %s\n", md_get_last_error());
        return 1;
    }
    md_model_set_input_size(m, 640, 640);
    /* 参数自省：POSE/HAND/VEHICLE_KEYPOINT 返回 "conf_threshold|nms_threshold|keypoints_num"
     * （keypoints_num 类型 'I'，须用 md_model_set_param_i）；默认 conf 0.30 / nms 0.5 / 17 点 */
    md_model_set_param_d(m, "conf_threshold", 0.30);
    md_model_set_param_d(m, "nms_threshold", 0.45);
    md_model_set_param_i(m, "keypoints_num", 17);

    MDImageHandle img = NULL;
    md_image_from_file(&img, "test.jpg");
    MDResultHandle res = NULL;
    if (md_model_predict(m, img, &res) != MD_OK) {
        fprintf(stderr, "predict failed: %s\n", md_get_last_error());
        return 1;
    }

    /* 结果遍历：bbox+score 在 items，关键点按实例序号读取 */
    const MDPoseItem* items = NULL;
    size_t n = 0;
    md_result_pose(res, &items, &n);
    for (size_t i = 0; i < n; i++) {
        const MDPoint3* kps = NULL;
        size_t kn = 0;
        md_result_keypoints(res, i, &kps, &kn);   /* kps[j].x/y/z，z 为关键点置信度 */
        printf("[%zu] score=%.3f box=(%.0f, %.0f, %.0f, %.0f) kps=%zu first=(%.1f, %.1f, %.2f)\n",
               i, items[i].score, items[i].x, items[i].y, items[i].w, items[i].h, kn,
               kn > 0 ? kps[0].x : 0.f, kn > 0 ? kps[0].y : 0.f, kn > 0 ? kps[0].z : 0.f);
    }
    /* 批量推理：md_result_pose_batch(bres, g, &items, &n) 取第 g 图项数组，
     * 关键点用 md_result_keypoints_batch(bres, g, j, &kps, &kn) 按 (图, 项) 读 */

    /* 可视化：md_draw_result 支持 MD_RES_POSE（底层 vis_pose 骨架连线），用法同 §3 */

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(m);
    md_option_destroy(opt);
    return 0;
}
```

手部 / 车辆 / 面部 Landmark 只是换 kind 创建，参数与结果读取完全同构：

```c
MDModelHandle hand = NULL, vk = NULL, fl = NULL;
md_model_create(&hand, MD_MODEL_HAND, "hand.onnx", opt);
md_model_set_param_i(hand, "keypoints_num", 21);   /* 手部 21 点（构造默认 21，可省略） */

md_model_create(&vk, MD_MODEL_VEHICLE_KEYPOINT, "vehicle.onnx", opt);
md_model_set_param_i(vk, "keypoints_num", 4);      /* 车辆 4 车轮点（构造默认 4，不同车型可覆盖） */

/* 面部 Landmark（InsightFace 2d106）：无参数（md_model_param_names 返回空表）、固定 106 点（z=0）、
 * 输入须为人脸裁剪图；多线程用 md_model_clone */
md_model_create(&fl, MD_MODEL_FACE_LANDMARK, "face_landmark.onnx", opt);

/* 三者结果读取同上：md_result_pose + md_result_keypoints（或 _batch 变体），
 * md_draw_result 亦按 MD_RES_POSE 绘制 */
```

## 9. OBB（旋转框检测）（`MD_MODEL_OBB`）

OBB 模型经 `md_model_create(kind=MD_MODEL_OBB, ...)` 创建。结果经 `md_result_obb` 以 `MDObbItem{ cx, cy, w, h, angle, score, label_id }` 数组返回（`cx/cy` 为旋转框中心、`angle` 为弧度角，坐标均为原图像素）。

```c
#include <stdio.h>
#include "modeldeploy/md_capi.h"

int main(void) {
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);

    MDModelHandle m = NULL;
    if (md_model_create(&m, MD_MODEL_OBB, "yolo11n-obb.onnx", opt) != MD_OK) {
        fprintf(stderr, "create failed: %s\n", md_get_last_error());
        return 1;
    }
    md_model_set_input_size(m, 1024, 1024);   /* letterbox 输入尺寸（默认 1024x1024） */
    /* 参数自省：OBB 返回 "conf_threshold|nms_threshold"（类型均 'D'） */
    md_model_set_param_d(m, "conf_threshold", 0.25);
    md_model_set_param_d(m, "nms_threshold", 0.45);

    MDImageHandle img = NULL;
    md_image_from_file(&img, "test.jpg");
    MDResultHandle res = NULL;
    md_model_predict(m, img, &res);

    const MDObbItem* items = NULL;
    size_t n = 0;
    md_result_obb(res, &items, &n);
    for (size_t i = 0; i < n; i++) {
        printf("[%zu] label=%d score=%.3f obb=(xc=%.1f, yc=%.1f, w=%.1f, h=%.1f, angle=%.3f)\n",
               i, items[i].label_id, items[i].score,
               items[i].cx, items[i].cy, items[i].w, items[i].h, items[i].angle);
    }

    /* 批量推理：md_result_obb_batch(bres, g, &items, &n) 取第 g 图项数组 */

    /* 可视化：md_draw_result 支持 MD_RES_OBB（底层 vis_obb），用法同 §3 */

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(m);
    md_option_destroy(opt);
    return 0;
}
```

## 10. 图像分类（`MD_MODEL_CLASSIFICATION`）

分类模型经 `md_model_create(kind=MD_MODEL_CLASSIFICATION, ...)` 创建。结果经 `md_result_classification` 以 `MDClassifyItem{ label_id, score }` 数组返回（Top-K 逐项，`label_id`/`score` 逐位配对）。

```c
#include <stdio.h>
#include "modeldeploy/md_capi.h"

int main(void) {
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);

    MDModelHandle m = NULL;
    if (md_model_create(&m, MD_MODEL_CLASSIFICATION, "yolo11n-cls.onnx", opt) != MD_OK) {
        fprintf(stderr, "create failed: %s\n", md_get_last_error());
        return 1;
    }
    md_model_set_input_size(m, 224, 224);     /* 输入尺寸（默认 224x224） */
    /* 参数自省：CLASSIFICATION 返回 "top_k|multi_label"（类型 'I' / 'B'） */
    md_model_set_param_i(m, "top_k", 5);      /* Top-K 输出个数（默认 1） */
    md_model_set_param_b(m, "multi_label", 0);/* 多标签模式（默认 0） */

    MDImageHandle img = NULL;
    md_image_from_file(&img, "test.jpg");
    MDResultHandle res = NULL;
    md_model_predict(m, img, &res);

    const MDClassifyItem* items = NULL;
    size_t n = 0;
    md_result_classification(res, &items, &n);
    for (size_t i = 0; i < n; i++) {
        printf("[%zu] label=%d score=%.3f\n", i, items[i].label_id, items[i].score);
    }

    /* 批量推理：md_result_classification_batch(bres, g, &items, &n) 取第 g 图项数组 */

    /* 可视化：md_draw_result 支持 MD_RES_CLASSIFICATION（底层 vis_cls），用法同 §3 */

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(m);
    md_option_destroy(opt);
    return 0;
}
```

## 11. OCR（`MD_MODEL_OCR` / `MD_MODEL_OCR_DET` / `MD_MODEL_OCR_REC` / `MD_MODEL_OCR_CLS`）

主流水线 `MD_MODEL_OCR` 经 `md_model_create` 创建，`model_path` 用 `|` 串联 **det/cls/rec/dict 四段**：`"det.onnx|cls.onnx|rec.onnx|dict.txt"`。结果 kind 为 `MD_RES_OCR`：每行文本 = `quad`（4 点共 8 个 int，原图像素）+ `text` + `score`（`md_result_ocr`），方向分类 label/score 按行用 `md_result_ocr_cls` 配对。**注意**：单图 OCR 结果的 `md_result_count` 恒为 1（单值包装），行数以 `md_result_ocr` 越界（返回非 `MD_OK`）为准。

子模型 kind（可独立部署）：`MD_MODEL_OCR_DET`（单段 det.onnx）、`MD_MODEL_OCR_REC`（`"rec.onnx|dict.txt"` 两段）、`MD_MODEL_OCR_CLS`（单段 cls.onnx）。

```c
#include <stdio.h>
#include "modeldeploy/md_capi.h"

int main(void) {
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);

    /* 主流水线：det|cls|rec|dict 四段路径（'|' 分隔） */
    MDModelHandle m = NULL;
    if (md_model_create(&m, MD_MODEL_OCR,
                        "det.onnx|cls.onnx|rec.onnx|dict.txt", opt) != MD_OK) {
        fprintf(stderr, "create failed: %s\n", md_get_last_error());
        return 1;
    }
    /* 参数自省：OCR 返回 "det_db_thresh|det_db_box_thresh|det_db_unclip_ratio|
     * det_db_score_mode|use_dilation|cls_thresh|max_side_len"（类型 D/D/D/S/B/D/I） */
    md_model_set_param_d(m, "det_db_thresh", 0.3);        /* DB 二值化阈值（默认 0.3） */
    md_model_set_param_d(m, "det_db_box_thresh", 0.6);    /* 框置信度阈值（默认 0.6） */
    md_model_set_param_d(m, "det_db_unclip_ratio", 1.5);  /* 扩框比例（默认 1.5） */
    md_model_set_param_s(m, "det_db_score_mode", "slow"); /* 框得分模式（默认 "slow"） */
    md_model_set_param_b(m, "use_dilation", 0);           /* 是否膨胀（默认 0） */
    md_model_set_param_d(m, "cls_thresh", 0.9);           /* 方向分类阈值（默认 0.9） */
    md_model_set_param_i(m, "max_side_len", 960);         /* 检测最长边（默认 960） */
    md_model_set_cls_batch_size(m, 6);                    /* 方向分类子模型 batch（默认 6） */
    md_model_set_rec_batch_size(m, 8);                    /* 识别子模型 batch（默认 8） */
    md_model_set_rec_image_shape(m, 3, 48, 320);          /* 识别输入形状（默认 3x48x320） */

    MDImageHandle img = NULL;
    md_image_from_file(&img, "test.jpg");
    MDResultHandle res = NULL;
    md_model_predict(m, img, &res);

    /* 逐行读直到 md_result_ocr 返回非 MD_OK（行数以越界为准，见上） */
    for (size_t i = 0; ; i++) {
        const int* quad = NULL; const char* text = NULL; float score = 0.f;
        int cls_label = 0; float cls_score = 0.f;
        if (md_result_ocr(res, i, &quad, &text, &score) != MD_OK) break;
        md_result_ocr_cls(res, i, &cls_label, &cls_score);
        printf("[%zu] '%s' score=%.3f cls=%d box=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
               i, text ? text : "", score, cls_label,
               quad[0], quad[1], quad[2], quad[3],
               quad[4], quad[5], quad[6], quad[7]);
    }

    /* 批量推理：md_result_ocr_batch_count(bres, &nimgs) 得图数，
     * md_result_ocr_batch(bres, g, j, &quad, &text, &score) +
     * md_result_ocr_cls_batch(bres, g, j, &cls_label, &cls_score) 按 (图, 行) 读 */

    /* 可视化：md_draw_result 支持 MD_RES_OCR（底层 vis_ocr），用法同 §3 */

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(m);
    md_option_destroy(opt);
    return 0;
}
```

## 12. OCR 进阶（版面 / 表格 / 公式 / 文档转 Markdown）

C API 仅绑定其中**公式识别**：经 `MD_MODEL_FORMULA_RECOGNIZER` 创建、`md_model_predict` 推理，结果用 `md_result_formula` 读出 LaTeX 字符串。
**版面 `StructureV2Layout` / 表格 `StructureV2Table` / `PPStructureV2Table` / 文档转 Markdown `DocToMarkdown` 本语言未绑定**（无对应 `MDModelKind` 与 `md_result_*` 读取接口），如需请用 C++ / Python 绑定。

`MD_MODEL_FORMULA_RECOGNIZER` 的 `model_path` 用 `|` 串联 **model[|dict]** 两段：`"formula.onnx|dict.txt"`（dict 可省略为 `"formula.onnx"`）。单图结果为 `MD_RES_FORMULA`（单值包装），`md_result_count` 恒为 1，用 `md_result_formula(h, 0, &latex)` 读取。

```c
#include <stdio.h>
#include "modeldeploy/md_capi.h"

int main(void) {
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);

    /* 公式识别：model|dict 两段路径（'|' 分隔，dict 可省） */
    MDModelHandle m = NULL;
    if (md_model_create(&m, MD_MODEL_FORMULA_RECOGNIZER,
                        "formula.onnx|dict.txt", opt) != MD_OK) {
        fprintf(stderr, "create failed: %s\n", md_get_last_error());
        return 1;
    }

    MDImageHandle img = NULL;
    md_image_from_file(&img, "equation.jpg");
    MDResultHandle res = NULL;
    if (md_model_predict(m, img, &res) != MD_OK) return 1;

    const char* latex = NULL;
    if (md_result_formula(res, 0, &latex) != MD_OK) return 1;
    printf("LaTeX: %s\n", latex ? latex : "");

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(m);
    md_option_destroy(opt);
    return 0;
}
```

## 13. 人脸（`MD_MODEL_FACE_DET` / `FACE_REC` / `FACE_AGE` / `FACE_GENDER` / `FACE_AS_PIPELINE` / `FACE_REC_PIPELINE`）

人脸六个 kind 均经 `md_model_create` 创建，结果按模型族不同读取：

| kind | 底层模型 | 结果 kind | 读取接口 |
|------|---------|-----------|----------|
| `MD_MODEL_FACE_DET` | `Scrfd`（框+5 关键点） | `MD_RES_FACE` | `md_result_face` + `md_result_face_kps` |
| `MD_MODEL_FACE_REC` | `SeetaFaceID`（512-d 特征） | `MD_RES_FACE_REC` | `md_result_face_embedding` |
| `MD_MODEL_FACE_AGE` | `SeetaFaceAge` | `MD_RES_AGE` | `md_result_age` |
| `MD_MODEL_FACE_GENDER` | `SeetaFaceGender` | `MD_RES_GENDER` | `md_result_gender` |
| `MD_MODEL_FACE_AS_PIPELINE` | `SeetaFaceAsPipeline`（det\|first\|second） | `MD_RES_ANTISPOOF` | `md_result_spoof` |
| `MD_MODEL_FACE_REC_PIPELINE` | `FaceRecognizerPipeline`（det\|rec） | `MD_RES_FACE_REC` | `md_result_face_embedding` |

参数自省：`FACE_DET` / `FACE_REC_PIPELINE` 返回 `"conf_threshold|nms_threshold|landmarks_per_face"`（类型 `D/D/I`）；`FACE_REC`/`FACE_AGE`/`FACE_GENDER`/`FACE_AS_PIPELINE` 无参数表（年龄/性别后处理 argmax、防伪流水线用默认 fuse 0.8 / clarity 0.3）。多子模型路径用 `|` 串联：`FACE_AS_PIPELINE` 为 `"det.onnx|first.onnx|second.onnx"`、`FACE_REC_PIPELINE` 为 `"det.onnx|rec.onnx"`。

```c
#include <stdio.h>
#include "modeldeploy/md_capi.h"

int main(void) {
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);

    MDImageHandle img = NULL;
    md_image_from_file(&img, "test.jpg");

    /* 1. 人脸检测：FACE_DET。MDFaceItem{x, y, w, h, score}，关键点按实例读 MDPoint */
    MDModelHandle det = NULL;
    md_model_create(&det, MD_MODEL_FACE_DET, "scrfd.onnx", opt);
    /* 参数自省：FACE_DET 返回 "conf_threshold|nms_threshold|landmarks_per_face"（D/D/I） */
    md_model_set_input_size(det, 640, 640);
    md_model_set_param_d(det, "conf_threshold", 0.30);
    md_model_set_param_d(det, "nms_threshold", 0.45);
    md_model_set_param_i(det, "landmarks_per_face", 5);
    MDResultHandle res = NULL;
    md_model_predict(det, img, &res);
    const MDFaceItem* items = NULL;
    size_t n = 0;
    md_result_face(res, &items, &n);
    for (size_t i = 0; i < n; i++) {
        const MDPoint* kps = NULL;
        size_t kn = 0;
        md_result_face_kps(res, i, &kps, &kn);      /* 5 关键点 (x, y) */
        printf("[%zu] score=%.3f box=(%.0f, %.0f, %.0f, %.0f) kps=%zu\n",
               i, items[i].score, items[i].x, items[i].y, items[i].w, items[i].h, kn);
    }
    md_result_destroy(res);
    md_model_destroy(det);

    /* 2. 人脸识别：FACE_REC（单面输出），embedding 归结果句柄所有 */
    MDModelHandle rec = NULL;
    md_model_create(&rec, MD_MODEL_FACE_REC, "rec.onnx", opt);
    md_model_predict(rec, img, &res);
    const float* emb = NULL;
    size_t emb_n = 0;
    md_result_face_embedding(res, 0, &emb, &emb_n);
    printf("embedding dim=%zu first=%.4f\n", emb_n, emb_n ? emb[0] : 0.f);
    md_result_destroy(res);
    md_model_destroy(rec);

    /* 3. 年龄 / 性别：FACE_AGE / FACE_GENDER（单值），gender 0=女 1=男 */
    MDModelHandle age = NULL, gender = NULL;
    md_model_create(&age, MD_MODEL_FACE_AGE, "age.onnx", opt);
    md_model_create(&gender, MD_MODEL_FACE_GENDER, "gender.onnx", opt);
    int a = 0, g = 0;
    md_model_predict(age, img, &res);    md_result_age(res, &a);    md_result_destroy(res);
    md_model_predict(gender, img, &res); md_result_gender(res, &g); md_result_destroy(res);
    printf("age=%d gender=%d\n", a, g);
    md_model_destroy(age); md_model_destroy(gender);

    /* 4. 防伪流水线：FACE_AS_PIPELINE，路径 '|' 三段；结果经 md_result_spoof */
    MDModelHandle asp = NULL;
    md_model_create(&asp, MD_MODEL_FACE_AS_PIPELINE,
                    "det.onnx|first.onnx|second.onnx", opt);
    md_model_predict(asp, img, &res);
    for (size_t i = 0; ; i++) {
        int label = 0;
        if (md_result_spoof(res, i, &label) != MD_OK) break;   /* 0=REAL 1=FUZZY 2=SPOOF */
        printf("face %zu: %s\n", i, label == 0 ? "REAL" : label == 1 ? "FUZZY" : "SPOOF");
    }
    md_result_destroy(res);
    md_model_destroy(asp);

    /* 5. 识别流水线：FACE_REC_PIPELINE，'|' 两段；embedding 按实例读 */
    MDModelHandle rp = NULL;
    md_model_create(&rp, MD_MODEL_FACE_REC_PIPELINE, "det.onnx|rec.onnx", opt);
    md_model_set_param_d(rp, "conf_threshold", 0.30);
    md_model_predict(rp, img, &res);
    for (size_t i = 0; ; i++) {
        const float* e = NULL; size_t en = 0;
        if (md_result_face_embedding(res, i, &e, &en) != MD_OK) break;
        printf("face %zu: dim=%zu\n", i, en);
    }
    md_result_destroy(res);
    md_model_destroy(rp);

    md_image_destroy(img);
    md_option_destroy(opt);
    return 0;
}
```

## 14. InsightFace 全流程（`MD_MODEL_INSIGHTFACE` / `MD_MODEL_INSIGHTFACE_DET`）

InsightFace Buffalo 全家桶（det + 2d106 + 3d68 + recognition）组合成一次 `md_model_predict`，输出检测框/5 关键点/姿态/特征/性别年龄。子模型分别创建。

```c
#include <stdio.h>
#include "modeldeploy/md_capi.h"

int main(void) {
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);

    MDImageHandle img = NULL;
    md_image_from_file(&img, "test.jpg");

    /* 1. 全流程分析：MD_MODEL_INSIGHTFACE。model_path 用 '|' 串联最多 5 段子模型路径
     *    det_10g.onnx|w600k_r50.onnx|2d106det.onnx|1k3d68.onnx|genderage.onnx
     *    （至少 4 段，第 5 段 genderage 可省略）。参数自省为 "det_thresh"（类型 'D'） */
    MDModelHandle m = NULL;
    if (md_model_create(&m, MD_MODEL_INSIGHTFACE,
            "det_10g.onnx|w600k_r50.onnx|2d106det.onnx|1k3d68.onnx|genderage.onnx",
            opt) != MD_OK) {
        fprintf(stderr, "create failed: %s\n", md_get_last_error());
        return 1;
    }
    md_model_set_param_d(m, "det_thresh", 0.5);    /* 检测阈值（默认 0.5） */

    MDResultHandle res = NULL;
    md_model_predict(m, img, &res);

    const MDInsightFaceItem* items = NULL;   /* {x, y, w, h, score, gender, age} */
    size_t n = 0;
    md_result_insightface(res, &items, &n);
    for (size_t i = 0; i < n; i++) {
        const MDPoint* kps = NULL; size_t kn = 0;
        md_result_insightface_kps(res, i, &kps, &kn);        /* 5 关键点 (x,y) */
        const float* emb = NULL; size_t en = 0;
        md_result_insightface_embedding(res, i, &emb, &en);  /* 512 维特征 */
        const float* pose = NULL; size_t pn = 0;
        md_result_insightface_pose(res, i, &pose, &pn);      /* 3 姿态角 [pitch,yaw,roll] */
        printf("[%zu] score=%.3f box=(%.0f,%.0f,%.0f,%.0f) kps=%zu emb=%zu gender=%d age=%d\n",
               i, items[i].score, items[i].x, items[i].y, items[i].w, items[i].h,
               kn, en, items[i].gender, items[i].age);
    }
    /* 批量：md_model_predict_batch + md_result_insightface_batch/_kps_batch/_embedding_batch/_pose_batch */
    /* 可视化：md_draw_result 支持 MD_RES_INSIGHTFACE（底层 vis_keypoints，alpha 经 MDDrawOptions，如 0.3） */

    /* 注意：C API 仅暴露 bbox/score/kps/embedding/pose/gender/age，不含 106/68 关键点（C++/Python 才有） */

    md_result_destroy(res);
    md_model_destroy(m);

    /* 2. 子模型：MD_MODEL_INSIGHTFACE_DET（仅检测，det_10g.onnx）。
     *    结果 kind 为 MD_RES_FACE，读取同人脸检测：md_result_face + md_result_face_kps */
    MDModelHandle det = NULL;
    md_model_create(&det, MD_MODEL_INSIGHTFACE_DET, "det_10g.onnx", opt);
    md_model_set_input_size(det, 640, 640);
    MDResultHandle det_res = NULL;
    md_model_predict(det, img, &det_res);
    const MDFaceItem* ditem = NULL; size_t dn = 0;
    md_result_face(det_res, &ditem, &dn);
    for (size_t i = 0; i < dn; i++) {
        const MDPoint* dk = NULL; size_t dkn = 0;
        md_result_face_kps(det_res, i, &dk, &dkn);   /* 5 关键点 (x,y) */
        printf("det[%zu] score=%.3f box=(%.0f,%.0f,%.0f,%.0f) kps=%zu\n",
               i, ditem[i].score, ditem[i].x, ditem[i].y, ditem[i].w, ditem[i].h, dkn);
    }
    md_result_destroy(det_res);
    md_model_destroy(det);

    md_image_destroy(img);
    md_option_destroy(opt);
    return 0;
}
```

## 15. 车牌 LPR（`MD_MODEL_LPR_PIPELINE` / `LPR_DET` / `LPR_REC`）

车牌三个 kind 均经 `md_model_create` 创建，结果统一为 `MD_RES_LPR`：`md_result_lpr` 返回 `MDLprItem{ x, y, w, h, score }` 数组，车牌字符串/颜色经 `md_result_plate` 按实例序号读，4 角点经 `md_result_lpr_keypoints` 读（`MDPoint{x,y}` 数组）。主流水线 `MD_MODEL_LPR_PIPELINE` 的 `model_path` 用 `|` 串联 **det|rec 两段**：`"det.onnx|rec.onnx"`。子模型：`MD_MODEL_LPR_DET`（单段 det.onnx，参数 `conf_threshold|nms_threshold|landmarks_per_card`，类型均 `'D'`）、`MD_MODEL_LPR_REC`（单段 rec.onnx，无参数表）。

```c
#include <stdio.h>
#include "modeldeploy/md_capi.h"

int main(void) {
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);

    /* 1. 主流水线：LPR pipeline，model_path 用 '|' 串联 det|rec 两段 */
    MDModelHandle m = NULL;
    if (md_model_create(&m, MD_MODEL_LPR_PIPELINE, "det.onnx|rec.onnx", opt) != MD_OK) {
        fprintf(stderr, "create failed: %s\n", md_get_last_error());
        return 1;
    }

    MDImageHandle img = NULL;
    md_image_from_file(&img, "test.jpg");
    MDResultHandle res = NULL;
    md_model_predict(m, img, &res);

    /* 结果遍历：MDLprItem{x,y,w,h,score} 数组 + 车牌字符串/颜色 + 4 角点 */
    const MDLprItem* items = NULL;
    size_t n = 0;
    md_result_lpr(res, &items, &n);
    for (size_t i = 0; i < n; i++) {
        const char* plate = NULL; const char* color = NULL;
        md_result_plate(res, i, &plate, &color);
        const MDPoint* kps = NULL; size_t kn = 0;
        md_result_lpr_keypoints(res, i, &kps, &kn);   /* 车牌 4 角点 (x, y) */
        printf("[%zu] '%s' color=%s score=%.3f box=(%.0f, %.0f, %.0f, %.0f) kps=%zu\n",
               i, plate ? plate : "", color ? color : "", items[i].score,
               items[i].x, items[i].y, items[i].w, items[i].h, kn);
        for (size_t j = 0; j < kn; j++)
            printf("  kp=(%.1f, %.1f)\n", kps[j].x, kps[j].y);
    }
    /* 批量推理：md_result_lpr_batch(bres, g, &items, &n) 取第 g 图项数组，车牌/角点用
     * md_result_plate_batch / md_result_lpr_keypoints_batch(bres, g, j, ...) 按 (图, 项) 读 */

    /* 可视化：md_draw_result 支持 MD_RES_LPR（底层 vis_lpr），用法同 §3 */

    md_result_destroy(res);
    md_model_destroy(m);

    /* 2. 子模型：LPR_DET（仅检测，参数 conf_threshold/nms_threshold/landmarks_per_card 均 'D'） */
    MDModelHandle det = NULL;
    md_model_create(&det, MD_MODEL_LPR_DET, "det.onnx", opt);
    md_model_set_input_size(det, 640, 640);
    md_model_set_param_d(det, "conf_threshold", 0.25);   /* 置信度阈值（默认 0.25） */
    md_model_set_param_d(det, "nms_threshold", 0.45);    /* NMS IoU 阈值（默认 0.5） */
    md_model_set_param_d(det, "landmarks_per_card", 4);  /* 每车牌角点数（默认 4） */
    MDResultHandle det_res = NULL;
    md_model_predict(det, img, &det_res);
    const MDLprItem* ditem = NULL; size_t dn = 0;
    md_result_lpr(det_res, &ditem, &dn);
    for (size_t i = 0; i < dn; i++) {
        const MDPoint* dk = NULL; size_t dkn = 0;
        md_result_lpr_keypoints(det_res, i, &dk, &dkn);  /* 车牌 4 角点 */
        printf("det[%zu] score=%.3f box=(%.0f, %.0f, %.0f, %.0f) kps=%zu\n",
               i, ditem[i].score, ditem[i].x, ditem[i].y, ditem[i].w, ditem[i].h, dkn);
    }
    md_result_destroy(det_res);
    md_model_destroy(det);

    /* 3. 子模型：LPR_REC（仅识别，无参数表）；输入车牌裁剪图 */
    MDModelHandle rec = NULL;
    md_model_create(&rec, MD_MODEL_LPR_REC, "rec.onnx", opt);
    MDImageHandle crop = NULL;
    md_image_from_file(&crop, "plate_crop.jpg");
    MDResultHandle rec_res = NULL;
    md_model_predict(rec, crop, &rec_res);
    const MDLprItem* ritem = NULL; size_t rn = 0;
    md_result_lpr(rec_res, &ritem, &rn);
    if (rn > 0) {
        const char* plate = NULL; const char* color = NULL;
        md_result_plate(rec_res, 0, &plate, &color);
        printf("rec: '%s' color=%s score=%.3f\n",
               plate ? plate : "", color ? color : "", ritem[0].score);
    }
    md_result_destroy(rec_res);
    md_image_destroy(crop);
    md_model_destroy(rec);

    md_image_destroy(img);
    md_option_destroy(opt);
    return 0;
}
```

## 16. 行人属性（`MD_MODEL_PED_ATTR`）+ 行人 ReID（`MD_MODEL_REID`）

行人属性 kind 经 `md_model_create` 创建，`model_path` 用 `|` 串联 **det|cls 两段**：`"det.onnx|cls.onnx"`，结果 kind 为 `MD_RES_ATTR`。行人 ReID kind 创建后 `predict` 输出 L2 归一化 512-d 行人特征，结果 kind 为 `MD_RES_REID`。**C API 未绑定 ReIdGallery**（C API 仅在音频 `md_audio_speaker_search_enroll`/`match` 提供声纹库），行人库检索匹配请用 C++ / Python。

```c
#include <stdio.h>
#include "modeldeploy/md_capi.h"

int main(void) {
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);

    MDImageHandle img = NULL;
    md_image_from_file(&img, "test.jpg");

    /* 1. 行人属性：MD_MODEL_PED_ATTR，det|cls 两段路径 */
    MDModelHandle attr = NULL;
    if (md_model_create(&attr, MD_MODEL_PED_ATTR, "det.onnx|cls.onnx", opt) != MD_OK) {
        fprintf(stderr, "create failed: %s\n", md_get_last_error());
        return 1;
    }
    md_model_set_input_size(attr, 1280, 1280);        /* 检测子模型输入尺寸 */
    md_model_set_cls_input_size(attr, 192, 256);      /* 分类子模型输入尺寸 */
    md_model_set_cls_batch_size(attr, 8);             /* 分类子模型 batch（>0 固定 / -1 自动） */
    md_model_set_param_d(attr, "det_threshold", 0.5); /* 检测阈值（默认 0.5） */

    MDResultHandle res = NULL;
    md_model_predict(attr, img, &res);
    const MDAttrItem* items = NULL;   /* {x,y,w,h,box_score,box_label_id} */
    size_t n = 0;
    md_result_attribute(res, &items, &n);
    for (size_t i = 0; i < n; i++) {
        const float* scores = NULL; size_t sn = 0;
        md_result_attr_scores(res, i, &scores, &sn);  /* 该行人的各属性得分 */
        printf("[%zu] box=(%.0f, %.0f, %.0f, %.0f) label=%d score=%.3f attrs=%zu\n",
               i, items[i].x, items[i].y, items[i].w, items[i].h,
               items[i].box_label_id, items[i].box_score, sn);
    }
    /* 批量：md_result_attribute_batch(bres, g, &items, &n) +
     *        md_result_attr_scores_batch(bres, g, j, &scores, &sn) 按 (图, 项) 读 */
    /* 可视化：md_draw_result 支持 MD_RES_ATTR（底层 vis_attr；
     *         abnormal_ids / show_attr 经 MDDrawOptions 透传） */
    MDDrawOptions dopt = {0};
    dopt.threshold = 0.5; dopt.alpha = 0.3; dopt.show_attr = 1;
    MDImageHandle canvas = NULL;
    md_image_clone(img, &canvas);
    md_draw_result(canvas, res, &dopt);
    md_image_save(canvas, "attr_vis.jpg");
    md_image_destroy(canvas);
    md_result_destroy(res);
    md_model_destroy(attr);

    /* 2. 行人 ReID：MD_MODEL_REID，单段 osnet.onnx，输出 L2 归一化 512 维特征 */
    MDModelHandle reid = NULL;
    md_model_create(&reid, MD_MODEL_REID, "osnet.onnx", opt);
    MDImageHandle crop = NULL;
    md_image_from_file(&crop, "person_crop.jpg");
    md_model_predict(reid, crop, &res);
    const float* emb = NULL; size_t en = 0;
    md_result_reid_embedding(res, 0, &emb, &en);
    printf("embedding dim=%zu\n", en);
    /* 批量：md_result_reid_embedding_batch(bres, g, &emb, &en) 按图读 */
    /* 注意：C API 无 ReIdGallery，行人库检索匹配请用 C++/Python */
    md_result_destroy(res);
    md_model_destroy(reid);
    md_image_destroy(crop);

    md_image_destroy(img);
    md_option_destroy(opt);
    return 0;
}
```

## 接口分组

C API 为**统一分发点**：模型经 `md_model_create(kind, path, opt)` 创建、`md_model_predict` 推理，各类模型差异只体现在 `MDModelKind` 枚举与 `md_result_*` 读结果接口上，**没有** per-model 的 create/predict 函数。

| 模块 | 接口 |
|------|------|
| 选项 | `md_option_create` / `md_option_set_device` / `md_option_set_backend` / `md_option_set_cpu_threads` / `md_option_set_fp16` / `md_option_set_model_path` / `md_option_set_config` 等 + `md_option_destroy` |
| 模型 | `md_model_create` / `md_model_predict` / `md_model_predict_batch` / `md_model_set_input_size` / `md_model_set_param_*` / `md_model_destroy` |
| 结果 | `md_result_count` / `md_result_detection` / `md_result_classification` / `md_result_instance_seg` / `md_result_ocr` / `md_result_face` / `md_result_lpr` / `md_result_attribute` 等 + `md_result_destroy` |
| 图像 | `md_image_from_file` / `md_image_from_bgr24` / `md_image_from_device_nv12` / `md_image_to_host_bytes` / `md_image_plane_bytes` / `md_image_save` / `md_image_destroy` 等 |
| 音频 | `md_audio_asr` / `md_audio_asr_wav` / `md_audio_tts` / `md_audio_tts_stream` / `md_wav_save` |
| 绘制 | `md_draw_rect` / `md_draw_polygon` / `md_draw_text` / `md_draw_result` |
| 视频 | `md_video_*`（解码/编码，见下） |

编译需 `BUILD_CAPI=ON`。

## 运行选项设备

`md_option_set_backend` + `md_option_set_device` 组合选择推理设备（`MD_DEV_CPU/GPU/OPENCL/VULKAN/TPU`，数值 `TPU=2,OPENCL=3,VULKAN=4` 与 C#/Rust 对齐）：

```c
md_option_set_backend(opt, MD_BK_MNN);          // OPENCL/VULKAN 需显式 MNN 后端，否则 fail-closed
md_option_set_device(opt, MD_DEV_OPENCL, 0);    // == MD_OK
md_option_set_device(opt, MD_DEV_VULKAN, 0);    // == MD_OK
```

设备帧 NV12 零拷贝借用见 `md_image_from_device_nv12`（语义同 C++ `ImageData::from_planes(..., Device)`，`dev` 取 `MD_DEV_*`）。

## 视频编解码（`md_video_*`）

视频接口经 **C API 阈值** 暴露给 C/C#/Rust，功能与 C++ `modeldeploy::video` 完全对齐。头文件 `capi/md_capi.h`（常量/句柄/enum 定义不设 `BUILD_` 守卫以保证 ABI 稳定；`.cpp` 实现 `#ifdef BUILD_VIDEO` 守卫，未编译时桩返回 `MD_ERR_UNSUPPORTED_BACKEND`）。

### 句柄与枚举

| 句柄 | 说明 |
|------|------|
| `MDVideoConfigHandle` | 解码/编码配置 |
| `MDVideoDecoderHandle` | 解码器 |
| `MDVideoEncoderHandle` | 编码器 |
| `MDVideoCapabilitiesHandle` | 能力探测 |

枚举：`MDCodecBackend{Auto,FFmpeg,GStreamer}`、`MDHwAccel{Auto,None,Cuda,Vaapi,Sophgo}`、`MDBackpressure{Block,Drop,OverwriteOldest}`、`MDVideoState{Idle,Opening,Running,Reconnecting,Eof,Error,Closed}`；统计结构 `MDVideoStats{frames_in,frames_out,dropped,avg_decode_ms,avg_encode_ms,reconnect_count,error_count}`。

### 配置

```c
MDVideoConfigHandle cfg; md_video_config_create(&cfg);
md_video_config_set_backend(cfg, MD_CODEC_FFMPEG);
md_video_config_set_hw_accel(cfg, MD_HW_CUDA);        // 或默认 Auto
md_video_config_set_device_only(cfg, 1);              // GPU 设备内存直通
md_video_config_set_backpressure(cfg, MD_BP_BLOCK);
md_video_config_set_async_queue_size(cfg, 30);
md_video_config_set_pooling(cfg, 1);
// 编码侧追加：md_video_config_set_fps/set_bitrate_kbps/set_gop/set_codec/set_preset/set_format
md_video_config_destroy(cfg);                          // 用完释放
```

### 能力探测

```c
MDVideoCapabilitiesHandle cap; md_video_capabilities_create(&cap);
int ff = 0, gst = 0; size_t n = 0;
md_video_capabilities_ffmpeg(cap, &ff);   md_video_capabilities_gstreamer(cap, &gst);
md_video_capabilities_hw_decoder_count(cap, &n);
// md_video_capabilities_hw_decoder(cap, i, &name) 逐个取硬解名
md_video_capabilities_destroy(cap);
```

### 解码

```c
MDVideoConfigHandle cfg; md_video_config_create(&cfg);
MDVideoDecoderHandle dec; md_video_decoder_create(cfg, &dec);
md_video_decoder_open(dec, "demo.mp4");

// 同步取帧：*frame 归调用方所有，用后必须 md_image_destroy
MDImageHandle frame; uint64_t pts;
while (md_video_decoder_read_frame(dec, &frame, &pts) == MD_OK) {
    /* 处理 NV12 帧 */ md_image_destroy(frame);
}
MDVideoStats st; md_video_decoder_stats(dec, &st);   // 统计
md_video_decoder_destroy(dec);
md_video_config_destroy(cfg);
```

**异步回调**：`md_video_decoder_set_callback(dec, cb, userdata)` 注册 `MDVideoFrameCb(frame, pts_ms, userdata)`，再 `md_video_decoder_start(dec)`。回调内 `frame` 同样是**自有所有权**，用后必须 `md_image_destroy`；回调自后台线程投递，注意线程安全。

### 编码

```c
MDVideoEncoderHandle enc; md_video_encoder_create(cfg, &enc);
md_video_encoder_open(enc, "out.mp4", 1280, 720, 25);   // (url,w,h,src_fps)
md_video_encoder_encode(enc, img, pts_ms);               // 同步；img 生命周期须覆盖本次调用
// 或异步：encode_async / start_async / stop_async
md_video_encoder_close(enc);            // 必须 close，mp4 尾部索引(moov)在此写盘
md_video_encoder_destroy(enc);
```

> **所有权关键点**：解码回调/同步读返回的 `MDImageHandle` 由接收方释放（`md_image_destroy`）；编码 `encode` 的 `img` 只借用、不持有，须保证调用期间有效。设备显存（`device_only` / `gpu_direct_input`）零拷贝直通，无需主机往返。
> 详见 [视频接口总览](../video/api.md)。
