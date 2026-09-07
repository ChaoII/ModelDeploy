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
