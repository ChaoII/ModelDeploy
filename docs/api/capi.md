# C API（`md_*` 前缀）

面向 C/C++ 嵌入式、其他语言 FFI 桥接。统一返回 `MDStatusCode`。

```c
#include "modeldeploy/md_model_capi.h"

// 创建模型
MDModel model = md_create_detection_model("yolo11n.onnx", md_create_default_runtime_option());

// 设置输入尺寸
md_set_detection_input_size(model, 640, 640);

// 推理
MDDetectionResults results;
md_detection_predict(model, img, &results);

// 读取结果
int n = md_get_detection_result_size(results);
// ...

// 释放
md_free_detection_results(results);
md_free_detection_model(model);
```

## 接口分组

| 模块 | 接口 |
|------|------|
| 检测 | `md_create_detection_model` / `md_detection_predict` |
| 分类 | `md_create_classification_model` / `md_classification_predict` |
| 分割 | `md_create_instance_seg_model` / `md_instance_seg_predict` |
| 姿态 | `md_create_keypoint_model` / `md_keypoint_predict` |
| 旋转框 | `md_create_obb_model` / `md_obb_predict` |
| 人脸 | `md_create_face_det/rec/age/gender/as_*_model` |
| 车牌 | `md_create_lpr_*_model` |
| OCR | `md_create_ocr_model` / `md_ocr_model_predict` |
| 行人属性 | `md_create_attr_model` / `md_attr_predict` |
| 图像 | `md_read_image` / `md_save_image` / `md_from_bgr24` / `md_image_to_host_bytes` / `md_image_plane_bytes` 等 |
| 绘制 | `md_draw_rect` / `md_draw_polygon` / `md_draw_text` |
| 视频 | `md_video_*`（解码/编码，见下） |

编译需 `BUILD_CAPI=ON`。

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
