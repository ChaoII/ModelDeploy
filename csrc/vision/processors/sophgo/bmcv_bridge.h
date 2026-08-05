//
// BMCV 预处理桥接：把 Sophgo BMCV 调用隔离在此文件。
// 该文件只依赖 libsophon 头(bmlib_runtime/bmcv_api*)，不包含任何 ModelDeploy 头，
// 避免 bmcv_api_ext.h 的 ROTATE_*/FLIP_* 枚举与项目 basic_types.h 冲突。
//
#pragma once
#include <cstdint>

namespace modeldeploy::vision {
    // BMCV letterbox(resize+pad) + BGR->RGB + alpha/beta 仿射，输出 FP32 CHW 平面到 dst。
    // handle: bm_handle_t（void* 透传，来自 SophgoProcessorBackend）
    // bgr:    输入 HWC BGR/RGB uint8（swap_rb=1 表示输入为 BGR，需转 RGB）
    // dst:    输出连续 [C,H,W] FP32（3 * dst_h * dst_w 个 float）
    // 返回 0 成功；非 0 失败（无 BMCV / 设备不可用）
    int md_bmcv_letterbox_normalize(void* handle,
                                    const uint8_t* bgr, int src_w, int src_h,
                                    float* dst, int dst_w, int dst_h,
                                    int pad_w, int pad_h, int resize_w, int resize_h,
                                    float alpha0, float alpha1, float alpha2,
                                    unsigned char pad_val, int swap_rb);

    // 设备内存版本（完整）：创建输出 bm_image(FP32 RGB_PLANAR, 设备内存)并完成 BMCV 处理，
    // 通过 out_img 返回该 bm_image 的 void*（调用方持有，用 md_bmcv_image_destroy 释放），
    // 通过 dev_mem 返回其连续设备内存（bm_device_mem_t*，零拷贝推理用）。
    // 返回 0 成功；非 0 失败。
    int md_bmcv_letterbox_normalize_device_full(void* handle,
                                                const uint8_t* bgr, int src_w, int src_h,
                                                void** out_img, void* dev_mem,
                                                int dst_w, int dst_h,
                                                int pad_w, int pad_h, int resize_w, int resize_h,
                                                float alpha0, float alpha1, float alpha2,
                                                unsigned char pad_val, int swap_rb);

    // 官方零拷贝预处理（SOPHON-DEMO YOLOv8_plus_det 方式）：
    // out_img 为调用方已 create 的 bm_image（FP32 RGB_PLANAR），input_mem 为
    // bmrt_tensor 分配的输入设备内存（bm_device_mem_t*）。内部 bm_image_attach(out_img,
    // input_mem) 后，vpp letterbox + convert_to 直接把结果写入该输入设备内存，
    // 供 bmrt_launch_tensor 零拷贝推理（无 D2H 读回 / H2D 上传）。
    // 注意顺序：必须 attach 之后 convert_to，与官方一致（否则 launch 读到空内存挂 TPU）。
    // 返回 0 成功；非 0 失败。src 上传与 letter 中间 bm_image 在内部创建/销毁。
    int md_bmcv_letterbox_normalize_attach(void* handle,
                                           const uint8_t* bgr, int src_w, int src_h,
                                           void* out_img, void* input_mem,
                                           int dst_w, int dst_h,
                                           int pad_w, int pad_h, int resize_w, int resize_h,
                                           float alpha0, float alpha1, float alpha2,
                                           unsigned char pad_val, int swap_rb);

    // 创建输出 bm_image（FP32 RGB_PLANAR，dst_h*dst_w），仅创建不分配设备内存
    // （由 md_bmcv_letterbox_normalize_attach attach 到输入设备内存）。返回新对象 void*
    // 或 nullptr 失败；调用方用 md_bmcv_image_destroy 释放。
    void* md_bmcv_image_create(void* handle, int dst_w, int dst_h);

    // 销毁 device 版预处理创建的 bm_image（释放设备内存）
    int md_bmcv_image_destroy(void* img);
} // namespace modeldeploy::vision
