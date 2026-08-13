//
// BMCV 预处理桥接：把 Sophgo BMCV 调用隔离在此文件。
// 该文件只依赖 libsophon 头(bmlib_runtime/bmcv_api*)，不包含任何 ModelDeploy 头，
// 避免 bmcv_api_ext.h 的 ROTATE_*/FLIP_* 枚举与项目 basic_types.h 冲突。
//
#pragma once
#include <cstdint>

namespace modeldeploy::vision {
    // BGR/RGB(swap_rb) letterbox + alpha/beta 仿射，FP32 CHW 结果直接写入 dev_mem（bm_device_mem_t*，
    // 由调用方持有的输入设备内存，零拷贝推理用）。返回 0 成功；非 0 失败。
    int md_bmcv_letterbox_normalize_to_devmem(void* handle,
                                              const uint8_t* bgr, int src_w, int src_h,
                                              void* dev_mem, int dst_w, int dst_h,
                                              int pad_w, int pad_h, int resize_w, int resize_h,
                                              float alpha0, float alpha1, float alpha2,
                                              unsigned char pad_val, int swap_rb);

    // NV12(host Y/UV 平面) letterbox + 归一化，FP32 CHW 结果写入 dev_mem（bm_device_mem_t*）。
    // step_y/step_uv 为平面行步长（通常等于 width；非连续时内部整理拷贝）。
    // src_is_device=true 时 src_y/src_uv 为 TPU 设备地址（Y 与 UV 连续），跳过 H2D 直接 attach。
    // 返回 0 成功；非 0 失败。
    int md_bmcv_nv12_letterbox_normalize_to_devmem(void* handle,
                                                   const uint8_t* src_y, const uint8_t* src_uv,
                                                   int src_w, int src_h,
                                                   int step_y, int step_uv,
                                                   void* dev_mem, int dst_w, int dst_h,
                                                   float alpha0, float alpha1, float alpha2,
                                                   unsigned char pad_val,
                                                   bool src_is_device = false);
} // namespace modeldeploy::vision
