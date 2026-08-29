//
// BMCV 预处理桥接：把 Sophgo BMCV 调用隔离在此文件。
// 该文件只依赖 libsophon 头(bmlib_runtime/bmcv_api*)，不包含任何 ModelDeploy 头，
// 避免 bmcv_api_ext.h 的 ROTATE_*/FLIP_* 枚举与项目 basic_types.h 冲突。
//
#pragma once
#include <cstdint>
#include <memory>

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

    // ── NV12 设备侧就地绘制（TPU 设备显存 x/uv，bm_image attach 两平面）──
    // 坐标均为原图坐标；r/g/b 为 BGR 颜色分量（0-255，与 CPU/CUDA 后端一致）。
    // 返回 0 成功；非 0 失败。
    int md_bmcv_draw_rect_nv12(void* handle, void* y_mem, void* uv_mem, int w, int h,
                               int x1, int y1, int x2, int y2,
                               int r, int g, int b, int thickness);
    int md_bmcv_draw_polygon_nv12(void* handle, void* y_mem, void* uv_mem, int w, int h,
                                  const float* xs, const float* ys, int npts,
                                  int r, int g, int b, int thickness);
    int md_bmcv_draw_points_nv12(void* handle, void* y_mem, void* uv_mem, int w, int h,
                                 const float* xs, const float* ys, int npts, int radius,
                                 int r, int g, int b);
    int md_bmcv_draw_text_nv12(void* handle, void* y_mem, void* uv_mem, int w, int h,
                               int x, int y, const char* text,
                               int r, int g, int b, int font_size);
    // 任意线段组（骨骼/骨架）：ns 段，第 i 段端点 (sx[i],sy[i])→(ex[i],ey[i])，同一颜色。
    int md_bmcv_draw_lines_nv12(void* handle, void* y_mem, void* uv_mem, int w, int h,
                                const float* sx, const float* sy,
                                const float* ex, const float* ey, int ns,
                                int r, int g, int b, int thickness);

    // ── TPU 设备帧中间算子（显存→显存，输出新分配设备内存，零拷贝）──
    // y_mem/uv_mem: 源 NV12 设备平面地址（Device::TPU）。输出设备内存由本函数分配，
    // 经 *owner 保活（出参 ImageData 持有；析构时释放设备显存，需 bm_handle_t 故由本层封装）。
    // 成功返回 0 并写出设备地址/尺寸与 owner；失败返回非 0（不分配，调用方 fail-closed 返回 false）。
    //
    // crop：现成 NV12 就地裁剪，输出仍为 NV12（两平面）。
    int md_bmcv_crop_nv12_devmem(void* handle, void* y_mem, void* uv_mem,
                                 int src_w, int src_h,
                                 float x, float y, float w, float h,
                                 void** out_y, void** out_uv, int* out_w, int* out_h,
                                 std::shared_ptr<void>* owner);
    // rotate：flag 0=90, 1=180, 2=270（与 RotateFlags 数值一致）；输出仍为 NV12，90/270 时 W/H 互换。
    int md_bmcv_rotate_nv12_devmem(void* handle, void* y_mem, void* uv_mem,
                                   int src_w, int src_h, int flag,
                                   void** out_y, void** out_uv, int* out_w, int* out_h,
                                   std::shared_ptr<void>* owner);
    // cvt_color：cvt_kind 0=PKG_BGR, 1=PLA_BGR, 2=GRAY；输出为单平面设备内存，地址经 out_data 返回。
    int md_bmcv_cvtcolor_nv12_devmem(void* handle, void* y_mem, void* uv_mem,
                                     int src_w, int src_h, int cvt_kind,
                                     void** out_data, int* out_w, int* out_h,
                                     std::shared_ptr<void>* owner);
} // namespace modeldeploy::vision
