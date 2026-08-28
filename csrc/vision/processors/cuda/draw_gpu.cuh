#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "core/md_decl.h"

namespace modeldeploy::vision {
    /// GPU 绘制盒：边框 + 半透明填充 + 顶部 8x16 标签条
    struct GpuDrawBox {
        int x1, y1, x2, y2;
        float score;
        int label_id;
        uint8_t r, g, b;   // BGR
        char label[32];
    };

    /// 在 BGR 图上绘制检测框（bgr/d_boxes 为设备指针；host 指针会自动上传并回拷）
    /// @return false 表示 CUDA 调用失败
    MODELDEPLOY_CXX_EXPORT bool draw_boxes_gpu(
        uint8_t* bgr, int width, int height,
        const GpuDrawBox* d_boxes, int num_boxes,
        float alpha = 0.15f, cudaStream_t stream = nullptr);

    // ── NV12 设备侧就地绘制（y/uv 为设备指针；颜色 BGR→YUV BT.601）──
    MODELDEPLOY_CXX_EXPORT bool draw_rect_nv12_gpu(
        uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
        float x, float yo, float rw, float rh,
        uint8_t r, uint8_t g, uint8_t b, int thickness,
        cudaStream_t stream = nullptr);

    MODELDEPLOY_CXX_EXPORT bool draw_polygon_nv12_gpu(
        uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
        const float* xs, const float* ys, int npts,
        uint8_t r, uint8_t g, uint8_t b, int thickness,
        cudaStream_t stream = nullptr);

    MODELDEPLOY_CXX_EXPORT bool draw_points_nv12_gpu(
        uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
        const float* xs, const float* ys, int npts,
        uint8_t r, uint8_t g, uint8_t b, int radius,
        cudaStream_t stream = nullptr);

    MODELDEPLOY_CXX_EXPORT bool draw_text_nv12_gpu(
        uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
        float x, float yo, const char* text,
        uint8_t r, uint8_t g, uint8_t b, int font_size,
        cudaStream_t stream = nullptr);

    MODELDEPLOY_CXX_EXPORT bool fill_rect_nv12_gpu(
        uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
        int x0, int y0, int x1, int y1,
        uint8_t r, uint8_t g, uint8_t b, float alpha,
        cudaStream_t stream = nullptr);

    MODELDEPLOY_CXX_EXPORT bool fill_polygon_nv12_gpu(
        uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
        const float* xs, const float* ys, int npts,
        uint8_t r, uint8_t g, uint8_t b, float alpha,
        cudaStream_t stream = nullptr);

    MODELDEPLOY_CXX_EXPORT bool draw_line_nv12_gpu(
        uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
        float x0, float y0, float x1, float y1,
        uint8_t r, uint8_t g, uint8_t b, int thickness,
        cudaStream_t stream = nullptr);

    MODELDEPLOY_CXX_EXPORT bool draw_text_cjk_nv12_gpu(
        uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
        float x, float yo, const char* text,
        uint8_t r, uint8_t g, uint8_t b, int font_size, int max_chars,
        cudaStream_t stream = nullptr);

    // 主机 OpenCV 渲染出的 BGR 文本标签 sprite → BGR→YUV 直写 NV12（不透明，按 ox,oy 锚定左上角）
    MODELDEPLOY_CXX_EXPORT bool blit_bgr_nv12_gpu(
        uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
        const uint8_t* bgr, int sw, int sh, int sstep,
        int ox, int oy, cudaStream_t stream = nullptr);

    MODELDEPLOY_CXX_EXPORT bool overlay_labels_nv12_gpu(
        uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
        const uint8_t* d_labels, int lw, int lh, float alpha,
        cudaStream_t stream = nullptr);

    MODELDEPLOY_CXX_EXPORT bool overlay_depth_nv12_gpu(
        uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
        const uint8_t* d_depth8, int dw, int dh, bool colorize, float alpha,
        cudaStream_t stream = nullptr);

    MODELDEPLOY_CXX_EXPORT bool overlay_mask_nv12_gpu(
        uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
        int bx0, int by0, int bw, int bh,
        const uint8_t* d_mask, int mw, int mh,
        uint8_t r, uint8_t g, uint8_t b, float alpha,
        cudaStream_t stream = nullptr);
}
