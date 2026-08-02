//
// Created by aichao on 2025/8/2.
// ARM64 NEON 融合预处理内核：4 像素/组（aarch64 基线 ISA，无需 target attribute）。
//

#include <cstdint>
#include <algorithm>
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define MD_HAS_NEON 1
#endif

#include "fused_preproc_simd.h"

#if MD_HAS_NEON

namespace modeldeploy::vision {
namespace {

inline void store_row_pad_neon(float* dst, int plane, int base, int n, float pad_value) {
    const float32x4_t padv = vdupq_n_f32(pad_value);
    int x = 0;
    for (; x + 4 <= n; x += 4) {
        vst1q_f32(dst + 0 * plane + base + x, padv);
        vst1q_f32(dst + 1 * plane + base + x, padv);
        vst1q_f32(dst + 2 * plane + base + x, padv);
    }
    for (; x < n; ++x) {
        dst[0 * plane + base + x] = pad_value;
        dst[1 * plane + base + x] = pad_value;
        dst[2 * plane + base + x] = pad_value;
    }
}

} // namespace

void fused_preproc_neon(const uint8_t* src, int src_w, int src_h,
                        float* dst, int dst_w, int dst_h,
                        float origin_x, float origin_y,
                        float scale_x, float scale_y,
                        const float* alpha, const float* beta,
                        bool swap_rb, float pad_value) {
    const float inv_scale_x = 1.0f / scale_x;
    const float inv_scale_y = 1.0f / scale_y;
    const float origin_shift_x = origin_x / scale_x;
    const float origin_shift_y = origin_y / scale_y;
    const float src_w_f = static_cast<float>(src_w);
    const float src_h_f = static_cast<float>(src_h);
    const int plane = dst_h * dst_w;
    const int n = dst_w;

    const float32x4_t a0 = vdupq_n_f32(alpha[0]);
    const float32x4_t b0 = vdupq_n_f32(beta[0]);
    const float32x4_t a1 = vdupq_n_f32(alpha[1]);
    const float32x4_t b1 = vdupq_n_f32(beta[1]);
    const float32x4_t a2 = vdupq_n_f32(alpha[2]);
    const float32x4_t b2 = vdupq_n_f32(beta[2]);

    for (int y = 0; y < dst_h; ++y) {
        const int base = y * dst_w;
        const float src_yf = static_cast<float>(y) * inv_scale_y - origin_shift_y;
        if (src_yf < 0.0f || src_yf >= src_h_f) {
            store_row_pad_neon(dst, plane, base, n, pad_value);
            continue;
        }
        const int src_y = static_cast<int>(src_yf);
        const uint8_t* src_row = src + src_y * src_w * 3;

        int x = 0;
        for (; x + 4 <= dst_w; x += 4) {
            float r[4], g[4], b[4];
            for (int i = 0; i < 4; ++i) {
                const int xx = x + i;
                const float src_xf = static_cast<float>(xx) * inv_scale_x - origin_shift_x;
                if (src_xf >= 0.0f && src_xf < src_w_f) {
                    const int sxi = static_cast<int>(src_xf);
                    const uint8_t* p = src_row + sxi * 3;
                    const float pb = p[0], pg = p[1], pr = p[2];
                    if (swap_rb) { r[i] = pr; g[i] = pg; b[i] = pb; }
                    else { r[i] = pb; g[i] = pg; b[i] = pr; }
                } else {
                    r[i] = g[i] = b[i] = pad_value;
                }
            }
            vst1q_f32(dst + 0 * plane + base + x, vfmaq_f32(b0, vld1q_f32(r), a0));
            vst1q_f32(dst + 1 * plane + base + x, vfmaq_f32(b1, vld1q_f32(g), a1));
            vst1q_f32(dst + 2 * plane + base + x, vfmaq_f32(b2, vld1q_f32(b), a2));
        }
        for (; x < dst_w; ++x) {
            const float src_xf = static_cast<float>(x) * inv_scale_x - origin_shift_x;
            float rv, gv, bv;
            if (src_xf >= 0.0f && src_xf < src_w_f) {
                const int sxi = static_cast<int>(src_xf);
                const uint8_t* p = src_row + sxi * 3;
                const float pb = p[0], pg = p[1], pr = p[2];
                if (swap_rb) { rv = pr; gv = pg; bv = pb; }
                else { rv = pb; gv = pg; bv = pr; }
            } else {
                rv = gv = bv = pad_value;
            }
            dst[0 * plane + base + x] = rv * alpha[0] + beta[0];
            dst[1 * plane + base + x] = gv * alpha[1] + beta[1];
            dst[2 * plane + base + x] = bv * alpha[2] + beta[2];
        }
    }
}


void fusion_rpnp_neon(const uint8_t* src, int src_w, int src_h,
                      float* dst, int dst_w, int dst_h,
                      int resize_w, int resize_h,
                      const float* alpha, const float* beta,
                      const float* pad) {
    const float kx = static_cast<float>(src_w) / resize_w;
    const float ky = static_cast<float>(src_h) / resize_h;
    const int last_sx = src_w - 1;
    const int last_sy = src_h - 1;
    const int plane = dst_h * dst_w;
    for (int y = 0; y < dst_h; ++y) {
        const int base = y * dst_w;
        const bool row_pad = y >= resize_h;
        for (int x = 0; x < dst_w; ++x) {
            const int idx = base + x;
            if (row_pad || x >= resize_w) {
                dst[0 * plane + idx] = pad[0];
                dst[1 * plane + idx] = pad[1];
                dst[2 * plane + idx] = pad[2];
                continue;
            }
            const int sx = std::min(static_cast<int>(x * kx), last_sx);
            const int sy = std::min(static_cast<int>(y * ky), last_sy);
            const uint8_t* p = src + (sy * src_w + sx) * 3;
            dst[0 * plane + idx] = p[2] * alpha[0] + beta[0];
            dst[1 * plane + idx] = p[1] * alpha[1] + beta[1];
            dst[2 * plane + idx] = p[0] * alpha[2] + beta[2];
        }
    }
}
} // namespace modeldeploy::vision
#endif // MD_HAS_NEON
