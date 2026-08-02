//
// Created by aichao on 2025/8/2.
// ARMv9 SVE 融合预处理内核（变长向量）。
// 需要编译时 __ARM_FEATURE_SVE（由 target attribute / 编译选项启用）。
//

#include <cstdint>
#include <algorithm>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define MD_HAS_SVE 1
#endif

#include "fused_preproc_simd.h"

#if MD_HAS_SVE
#if defined(__GNUC__) || defined(__clang__)
#define MD_TARGET_SVE __attribute__((target("arch=armv9-a+sve")))
#else
#define MD_TARGET_SVE
#endif

namespace modeldeploy::vision {

MD_TARGET_SVE void fused_preproc_sve(const uint8_t* src, int src_w, int src_h,
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
    // SVE 向量宽度（float 元素数），运行时确定，上限 64（2048-bit）
    const uint64_t lanes = svcntw();
    const int max_lanes = 64;

    const svfloat32_t a0 = svdup_f32(alpha[0]);
    const svfloat32_t b0 = svdup_f32(beta[0]);
    const svfloat32_t a1 = svdup_f32(alpha[1]);
    const svfloat32_t b1 = svdup_f32(beta[1]);
    const svfloat32_t a2 = svdup_f32(alpha[2]);
    const svfloat32_t b2 = svdup_f32(beta[2]);
    const svfloat32_t padv = svdup_f32(pad_value);

    for (int y = 0; y < dst_h; ++y) {
        const int base = y * dst_w;
        const float src_yf = static_cast<float>(y) * inv_scale_y - origin_shift_y;
        if (src_yf < 0.0f || src_yf >= src_h_f) {
            int x = 0;
            for (; x + static_cast<int>(lanes) <= n; x += static_cast<int>(lanes)) {
                svst1_f32(svptrue_b32(), dst + 0 * plane + base + x, padv);
                svst1_f32(svptrue_b32(), dst + 1 * plane + base + x, padv);
                svst1_f32(svptrue_b32(), dst + 2 * plane + base + x, padv);
            }
            for (; x < n; ++x) {
                dst[0 * plane + base + x] = pad_value;
                dst[1 * plane + base + x] = pad_value;
                dst[2 * plane + base + x] = pad_value;
            }
            continue;
        }
        const int src_y = static_cast<int>(src_yf);
        const uint8_t* src_row = src + src_y * src_w * 3;

        int x = 0;
        for (; x + static_cast<int>(lanes) <= n; x += static_cast<int>(lanes)) {
            float r[max_lanes], g[max_lanes], b[max_lanes];
            for (uint64_t i = 0; i < lanes; ++i) {
                const int xx = x + static_cast<int>(i);
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
            const svbool_t pg = svptrue_b32();
            svst1_f32(pg, dst + 0 * plane + base + x, svmla_f32(b0, svld1_f32(pg, r), a0));
            svst1_f32(pg, dst + 1 * plane + base + x, svmla_f32(b1, svld1_f32(pg, g), a1));
            svst1_f32(pg, dst + 2 * plane + base + x, svmla_f32(b2, svld1_f32(pg, b), a2));
        }
        for (; x < n; ++x) {
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


void fusion_rpnp_sve(const uint8_t* src, int src_w, int src_h,
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
#endif // MD_HAS_SVE
