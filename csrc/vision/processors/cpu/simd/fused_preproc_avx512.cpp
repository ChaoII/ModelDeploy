//
// Created by aichao on 2025/8/2.
// AVX512 融合预处理内核：16 像素/组。
//

#include <cstdint>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define MD_X86_SIMD 1
#endif
#include "fused_preproc_simd.h"

#if MD_X86_SIMD
#if defined(__GNUC__) || defined(__clang__)
#define MD_TARGET_AVX512 __attribute__((target("avx512f")))
#else
#define MD_TARGET_AVX512
#endif

namespace modeldeploy::vision {
namespace {

MD_TARGET_AVX512 inline void store_row_pad512(float* dst, int plane, int base, int w, int n, __m512 padv) {
    int x = 0;
    for (; x + 16 <= n; x += 16) {
        _mm512_storeu_ps(dst + 0 * plane + base + x, padv);
        _mm512_storeu_ps(dst + 1 * plane + base + x, padv);
        _mm512_storeu_ps(dst + 2 * plane + base + x, padv);
    }
    const float pv = _mm512_cvtss_f32(padv);
    for (; x < n; ++x) {
        dst[0 * plane + base + x] = pv;
        dst[1 * plane + base + x] = pv;
        dst[2 * plane + base + x] = pv;
    }
}

} // namespace

MD_TARGET_AVX512 void fused_preproc_avx512(const uint8_t* src, int src_w, int src_h,
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

    const __m512 a0 = _mm512_set1_ps(alpha[0]);
    const __m512 b0 = _mm512_set1_ps(beta[0]);
    const __m512 a1 = _mm512_set1_ps(alpha[1]);
    const __m512 b1 = _mm512_set1_ps(beta[1]);
    const __m512 a2 = _mm512_set1_ps(alpha[2]);
    const __m512 b2 = _mm512_set1_ps(beta[2]);
    const __m512 padv = _mm512_set1_ps(pad_value);

    for (int y = 0; y < dst_h; ++y) {
        const int base = y * dst_w;
        const float src_yf = static_cast<float>(y) * inv_scale_y - origin_shift_y;
        if (src_yf < 0.0f || src_yf >= src_h_f) {
            store_row_pad512(dst, plane, base, dst_w, n, padv);
            continue;
        }
        const int src_y = static_cast<int>(src_yf);
        const uint8_t* src_row = src + src_y * src_w * 3;

        int x = 0;
        for (; x + 16 <= dst_w; x += 16) {
            float r[16], g[16], b[16];
            for (int i = 0; i < 16; ++i) {
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
            _mm512_storeu_ps(dst + 0 * plane + base + x, _mm512_fmadd_ps(_mm512_loadu_ps(r), a0, b0));
            _mm512_storeu_ps(dst + 1 * plane + base + x, _mm512_fmadd_ps(_mm512_loadu_ps(g), a1, b1));
            _mm512_storeu_ps(dst + 2 * plane + base + x, _mm512_fmadd_ps(_mm512_loadu_ps(b), a2, b2));
        }
        // 尾部（<16）标量
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

} // namespace modeldeploy::vision
#endif // MD_X86_SIMD
