//
// Created by aichao on 2025/8/2.
// CPU ISA 运行时探测与融合预处理内核派发。
//

#include <cstdint>
#include <algorithm>
#include "fused_preproc_simd.h"

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#if defined(_M_X64) || defined(__x86_64__)
#define MD_X86 1
#endif
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#define MD_ARM64 1
#endif

#if defined(MD_X86)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

#if defined(MD_ARM64) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

namespace modeldeploy::vision {
namespace {

// 标量兜底内核（与 CpuProcessorBackend::fused_preprocess 原逻辑一致）
void fused_preproc_scalar(const uint8_t* src, int src_w, int src_h,
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

    for (int y = 0; y < dst_h; ++y) {
        const int base = y * dst_w;
        const float src_yf = static_cast<float>(y) * inv_scale_y - origin_shift_y;
        for (int x = 0; x < dst_w; ++x) {
            const float src_xf = static_cast<float>(x) * inv_scale_x - origin_shift_x;
            if (src_xf < 0.0f || src_xf >= src_w_f || src_yf < 0.0f || src_yf >= src_h_f) {
                dst[0 * plane + base + x] = pad_value;
                dst[1 * plane + base + x] = pad_value;
                dst[2 * plane + base + x] = pad_value;
                continue;
            }
            const int sx = static_cast<int>(src_xf);
            const int sy = static_cast<int>(src_yf);
            const uint8_t* p = src + (sy * src_w + sx) * 3;
            const float pb = p[0], pg = p[1], pr = p[2];
            float rv, gv, bv;
            if (swap_rb) { rv = pr; gv = pg; bv = pb; }
            else { rv = pb; gv = pg; bv = pr; }
            dst[0 * plane + base + x] = rv * alpha[0] + beta[0];
            dst[1 * plane + base + x] = gv * alpha[1] + beta[1];
            dst[2 * plane + base + x] = bv * alpha[2] + beta[2];
        }
    }
}

// OCR det per-channel-pad scalar kernel（resize + pad right/bottom + swap + affine）
void fusion_rpnp_scalar(const uint8_t* src, int src_w, int src_h,
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
            const float pb = p[0], pg = p[1], pr = p[2];
            dst[0 * plane + idx] = pr * alpha[0] + beta[0];
            dst[1 * plane + idx] = pg * alpha[1] + beta[1];
            dst[2 * plane + idx] = pb * alpha[2] + beta[2];
        }
    }
}

} // namespace

// 外部内核声明（modeldeploy::vision 作用域，各 ISA 实现文件按编译平台定义）
void fused_preproc_avx2(const uint8_t*, int, int, float*, int, int,
                        float, float, float, float,
                        const float*, const float*, bool, float);
void fused_preproc_avx512(const uint8_t*, int, int, float*, int, int,
                          float, float, float, float,
                          const float*, const float*, bool, float);
void fused_preproc_neon(const uint8_t*, int, int, float*, int, int,
                        float, float, float, float,
                        const float*, const float*, bool, float);
void fused_preproc_sve(const uint8_t*, int, int, float*, int, int,
                       float, float, float, float,
                       const float*, const float*, bool, float);

// OCR det per-channel-pad kernels（外部 ISA 实现）
void fusion_rpnp_avx2(const uint8_t*, int, int, float*, int, int, int, int,
                      const float*, const float*, const float*);
void fusion_rpnp_avx512(const uint8_t*, int, int, float*, int, int, int, int,
                        const float*, const float*, const float*);
void fusion_rpnp_neon(const uint8_t*, int, int, float*, int, int, int, int,
                      const float*, const float*, const float*);
void fusion_rpnp_sve(const uint8_t*, int, int, float*, int, int, int, int,
                     const float*, const float*, const float*);

FusedPreprocKernel get_fused_preproc_kernel() {
#if defined(MD_ARM64)
#if defined(__ARM_FEATURE_SVE)
#if defined(__linux__)
    // 运行时探测 SVE hwcap（仅在编译带 SVE 支持时）
    if (getauxval(AT_HWCAP) & HWCAP_SVE) {
        return fused_preproc_sve;
    }
#else
    // 非 Linux（如 Android/macOS arm64），无 SVE 探测接口则直接启用（编译即保证支持）
    return fused_preproc_sve;
#endif
#endif
    return fused_preproc_neon;
#elif defined(MD_X86)
    // 优先 AVX512，其次 AVX2，兜底标量
#if defined(_MSC_VER)
    {
        int cpu_info[4] = {0};
        __cpuidex(cpu_info, 1, 0);
        const bool osxsave = (cpu_info[2] & (1u << 27)) != 0;
        const uint64_t xcr0 = _xgetbv(0);
        const bool os_ymm = osxsave && (xcr0 & 0x6) == 0x6;         // XMM|YMM
        const bool os_zmm = osxsave && (xcr0 & 0xE6) == 0xE6;      // +opmask|ZMM_hi|ZMM
        __cpuidex(cpu_info, 7, 0);
        const bool has_avx2 = (cpu_info[1] & (1u << 5)) != 0;
        const bool has_avx512f = (cpu_info[1] & (1u << 16)) != 0;
        if (has_avx512f && os_zmm) return fused_preproc_avx512;
        if (has_avx2 && os_ymm) return fused_preproc_avx2;
        return fused_preproc_scalar;
    }
#elif defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("avx512f")) return fused_preproc_avx512;
    if (__builtin_cpu_supports("avx2")) return fused_preproc_avx2;
    return fused_preproc_scalar;
#else
    return fused_preproc_scalar;
#endif
#else
    return fused_preproc_scalar;
#endif
}

FusedPreprocPadKernel get_fusion_rpnp_kernel() {
#if defined(MD_ARM64)
#if defined(__ARM_FEATURE_SVE)
#if defined(__linux__)
    if (getauxval(AT_HWCAP) & HWCAP_SVE) {
        return fusion_rpnp_sve;
    }
#else
    return fusion_rpnp_sve;
#endif
#endif
    return fusion_rpnp_neon;
#elif defined(MD_X86)
#if defined(_MSC_VER)
    {
        int cpu_info[4] = {0};
        __cpuidex(cpu_info, 1, 0);
        const bool osxsave = (cpu_info[2] & (1u << 27)) != 0;
        const uint64_t xcr0 = _xgetbv(0);
        const bool os_ymm = osxsave && (xcr0 & 0x6) == 0x6;
        const bool os_zmm = osxsave && (xcr0 & 0xE6) == 0xE6;
        __cpuidex(cpu_info, 7, 0);
        const bool has_avx2 = (cpu_info[1] & (1u << 5)) != 0;
        const bool has_avx512f = (cpu_info[1] & (1u << 16)) != 0;
        if (has_avx512f && os_zmm) return fusion_rpnp_avx512;
        if (has_avx2 && os_ymm) return fusion_rpnp_avx2;
        return fusion_rpnp_scalar;
    }
#elif defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("avx512f")) return fusion_rpnp_avx512;
    if (__builtin_cpu_supports("avx2")) return fusion_rpnp_avx2;
    return fusion_rpnp_scalar;
#else
    return fusion_rpnp_scalar;
#endif
#else
    return fusion_rpnp_scalar;
#endif
}

} // namespace modeldeploy::vision
