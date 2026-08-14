//
// Created by aichao on 2025/8/2.
// CPU SIMD 融合预处理派发。
//
#pragma once

#include <cstdint>
#include "core/md_decl.h"

namespace modeldeploy::vision {

// 融合预处理 SIMD 内核公共签名。
// src: BGR 打包 uint8; 映射 src = (dst - origin)/scale; src 越界写 pad_value（仿射后空间）。
// alpha/beta 每通道仿射；swap_rb=true 输出 C0=R。
// dst 输出 [3, dst_h, dst_w] FP32 CHW，已含 batch 维处理由调用方负责。
using FusedPreprocKernel = void (*)(
    const uint8_t* src, int src_w, int src_h,
    float* dst, int dst_w, int dst_h,
    float origin_x, float origin_y,
    float scale_x, float scale_y,
    const float* alpha, const float* beta,
    bool swap_rb, float pad_value);

// 选择当前 CPU 上最快的融合预处理 SIMD 内核（运行时 ISA 探测）。
// 保证非空：总有一个可用的实现（scalar/AVX2/AVX512/NEON/SVE）。
MODELDEPLOY_CXX_EXPORT FusedPreprocKernel get_fused_preproc_kernel();

// 双线性插值融合预处理内核。与 FusedPreprocKernel 同签名，但采样用双线性（4 邻加权）。
// 边界：src 坐标 clamp 到 [0, src_w-1]/[0, src_h-1]；整点越界（dst 映射的 src 完全超出）写 pad_value。
using FusedBilinearPreprocKernel = void (*)(
    const uint8_t* src, int src_w, int src_h,
    float* dst, int dst_w, int dst_h,
    float origin_x, float origin_y,
    float scale_x, float scale_y,
    const float* alpha, const float* beta,
    bool swap_rb, float pad_value);

// 选择当前 CPU 上最快的双线性融合预处理内核（运行时 ISA 探测，保证非空）。
MODELDEPLOY_CXX_EXPORT FusedBilinearPreprocKernel get_fused_bilinear_preproc_kernel();

// 颜色矩阵融合内核：采样/裁剪（origin/scale 映射）→ 3x3 颜色矩阵 + 偏置 → 写 CHW FP32。
// src: BGR 打包 uint8；采样出 (r,g,b)（swap 语义已在矩阵中体现）。
// 输出 dst_c = Σ_j mat[c][j]*ch_j + bias[c]，其中 ch_0=R, ch_1=G, ch_2=B。
// 可表达 BGR2YCrCb / BGR2RGB / 任意 3x3 线性颜色变换 + 每通道偏置。
using FusedColorMatrixKernel = void (*)(
    const uint8_t* src, int src_w, int src_h,
    float* dst, int dst_w, int dst_h,
    float origin_x, float origin_y,
    float scale_x, float scale_y,
    const float mat[3][3], const float bias[3],
    float pad_value);

// 选择当前 CPU 上最快的颜色矩阵融合内核（运行时 ISA 探测，保证非空）。
MODELDEPLOY_CXX_EXPORT FusedColorMatrixKernel get_fused_color_matrix_kernel();

// OCR det per-channel-pad kernel (resize then pad right/bottom).
// src: BGR uint8; sx = dx*src_w/resize_w, sy = dy*src_h/resize_h;
// pad zone (dy>=resize_h || dx>=resize_w) writes pad[c] per channel.
// swap_rb=true -> C0=R; pad already in affine space.
using FusedPreprocPadKernel = void (*)(
    const uint8_t* src, int src_w, int src_h,
    float* dst, int dst_w, int dst_h,
    int resize_w, int resize_h,
    const float* alpha, const float* beta,
    const float* pad);

MODELDEPLOY_CXX_EXPORT FusedPreprocPadKernel get_fusion_rpnp_kernel();

} // namespace modeldeploy::vision
