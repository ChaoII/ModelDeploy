//
// Created by aichao on 2025/8/2.
// CPU SIMD 融合预处理派发。
//
#pragma once

#include <cstdint>

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
FusedPreprocKernel get_fused_preproc_kernel();

} // namespace modeldeploy::vision
