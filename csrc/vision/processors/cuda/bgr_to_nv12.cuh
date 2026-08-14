#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "core/md_decl.h"

namespace modeldeploy::vision {
    /// GPU BGR（BGR24 连续内存）→ NV12（BT.709 limited range）
    /// bgr 可为 device 或 host 指针（host 自动上传）；nv12 输出 device buffer（host 指针会自动回拷）
    /// @return false 表示 CUDA 调用失败
    MODELDEPLOY_CXX_EXPORT bool bgr_to_nv12_cuda(
        const uint8_t* bgr, int width, int height,
        uint8_t* nv12, cudaStream_t stream = nullptr);
}
