#pragma once
#include <cuda_runtime.h>
#include "core/md_decl.h"

namespace modeldeploy::vision {
    MODELDEPLOY_CXX_EXPORT bool nv12_to_bgr_cuda(const uint8_t* src_y,
                           const uint8_t* src_uv,
                           int src_w, int src_h,
                           int step_y, int step_uv,
                           uint8_t* dst_bgr,
                           cudaStream_t stream = nullptr);
}