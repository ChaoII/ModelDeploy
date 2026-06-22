#pragma once
#include "core/md_decl.h"
#include <cstdint>

namespace modeldeploy::vision {
    MODELDEPLOY_CXX_EXPORT bool nv12_to_bgr_cpu(const uint8_t* src_y,
                          const uint8_t* src_uv,
                          int src_w, int src_h,
                          int step_y, int step_uv,
                          uint8_t* dst_bgr);
}