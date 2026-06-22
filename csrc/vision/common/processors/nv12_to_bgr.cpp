#include "vision/common/processors/nv12_to_bgr.h"
#include <cstdint>
#include <algorithm>

namespace modeldeploy::vision {
    bool nv12_to_bgr_cpu(const uint8_t* src_y,
                          const uint8_t* src_uv,
                          int src_w, int src_h,
                          int step_y, int step_uv,
                          uint8_t* dst_bgr) {
        const int uv_h = src_h / 2;
        for (int y = 0; y < src_h; ++y) {
            const uint8_t* y_row = src_y + y * step_y;
            uint8_t* dst_row = dst_bgr + y * src_w * 3;
            const int uv_y = y >> 1;
            const uint8_t* uv_row = src_uv + uv_y * step_uv;
            for (int x = 0; x < src_w; ++x) {
                float y_val = static_cast<float>(y_row[x]);
                const int uv_x = x >> 1;
                const int safe_uv_x = (uv_x < (src_w >> 1)) ? uv_x : (src_w >> 1) - 1;
                float u_val = static_cast<float>(uv_row[safe_uv_x * 2 + 0]) - 128.0f;
                float v_val = static_cast<float>(uv_row[safe_uv_x * 2 + 1]) - 128.0f;
                float r = y_val + 1.402f * v_val;
                float g = y_val - 0.344136f * u_val - 0.714136f * v_val;
                float b = y_val + 1.772f * u_val;
                r = std::min(std::max(r, 0.0f), 255.0f);
                g = std::min(std::max(g, 0.0f), 255.0f);
                b = std::min(std::max(b, 0.0f), 255.0f);
                dst_row[x * 3 + 0] = static_cast<uint8_t>(b);
                dst_row[x * 3 + 1] = static_cast<uint8_t>(g);
                dst_row[x * 3 + 2] = static_cast<uint8_t>(r);
            }
        }
        return true;
    }
}