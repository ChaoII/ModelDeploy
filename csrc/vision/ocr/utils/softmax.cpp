//
// Created by aichao on 2025/2/21.
//

#include "vision/ocr/utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    static float fast_exp(const float x) {
        union {
            uint32_t i;
            float f;
        } v{};
        v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
        return v.f;
    }

    std::vector<float> softmax(std::vector<float>& src) {
        const int length = static_cast<int>(src.size());
        std::vector<float> dst;
        dst.resize(length);
        const float alpha = *std::max_element(&src[0], &src[0 + length]);
        float denominator{0};
        for (int i = 0; i < length; ++i) {
            dst[i] = fast_exp(src[i] - alpha);
            denominator += dst[i];
        }
        for (int i = 0; i < length; ++i) {
            dst[i] /= denominator;
        }
        return dst;
    }
}
