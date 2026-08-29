#include "vision/processors/cuda/cjk_font_device.h"
#include "vision/processors/cuda/cjk_font.h"
#include <cuda_runtime.h>

namespace modeldeploy::vision {
    const uint8_t* cjk_device_glyphs() {
        static uint8_t* d = nullptr;
        if (!d) {
            const size_t bytes_per_glyph =
                static_cast<size_t>(kCjkFontGlyphStride) * kCjkFontGlyphRows;
            const size_t n = kCjkFontIndexSize * bytes_per_glyph;
            uint8_t* buf = nullptr;
            if (cudaMalloc(&buf, n) != cudaSuccess) return nullptr;
            if (cudaMemcpy(buf, kCjkFontGlyphBitmaps, n, cudaMemcpyHostToDevice) != cudaSuccess) {
                cudaFree(buf);
                return nullptr;
            }
            d = buf;
        }
        return d;
    }
}
