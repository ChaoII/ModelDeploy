// csrc/audio/tts/common/codec_streaming.h
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>
namespace modeldeploy::audio::tts::common {
// 把 held_tail（上一块尾部，跨帧叠加窗口）与 samples 开头叠加融合（线性 crossfade）。
template <typename T>
void CrossfadeBlend(std::vector<T>& samples, const std::vector<T>& held_tail, int32_t cf) {
    const int32_t n = std::min(cf, std::min((int32_t)held_tail.size(), (int32_t)samples.size()));
    for (int32_t i = 0; i < n; ++i) {
        const T t = static_cast<T>(i + 1) / (T)(n + 1);
        samples[i] = held_tail[i] * (T(1) - t) + samples[i] * t;
    }
}
}
