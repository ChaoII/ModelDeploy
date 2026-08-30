// csrc/audio/tts/common/sampling.h
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace modeldeploy::audio::tts::sampling {

// 多轮次自回归采样（Qwen3 主采样）。generated 用于 repetition penalty 抑制。
// suppress_start/suppress_end 区间内（除 suppress_exception 外）logits 置 -1e9。
inline int64_t SampleFromLogits(const float* logits, int64_t vocab_size, float temperature,
                                int32_t top_k, float top_p, float repetition_penalty,
                                const std::vector<int64_t>& generated, int64_t suppress_start,
                                int64_t suppress_end, int64_t suppress_exception,
                                bool suppress_eos, std::mt19937* rng) {
    std::vector<float> buf(logits, logits + vocab_size);
    if (suppress_start >= 0 && suppress_end > suppress_start)
        for (int64_t i = suppress_start; i < std::min(suppress_end, vocab_size); ++i)
            if (i != suppress_exception) buf[i] = -1e9f;
    if (suppress_eos && suppress_exception >= 0 && suppress_exception < vocab_size)
        buf[suppress_exception] = -1e9f;
    if (repetition_penalty > 1.0f)
        for (auto id : generated)
            if (id >= 0 && id < vocab_size)
                buf[id] = buf[id] > 0 ? buf[id] / repetition_penalty : buf[id] * repetition_penalty;
    if (temperature < 1e-6f)
        return static_cast<int64_t>(std::max_element(buf.begin(), buf.end()) - buf.begin());
    for (auto& v : buf) v /= temperature;
    if (top_k > 0 && top_k < static_cast<int32_t>(buf.size())) {
        std::vector<float> tmp(buf.begin(), buf.end());
        std::partial_sort(tmp.begin(), tmp.begin() + top_k, tmp.end(), std::greater<float>());
        const float thr = tmp[top_k - 1];
        for (auto& v : buf) if (v < thr) v = -1e9f;
    }
    const float max_v = *std::max_element(buf.begin(), buf.end());
    float sum = 0;
    for (auto& v : buf) { v = std::exp(v - max_v); sum += v; }
    for (auto& v : buf) v /= sum;
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<std::pair<float, int64_t>> pi(buf.size());
        for (size_t i = 0; i < buf.size(); ++i) pi[i] = {buf[i], static_cast<int64_t>(i)};
        std::sort(pi.begin(), pi.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        float cum = 0; int64_t cut = static_cast<int64_t>(buf.size());
        for (int64_t i = 0; i < static_cast<int64_t>(pi.size()); ++i) {
            cum += pi[i].first;
            if (cum >= top_p) { cut = i + 1; break; }
        }
        for (int64_t i = cut; i < static_cast<int64_t>(pi.size()); ++i) buf[pi[i].second] = 0.0f;
        float ns = 0; for (auto v : buf) ns += v;
        if (ns > 0) for (auto& v : buf) v /= ns;
    }
    return static_cast<int64_t>(std::discrete_distribution<int64_t>(buf.begin(), buf.end())(*rng));
}

// 一步 Gumbel 采样（Audio8 语义：argmax(probs / noise)，noise≈Exp(1) 逆变换）。
// 注意：本函数会就地改写 logits_copy，调用方传入可修改的拷贝。
inline int64_t GumbelSample(float* logits_copy, int64_t n, float temperature, float top_p,
                            int32_t top_k, std::mt19937_64& rng) {
    std::vector<float> values(logits_copy, logits_copy + n);
    std::vector<int64_t> order(n);
    for (int64_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int64_t a, int64_t b) { return values[a] > values[b]; });
    std::vector<float> sorted(n);
    for (int64_t i = 0; i < n; ++i) sorted[i] = values[order[i]];
    const float base_max = sorted[0];
    std::vector<float> base(n);
    float s = 0;
    for (int64_t i = 0; i < n; ++i) { base[i] = std::exp(sorted[i] - base_max); s += base[i]; }
    for (auto& b : base) b /= s;
    std::vector<char> remove(n, 0);
    float cum = 0;
    for (int64_t i = 0; i < n; ++i) { cum += base[i]; if (cum > top_p || i >= top_k) remove[i] = 1; }
    remove[0] = 0;
    std::vector<float> masked(logits_copy, logits_copy + n);
    for (int64_t i = 0; i < n; ++i) if (remove[i]) masked[order[i]] = -1e9f;
    const float tmp = std::max(temperature, 1e-5f);
    for (auto& v : masked) v /= tmp;
    const float mx = *std::max_element(masked.begin(), masked.end());
    for (auto& v : masked) v = std::exp(v - mx);
    float ps = 0; for (auto v : masked) ps += v;
    std::uniform_real_distribution<float> uniform(1e-12f, 1.0f);
    std::vector<float> probs(n), noise(n);
    for (int64_t i = 0; i < n; ++i) {
        probs[i] = masked[i] / ps;
        noise[i] = -std::log(std::clamp(uniform(rng), 1e-12f, 1.0f));
    }
    int64_t best = 0; float best_v = probs[0] / noise[0];
    for (int64_t i = 1; i < n; ++i) if (probs[i] / noise[i] > best_v) { best_v = probs[i] / noise[i]; best = i; }
    return best;
}
}  // namespace modeldeploy::audio::tts::sampling
