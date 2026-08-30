// csrc/audio/tts/common/kv_cache.h
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>
namespace modeldeploy::audio::tts::common {
// Audio8: 预分配 max_seq_len 的静态 KV cache（布局 [n_heads, max_seq_len, head_dim]）。
struct StaticKVCache {
    int32_t n_heads = 0;
    int32_t head_dim = 0;
    int32_t max_seq_len = 0;
    std::vector<float> key;
    std::vector<float> value;
    void Init(int32_t heads, int32_t dim, int32_t seq) {
        n_heads = heads; head_dim = dim; max_seq_len = seq;
        key.assign(static_cast<size_t>(heads) * static_cast<size_t>(seq) * static_cast<size_t>(dim), 0.0f);
        value.assign(static_cast<size_t>(heads) * static_cast<size_t>(seq) * static_cast<size_t>(dim), 0.0f);
    }
    // 覆盖写入从 start_pos 开始的片段；deltas 布局 [num_layers*2..., ]由调用方切好。
    // deltas_h = [layers, n_heads, 1, head_dim] 扁平后按层写入 cache[2*layer]/cache[2*layer+1]。
    // 简化接口：调用方直接对 key/value 操作（见 Audio8 实现），本结构只承担存储。
};
}
