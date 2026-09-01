// csrc/audio/tts/audio8/audio8_runtime.h
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "audio/tts/audio8/audio8_manifest.h"
#include "core/enum_variables.h"

namespace modeldeploy::audio::tts::audio8 {

// Audio8 三个 ONNX session 的直接 ORT 封装（slow_ar / fast_ar / codec_decoder）。
// KV cache 为 fp16（uint16_t 位模式，与模型输入 tensor(float16) 对齐），
// 因此不复用 float32 的 StaticKVCache（见 audio8.cpp 偏离说明）。
// 会话本身不保留推理状态：每次调用由调用方传入完整 KV buffer 并写回 delta，
// 因此同一实例可被 clone() 浅共享，但并发调用需由调用方串行化。
class MODELDEPLOY_CXX_EXPORT Audio8Runtime {
public:
    Audio8Runtime();
    ~Audio8Runtime();
    Audio8Runtime(const Audio8Runtime&) = delete;
    Audio8Runtime& operator=(const Audio8Runtime&) = delete;

    bool Load(const Audio8Manifest& manifest, int32_t threads, Device device = Device::CPU,
              int32_t device_id = 0);

    [[nodiscard]] bool valid() const { return loaded_; }
    [[nodiscard]] int64_t num_codebooks() const { return num_codebooks_; }
    [[nodiscard]] int64_t slow_logits_size() const { return slow_logits_size_; }
    [[nodiscard]] int64_t fast_dim() const { return fast_dim_; }

    // slow 一步（prefill 与迭代统一）：codes 为 [1, num_codebooks+1, seq] 扁平 int64，
    // positions 为 [seq]；cache 为 2*num_layers 段 [1, n_local_heads, max_seq_len, head_dim]
    // 的 fp16 位模式扁平数组（调用方按 Load 时尺寸预分配）。
    // 输出最后一行 logits（slow_logits_size 个 float）与最后一行 hidden（fast_dim 个 fp16 位模式）。
    bool SlowStep(const std::vector<int64_t>& codes, const std::vector<int64_t>& positions,
                  std::vector<uint16_t>* cache, std::vector<float>* last_logits,
                  std::vector<uint16_t>* last_hidden);

    // fast 一步：token 为 codebook token；use_hidden=true 时模型忽略 token 改用 slow_hidden。
    // fast_cache 为 2*num_fast_layers 段 [1, fast_n_local_heads, num_codebooks, fast_head_dim] fp16 扁平。
    bool FastStep(int64_t token, bool use_hidden, int64_t position,
                  const std::vector<uint16_t>& slow_hidden, std::vector<uint16_t>* fast_cache,
                  std::vector<float>* last_logits);

    // codec 解码：codes 为 [num_codebooks, frames] 扁平 int64；输出 audio 为 44.1kHz float。
    bool DecodeCodes(const std::vector<int64_t>& codes, int64_t frames,
                     std::vector<float>* audio);

    // 预分配输出缓冲，避免 predict 循环内反复扩容（可选优化）。
    void ReserveSpeech(size_t num_frames);

    // ---- GPU 低延迟路径(WITH_GPU)----
    // KV cache 常驻 GPU:每层 K/V 设备缓冲 + 每 session 的 IoBinding。
    // 状态不持有在 session 上,每次生成调用创建一个(与 CPU cache vector 同样由调用方私有),
    // 故 clone() 共享 session 时各调用并发安全。
    struct Audio8GpuState;  // 不透明,定义于 audio8_runtime.cpp(WITH_GPU)
    // unique_ptr 需要"不完整类型可删除":用带外部定义 deleter 的 PIMPL 惯用法。
    // (默认 default_delete 会在测试/audio8.cpp 等不含完整定义的 TU 实例化析构时编译失败;
    //  deleter 的 operator() 在 .cpp 内实现,那里的 Audio8GpuState 是完整类型。)
    struct MODELDEPLOY_CXX_EXPORT Audio8GpuStateDeleter {
        void operator()(Audio8GpuState* p) const noexcept;  // 定义于 audio8_runtime.cpp
    };
    using Audio8GpuStatePtr = std::unique_ptr<Audio8GpuState, Audio8GpuStateDeleter>;
    // device/device_id 与 Load 一致;GPU/CUDA 不可用或申请失败返回 nullptr(调用方走 CPU)。
    Audio8GpuStatePtr MakeGpuState(Device device, int32_t device_id);
    // slow 一步的 GPU 变体:与 SlowStep 同语义,但 cache 常驻 GPU、delta 输出绑 GPU 并原地写回。
    bool SlowStepGpu(Audio8GpuState* g, const std::vector<int64_t>& codes,
                     const std::vector<int64_t>& positions,
                     std::vector<float>* last_logits,
                     std::vector<uint16_t>* last_hidden);
    // fast 一步的 GPU 变体:同 CPU FastStep 语义。
    bool FastStepGpu(Audio8GpuState* g, int64_t token, bool use_hidden, int64_t position,
                     const std::vector<uint16_t>& slow_hidden,
                     std::vector<float>* last_logits);
    // 把 GPU cache 回拷 host(仅测试/诊断用):kind=0 slow,1 fast。
    bool CopyGpuCacheToHost(Audio8GpuState* g, int kind, uint16_t* dst);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int64_t num_codebooks_ = 0;
    int64_t slow_logits_size_ = 0;
    int64_t fast_dim_ = 0;
    bool loaded_ = false;
};

}  // namespace modeldeploy::audio::tts::audio8
