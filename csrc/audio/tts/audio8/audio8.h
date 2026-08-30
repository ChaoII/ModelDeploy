// csrc/audio/tts/audio8/audio8.h
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "audio/tts/audio8/audio8_manifest.h"
#include "audio/tts/audio8/audio8_runtime.h"
#include "audio/tts/tts_model.h"
#include "runtime/runtime_option.h"

namespace modeldeploy::audio::tts {

// Audio8-TTS-Preview-0.6B（DualAR + 内置 codec，44.1kHz）。
// 官方实现参照 _src_audio8/onnx_runtime/arktts_runtime/*.py。
// - 声音来源：{model_dir}/voices/ 下的注册 voice（codes.npy + meta.json）。
// - speed 官方无此参数：预留、不影响合成。
class MODELDEPLOY_CXX_EXPORT Audio8 : public ITtsModel {
public:
    Audio8();
    ~Audio8() override;
    Audio8(const Audio8&) = delete;
    Audio8& operator=(const Audio8&) = delete;

    // model_dir 指向 audio8_preview 根目录；opt 仅用 cpu_thread_num。
    bool Load(const std::string& model_dir, const RuntimeOption& opt);

    bool predict(const std::string& text, const std::string& voice, float speed,
                 std::vector<float>* out) override;

    // chunk_frames == 0 等价一次性合成（单次回调整段音频）；
    // > 0 时按官方 stream() 滑动窗口 + guard 逐块回调，cb 返回 false 立即中止。
    bool predict_stream(const std::string& text, const std::string& voice, float speed,
                        int chunk_frames,
                        const std::function<bool(const float*, int, float)>& cb) override;

    [[nodiscard]] int32_t get_sample_rate() const override;

    // 浅克隆：共享底层 ORT session（不重新加载模型）。
    std::unique_ptr<Audio8> clone() const;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace modeldeploy::audio::tts
