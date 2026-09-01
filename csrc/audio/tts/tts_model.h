// csrc/audio/tts/tts_model.h
#pragma once

#include <functional>
#include <string>
#include <vector>
#include "base_model.h"

namespace modeldeploy::audio::tts {
// 统一 TTS 抽象。chunk_frames == 0 表示一次性合成；
// >0 时逐块回调，cb 返回 false 可提前中止。
// chunk_frames 单位为模型相关（各模型不同，非"秒"）：
//   - Kokoro ：UTF-8 字符数（近似每段合成块的目标大小）。
// 上层应传相对小的值以观察到多次音频回调（如 120）。
class MODELDEPLOY_CXX_EXPORT ITtsModel : public BaseModel {
public:
    ~ITtsModel() override = default;
    virtual bool predict(const std::string& text, const std::string& voice, float speed,
                         std::vector<float>* out) = 0;
    virtual bool predict_stream(const std::string& text, const std::string& voice, float speed,
                                int chunk_frames,
                                const std::function<bool(const float*, int, float)>& cb) = 0;
    [[nodiscard]] virtual int32_t get_sample_rate() const = 0;
};
}  // namespace modeldeploy::audio::tts
