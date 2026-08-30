// csrc/audio/tts/tts_model.h
#pragma once

#include <functional>
#include <string>
#include <vector>
#include "base_model.h"

namespace modeldeploy::audio::tts {
// 三模型统一 TTS 抽象。chunk_frames == 0 表示一次性合成；
// >0 时逐块回调，cb 返回 false 可提前中止。
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
