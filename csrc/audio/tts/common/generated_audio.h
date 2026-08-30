// csrc/audio/tts/common/generated_audio.h
#pragma once
#include <vector>
namespace modeldeploy::audio::tts {
struct GeneratedAudio {
    int32_t sample_rate = 0;
    std::vector<float> samples;
};
}  // namespace modeldeploy::audio::tts
