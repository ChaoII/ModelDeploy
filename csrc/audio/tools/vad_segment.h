#pragma once
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::audio::tool {
struct MODELDEPLOY_CXX_EXPORT Seg { int start_ms; int end_ms; std::vector<float> samples; };
class MODELDEPLOY_CXX_EXPORT VadSegment {
public:
    VadSegment(int sample_rate = 16000, float energy_threshold = 0.01f,
               int min_speech_ms = 200, int min_silence_ms = 200)
        : sr_(sample_rate), thr_(energy_threshold), min_speech_(min_speech_ms), min_silence_(min_silence_ms) {}
    void reset() { buf_.clear(); }
    void feed(const std::vector<float>& samples) { buf_.insert(buf_.end(), samples.begin(), samples.end()); }
    std::vector<Seg> segments() const;
private:
    int sr_;
    float thr_;
    int min_speech_, min_silence_;
    std::vector<float> buf_;
};
} // namespace modeldeploy::audio::tool
