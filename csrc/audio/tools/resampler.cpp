#include "audio/tools/resampler.h"
#include <samplerate/include/samplerate.h>
#include <cmath>
namespace modeldeploy::audio::tool {
std::vector<float> Resampler::resample(const std::vector<float>& in, int in_sr, int out_sr) {
    if (in.empty() || in_sr <= 0 || out_sr <= 0) return {};
    if (in_sr == out_sr) return in;
    const long n_out = (long)std::llround((double)in.size() * out_sr / in_sr);
    std::vector<float> out(n_out, 0.0f);
    SRC_DATA data{};
    data.data_in = const_cast<float*>(in.data());
    data.input_frames = (long)in.size();
    data.data_out = out.data();
    data.output_frames = n_out;
    data.src_ratio = (double)out_sr / in_sr;
    if (src_simple(&data, SRC_SINC_BEST_QUALITY, 1) != 0) return {};
    out.resize(data.output_frames_gen);
    return out;
}
} // namespace modeldeploy::audio::tool
