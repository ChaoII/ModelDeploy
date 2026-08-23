#include "audio/tools/wav_io.h"
#include "utils/wave_helper.h"
#include <cstdio>

namespace modeldeploy::audio::tool {

bool read_wav(const std::string& path, WavData* out) {
    if (!out) return false;
    int sr = 0;
    if (!load_wav_file(path.c_str(), &sr, out->samples)) return false;
    out->meta.sample_rate = sr;
    out->meta.channels = 1;
    out->meta.bits = 32;
    out->meta.format = 3;
    if (sr > 0) out->meta.duration_ms = (uint32_t)(out->samples.size() * 1000u / (uint32_t)sr);
    return true;
}

bool write_wav(const std::string& path, const std::vector<float>& samples, int sample_rate) {
    if (samples.empty() || sample_rate <= 0) return false;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t n = (uint32_t)samples.size();
    const uint32_t data_bytes = n * 2;
    const uint32_t byte_rate = (uint32_t)sample_rate * 2;
    auto w16 = [&](uint16_t v){ std::fwrite(&v, 2, 1, f); };
    auto w32 = [&](uint32_t v){ std::fwrite(&v, 4, 1, f); };
    std::fwrite("RIFF", 1, 4, f); w32(36 + data_bytes);
    std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(1); w32((uint32_t)sample_rate);
    w32(byte_rate); w16(2); w16(16);
    std::fwrite("data", 1, 4, f); w32(data_bytes);
    for (uint32_t i = 0; i < n; ++i) {
        int16_t v = (int16_t)(samples[i] * 32767.0f);
        std::fwrite(&v, 2, 1, f);
    }
    std::fclose(f);
    return true;
}
} // namespace modeldeploy::audio::tool
