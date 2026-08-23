#include "audio/tools/audio_meta.h"
#include <cstdio>
#include <cstring>
namespace modeldeploy::audio::tool {
WavMeta parse_meta(const std::string& path) {
    WavMeta meta;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return meta;
    char riff[4] = {0}; std::fread(riff, 1, 4, f);
    if (std::memcmp(riff, "RIFF", 4) != 0) { std::fclose(f); return meta; }
    uint32_t dat; std::fread(&dat, 4, 1, f);
    std::fseek(f, 12, SEEK_SET); // skip RIFF+size+WAVE
    char id[4];
    uint32_t size = 0;
    bool have_fmt = false;
    while (std::fread(id, 1, 4, f) == 4 && std::fread(&size, 4, 1, f) == 1) {
        if (std::memcmp(id, "fmt ", 4) == 0) {
            uint16_t fmt, ch; uint32_t sr, br; uint16_t ba, bits;
            std::fread(&fmt, 2, 1, f); std::fread(&ch, 2, 1, f);
            std::fread(&sr, 4, 1, f); std::fread(&br, 4, 1, f);
            std::fread(&ba, 2, 1, f); std::fread(&bits, 2, 1, f);
            meta.format = fmt; meta.channels = ch; meta.sample_rate = (int)sr; meta.bits = bits;
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0 && have_fmt) {
            const uint32_t bytes = size;
            if (meta.sample_rate > 0 && bytes > 0)
                meta.duration_ms = (uint32_t)(bytes * 1000ull / ((uint64_t)meta.sample_rate * (uint64_t)meta.channels * ((uint64_t)meta.bits / 8)));
            break;
        }
        std::fseek(f, (long)size + (size & 1), SEEK_CUR);
    }
    std::fclose(f);
    return meta;
}
} // namespace modeldeploy::audio::tool
