#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::audio::tool {
struct MODELDEPLOY_CXX_EXPORT WavMeta {
    int channels{1}; int sample_rate{0}; int bits{16}; int format{1}; uint32_t duration_ms{0};
};
struct MODELDEPLOY_CXX_EXPORT WavData { WavMeta meta; std::vector<float> samples; };
MODELDEPLOY_CXX_EXPORT bool read_wav(const std::string& path, WavData* out);
MODELDEPLOY_CXX_EXPORT bool write_wav(const std::string& path, const std::vector<float>& samples, int sample_rate);
} // namespace modeldeploy::audio::tool
