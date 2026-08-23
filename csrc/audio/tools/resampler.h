#pragma once
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::audio::tool {
class MODELDEPLOY_CXX_EXPORT Resampler {
public:
    static std::vector<float> resample(const std::vector<float>& in, int in_sr, int out_sr);
};
} // namespace modeldeploy::audio::tool
