#pragma once
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::audio::tool {
class MODELDEPLOY_CXX_EXPORT Fbank {
public:
    explicit Fbank(int sample_rate = 16000, int num_bins = 80);
    std::vector<std::vector<float>> compute(const std::vector<float>& samples) const;
private:
    int sr_;
    int bins_;
};
} // namespace modeldeploy::audio::tool
