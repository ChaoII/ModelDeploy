#pragma once
#include <complex>
#include <cstddef>
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::audio::tool {
class MODELDEPLOY_CXX_EXPORT Waveform {
public:
    static std::vector<float> downsample(const std::vector<float>& in, size_t points);
};
class MODELDEPLOY_CXX_EXPORT Spectrum {
public:
    explicit Spectrum(int fft_n = 1024) : fft_n_(fft_n) {}
    std::vector<float> magnitudes(const std::vector<float>& samples) const;
private:
    int fft_n_;
    void fft(std::vector<std::complex<float>>& a) const;
};
} // namespace modeldeploy::audio::tool
