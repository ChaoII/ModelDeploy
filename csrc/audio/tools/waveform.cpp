#include "audio/tools/waveform.h"
#include <algorithm>
#include <cmath>
namespace modeldeploy::audio::tool {
std::vector<float> Waveform::downsample(const std::vector<float>& in, size_t points) {
    std::vector<float> out;
    if (in.empty()) return out;
    const size_t step = (in.size() > points) ? in.size() / points : 1;
    for (size_t i = 0; i < in.size(); i += step) out.push_back(in[i]);
    return out;
}
void Spectrum::fft(std::vector<std::complex<float>>& a) const {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * 3.14159265358979f / (float)len;
        const std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t j = 0; j < len / 2; ++j) {
                std::complex<float> u = a[i + j];
                std::complex<float> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}
std::vector<float> Spectrum::magnitudes(const std::vector<float>& samples) const {
    size_t n = 1;
    while (n < (size_t)fft_n_) n <<= 1;
    std::vector<std::complex<float>> a(n, std::complex<float>(0.0f, 0.0f));
    for (size_t i = 0; i < samples.size() && i < n; ++i) a[i] = std::complex<float>(samples[i], 0.0f);
    fft(a);
    std::vector<float> mag(n / 2 + 1);
    for (size_t k = 0; k <= n / 2; ++k) mag[k] = std::abs(a[k]);
    return mag;
}
} // namespace modeldeploy::audio::tool
