#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <cstdio>
#include "audio/tools/wav_io.h"
#include "audio/tools/audio_meta.h"
using namespace modeldeploy::audio::tool;

TEST_CASE("WavIO write then read roundtrip", "[audio_tools]") {
    std::vector<float> sine(1600);
    for (size_t i = 0; i < sine.size(); ++i) sine[i] = 0.5f * std::sin(2 * 3.14159265f * 440.0f * (i / 16000.0f));
    const std::string path = "audio_roundtrip_test.wav";
    REQUIRE(write_wav(path, sine, 16000));
    WavData d;
    REQUIRE(read_wav(path, &d));
    REQUIRE(d.meta.sample_rate == 16000);
    REQUIRE(d.meta.channels == 1);
    REQUIRE(d.samples.size() == sine.size());
    REQUIRE(d.samples[100] == Catch::Approx(sine[100]).margin(1e-3f));  // 16-bit 量化误差
    auto meta = parse_meta(path);
    REQUIRE(meta.sample_rate == 16000);
    REQUIRE(meta.bits == 16);
    std::remove(path.c_str());
}
#include "audio/tools/resampler.h"

TEST_CASE("Resampler 8k->16k doubles length, frequency preserved", "[audio_tools]") {
    std::vector<float> sine(800);
    for (size_t i = 0; i < sine.size(); ++i) sine[i] = std::sin(2 * 3.14159265f * 1000.0f * (i / 8000.0f));
    auto out = Resampler::resample(sine, 8000, 16000);
    REQUIRE(out.size() == sine.size() * 2);
    const float* p = out.data();
    const size_t N = out.size();
    const double target = (double)N * 1000.0 / 16000.0;
    double best1 = 0.0, best2 = 0.0; size_t bestb = 0;
    for (size_t k = 0; k < N / 2; ++k) {
        double re = 0, im = 0;
        for (size_t i = 0; i < N; ++i) {
            const double a = 2 * 3.14159265 * k * i / N;
            re += p[i] * std::cos(a); im -= p[i] * std::sin(a);
        }
        const double e = re * re + im * im;
        if (e > best2) { best2 = best1; best1 = e; bestb = k; }
        (void)best2;
    }
    REQUIRE(std::abs((double)(int)bestb - target) <= 2);
}
#include "audio/tools/fbank.h"

TEST_CASE("Fbank produces frames x bins non-degenerate", "[audio_tools]") {
    Fbank fb(16000, 80);
    std::vector<float> s(16000);
    for (size_t i = 0; i < s.size(); ++i) s[i] = 0.5f * std::sin(2 * 3.14159265f * 440.0f * (i / 16000.0f));
    auto frames = fb.compute(s);
    REQUIRE_FALSE(frames.empty());
    REQUIRE(frames[0].size() == 80);
    float energy = 0;
    for (const auto& row : frames) for (float v : row) energy += v * v;
    REQUIRE(energy > 0.0f);
}
#include "audio/tools/waveform.h"

TEST_CASE("Spectrum single tone peak bin", "[audio_tools]") {
    const size_t N = 1024;
    std::vector<float> s(N);
    for (size_t i = 0; i < N; ++i) s[i] = std::sin(2 * 3.14159265f * 100.0f * (i / 1024.0f));
    Spectrum sp(1024);
    auto mag = sp.magnitudes(s);
    REQUIRE(mag.size() == N / 2 + 1);
    size_t best = 0; float maxv = -1;
    for (size_t k = 0; k < mag.size(); ++k) if (mag[k] > maxv) { maxv = mag[k]; best = k; }
    REQUIRE((size_t)(std::abs((long)((int)best - 100))) <= 2);
}

TEST_CASE("Waveform downsample reduces length", "[audio_tools]") {
    std::vector<float> s(4000, 0.5f);
    auto d = Waveform::downsample(s, 200);
    REQUIRE(d.size() <= 200);
}
#include "audio/tools/vad_segment.h"

TEST_CASE("VadSegment splits speech vs silence", "[audio_tools]") {
    const int sr = 16000;
    std::vector<float> sig;
    auto tone = [&](float amp, float dur_s){ for (int i = 0; i < (int)(sr*dur_s); ++i) sig.push_back(amp * std::sin(2*3.14159265f*440.0f*(i/(double)sr))); };
    auto silence = [&](float dur_s){ sig.resize(sig.size() + (size_t)(sr*dur_s), 0.0f); };
    tone(0.9f, 1.0f); silence(0.5f); tone(0.9f, 1.0f);
    VadSegment vad(sr, 0.01f, 200, 200);
    vad.feed(sig);
    auto segs = vad.segments();
    REQUIRE(segs.size() >= 2);
    REQUIRE(segs[0].samples.size() > 0);
    REQUIRE(segs[1].start_ms > segs[0].end_ms);
}
