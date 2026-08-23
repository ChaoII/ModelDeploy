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
