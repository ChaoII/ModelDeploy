#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <string>
#include "audio/solutions/speaker_search.h"
#include "audio/solutions/speaker_diarization.h"
#include "audio/solutions/streaming_stt.h"
#include "audio/solutions/tts_batcher.h"
using namespace modeldeploy::audio::solution;

TEST_CASE("SpeakerSearch enroll/match no weights", "[audio_solution]") {
    SpeakerSearch s;
    s.enroll("alice", {1.0f, 0.0f, 0.0f});
    s.enroll("bob",   {0.0f, 1.0f, 0.0f});
    REQUIRE(s.size() == 2);
    auto r = s.match({0.99f, 0.1f, 0.0f}, 1);
    REQUIRE_FALSE(r.empty());
    REQUIRE(r[0].first == "alice");
    REQUIRE(r[0].second > 0.9f);
}

TEST_CASE("SpeakerDiarization assign_speakers clusters", "[audio_solution]") {
    SpeakerDiarization d;
    std::vector<std::vector<float>> embs = {
        {1.0f, 0.0f}, {0.98f, 0.02f}, {0.0f, 1.0f}
    };
    auto ids = d.assign_speakers(embs, 0.5f);
    REQUIRE(ids.size() == 3);
    REQUIRE(ids[0] == ids[1]);
    REQUIRE(ids[2] != ids[0]);
}

TEST_CASE("StreamingSTT VAD triggers callback on speech", "[audio_solution]") {
    int calls = 0;
    StreamingSTT stt([&](const std::string&){ ++calls; });
    std::vector<float> audio;
    for (int i = 0; i < 16000; ++i) audio.push_back(0.5f * std::sin(2 * 3.14159265f * 440.0f * (i / 16000.0)));
    stt.push(audio, 16000);
    stt.run_once();
    REQUIRE(calls >= 1);
}

TEST_CASE("TTSBatcher enqueue/dequeue with mock synth", "[audio_solution]") {
    int synth_calls = 0;
    TTSBatcher bat([&](const std::string&){ ++synth_calls; return std::vector<float>{1.0f, 2.0f, 3.0f}; });
    bat.enqueue({"hello", "world"});
    REQUIRE(bat.pending() == 2);
    auto wavs = bat.dequeue_all();
    REQUIRE(wavs.size() == 2);
    REQUIRE(wavs[0] == std::vector<float>({1.0f, 2.0f, 3.0f}));
    REQUIRE(synth_calls == 2);
    REQUIRE(bat.pending() == 0);
}
