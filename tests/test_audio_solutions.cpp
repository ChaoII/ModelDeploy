#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <filesystem>
#include <string>
#include "audio/solutions/speaker_search.h"
#include "audio/solutions/speaker_diarization.h"
#include "audio/solutions/streaming_stt.h"
#include "audio/solutions/tts_batcher.h"
#include "audio/asr/sense_voice.h"
#include "audio/tts/kokoro.h"
#include "audio/tools/wav_io.h"
#include "tests/utils.h"
using namespace modeldeploy::audio::solution;
namespace fs = std::filesystem;
using modeldeploy::audio::tool::WavData;
using modeldeploy::audio::tool::read_wav;
using modeldeploy::audio::asr::SenseVoice;
using modeldeploy::audio::tts::Kokoro;

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
    bat.enqueue(std::vector<std::string>{"hello", "world"});
    REQUIRE(bat.pending() == 2);
    auto wavs = bat.dequeue_all();
    REQUIRE(wavs.size() == 2);
    REQUIRE(wavs[0] == std::vector<float>({1.0f, 2.0f, 3.0f}));
    REQUIRE(synth_calls == 2);
    REQUIRE(bat.pending() == 0);
}

// ---- 数据与逻辑打通：解决方案层真正走 模型回调 ---- //

TEST_CASE("StreamingSTT transcribes each VAD segment via callback", "[audio_solution]") {
    std::vector<std::string> emitted;
    StreamingSTT stt([&](const std::string& t){ emitted.push_back(t); },
                     [&](const std::vector<float>&, int){ return std::string("[text]"); });
    // 两段带 300ms 间隙的语音
    std::vector<float> audio;
    auto tone = [&](float freq, int ms) {
        for (int i = 0; i < 16000 * ms / 1000; ++i)
            audio.push_back(0.5f * std::sin(2 * 3.14159265f * freq * (i / 16000.0f)));
    };
    tone(440.0f, 400);
    audio.insert(audio.end(), 16000 / 3, 0.0f);
    tone(880.0f, 400);
    stt.push(audio, 16000);
    stt.run_once();
    REQUIRE_FALSE(emitted.empty());
    for (const auto& t : emitted) REQUIRE(t == "[text]");
}

TEST_CASE("StreamingSTT real SenseVoice transcribes zh.wav", "[audio_solution]") {
    const auto data = get_test_data_path();
    const auto model = data / "test_models" / "onnx" / "sense_voice" / "model.int8.onnx";
    const auto toks  = data / "test_models" / "onnx" / "sense_voice" / "tokens.txt";
    const auto wav   = data / "test_models" / "onnx" / "sense_voice" / "test_wavs" / "zh.wav";
    if (!(fs::exists(model) && fs::exists(toks) && fs::exists(wav))) return;
    modeldeploy::RuntimeOption opt; opt.use_cpu();
    SenseVoice sv(model.string(), toks.string(), opt);
    if (!sv.is_initialized()) return;  // 权重无法加载时跳过，不误报回归
    WavData wd;
    if (!read_wav(wav.string(), &wd) || wd.samples.empty()) return;
    std::string got;
    StreamingSTT stt([&](const std::string& t){ if (!t.empty()) got = t; },
                     StreamingSTT::sense_voice(sv), wd.meta.sample_rate);
    stt.push(wd.samples, wd.meta.sample_rate);
    stt.run_once();
    REQUIRE_FALSE(got.empty());  // 真实模型应转写出非空文本
}

TEST_CASE("TTSBatcher passes each queued text to synth (text length preserved)", "[audio_solution]") {
    int calls = 0;
    TTSBatcher bat;
    const std::string a(10, 'a'), b(5, 'b');
    bat.set_synth([&](const std::string& s){ ++calls; return std::vector<float>((size_t)s.size(), 1.0f); });
    bat.enqueue(std::vector<std::string>{a, b});
    auto out = bat.dequeue_all();
    REQUIRE(out.size() == 2);
    REQUIRE(out[0].size() == a.size());
    REQUIRE(out[1].size() == b.size());
    REQUIRE(calls == 2);
    REQUIRE(bat.pending() == 0);
}

TEST_CASE("TTSBatcher split_for_synthesis respects max_chars", "[audio_solution]") {
    const std::string text = "一二三四五六七八九十。Hello world this is a sentence. 再补几个汉字组成较长一段。";
    auto parts = TTSBatcher::split_for_synthesis(text, 12);
    REQUIRE(parts.size() > 1);
    std::string joined;
    for (auto& p : parts) joined += p;
    REQUIRE(joined == text);  // 不丢字
}

TEST_CASE("TTSBatcher real Kokoro synthesizes audio", "[audio_solution]") {
    const auto data = get_test_data_path();
    const auto d = data / "test_models" / "onnx" / "kokoro_v1_1";
    const auto model = d / "model.onnx";
    const auto toks  = d / "tokens.txt";
    const auto voices = d / "voices.bin";
    const auto dict = d / "dict";
    if (!(fs::exists(model) && fs::exists(toks) && fs::exists(voices) && fs::exists(dict))) return;
    modeldeploy::RuntimeOption opt; opt.use_cpu();
    Kokoro koro(model.string(), toks.string(),
                {(d / "lexicon-us-en.txt").string(), (d / "lexicon-zh.txt").string()},
                voices.string(), dict.string(), d.string(), opt);
    if (!koro.is_initialized()) return;
    auto synth = TTSBatcher::kokoro_synth(koro, "zf_001", 1.0f);
    auto audio = synth("你好，世界。");
    REQUIRE(audio.size() > 0);
}

TEST_CASE("SpeakerDiarization run() clusters via embed callback", "[audio_solution]") {
    SpeakerDiarization d;
    int calls = 0;
    auto embed = [&](const std::vector<float>&, int){
        ++calls;
        return (calls % 2) ? std::vector<float>{0.0f, 1.0f} : std::vector<float>{1.0f, 0.0f};
    };
    std::vector<float> audio;
    auto tone = [&](float freq, int ms) {
        for (int i = 0; i < 16000 * ms / 1000; ++i)
            audio.push_back(0.5f * std::sin(2 * 3.14159265f * freq * (i / 16000.0f)));
    };
    tone(440.0f, 400); audio.insert(audio.end(), 16000 / 3, 0.0f);
    tone(880.0f, 400); audio.insert(audio.end(), 16000 / 3, 0.0f);
    tone(220.0f, 400);
    std::vector<Segment> segs;
    REQUIRE(d.run(audio, &segs, embed, 0.5f));
    REQUIRE(segs.size() >= 3);
    // 交替的两个方向应被归为两个说话人
    REQUIRE(segs[0].speaker_id == segs[2].speaker_id);
    REQUIRE(segs[0].speaker_id != segs[1].speaker_id);
}
