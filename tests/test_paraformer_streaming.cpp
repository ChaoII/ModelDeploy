#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>
#include "audio/asr/paraformer_streaming.h"
#include "audio/tools/wav_io.h"
#include "tests/utils.h"
using namespace modeldeploy::audio::asr;
using namespace modeldeploy::audio::tool;
namespace fs = std::filesystem;

TEST_CASE("ParaformerStreaming resets and reports empty before feed", "[audio_asr]") {
    const auto data = get_test_data_path();
    const auto dir = data / "test_models" / "onnx" / "paraformer_streaming";
    const auto enc = dir / "encoder.int8.onnx";
    const auto dec = dir / "decoder.int8.onnx";
    const auto toks = dir / "tokens.txt";
    if (!(fs::exists(enc) && fs::exists(dec) && fs::exists(toks))) return;

    ParaformerStreamingAsr asr(enc.string(), dec.string(), toks.string());
    REQUIRE(asr.is_initialized());
    REQUIRE(asr.vocab_size() > 0);

    StreamingAsrResult r;
    bool ok = asr.decode(false, &r);
    (void)ok;  // 无样本时不应产出
    REQUIRE(r.tokens.empty());
    asr.reset();  // 复位不应崩
}

TEST_CASE("ParaformerStreaming transcribes real wav via streaming", "[audio_asr]") {
    const auto data = get_test_data_path();
    const auto dir = data / "test_models" / "onnx" / "paraformer_streaming";
    const auto enc = dir / "encoder.int8.onnx";
    const auto dec = dir / "decoder.int8.onnx";
    const auto toks = dir / "tokens.txt";
    const auto wav = data / "test_models" / "onnx" / "sense_voice" / "test_wavs" / "zh.wav";
    if (!(fs::exists(enc) && fs::exists(dec) && fs::exists(toks) && fs::exists(wav))) return;

    ParaformerStreamingAsr asr(enc.string(), dec.string(), toks.string());
    if (!asr.is_initialized()) return;

    WavData wd;
    if (!read_wav(wav.string(), &wd) || wd.samples.empty()) return;

    // int16 量纲（[-32768,32767]），流式喂入 100ms 块
    int sr = wd.meta.sample_rate > 0 ? wd.meta.sample_rate : 16000;
    const size_t chunk = (size_t)sr / 10;  // 100ms
    std::string partials;
    size_t i = 0;
    for (; i + chunk <= wd.samples.size(); i += chunk) {
        std::vector<float> block;
        block.reserve(chunk);
        for (size_t k = 0; k < chunk; ++k) block.push_back(wd.samples[i + k] * 32768.0f);
        asr.accept_waveform(block);
        StreamingAsrResult r;
        if (asr.decode(false, &r) && !r.text.empty()) {
            partials = asr.text();
        }
    }
    // 剩余尾部
    if (i < wd.samples.size()) {
        std::vector<float> tail;
        for (; i < wd.samples.size(); ++i) tail.push_back(wd.samples[i] * 32768.0f);
        asr.accept_waveform(tail);
    }
    asr.input_finished();
    StreamingAsrResult final;
    asr.decode(true, &final);

    std::string full = asr.text();
    REQUIRE_FALSE(full.empty());          // 真实模型应转写出非空文本
    REQUIRE(final.confidence >= 0.0f);
    REQUIRE(final.confidence <= 1.0f);
    REQUIRE(final.is_final);              // flush 后应标记 final
    REQUIRE(partials.size() <= full.size());  // 部分结果不会比全文长
}
