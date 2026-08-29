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

// A3：非 ORT 后端必须明确失败，而非静默钳制 ORT 或产生不可诊断的加载错误。
// 后端校验发生在加载任何模型之前，因此无需测试数据、可脱机运行。
TEST_CASE("ParaformerStreaming rejects non-ORT backend with clear error", "[audio_asr]") {
    modeldeploy::RuntimeOption ro;
    ro.use_mnn_backend();                 // 故意选非 ORT
    ro.set_model_path("encoder.int8.onnx");  // 任意路径；校验先于文件加载
    ParaformerStreamingAsr asr(ro, "decoder.int8.onnx", "tokens.txt");
    // 非 ORT 后端必须在加载任何模型前明确失败。
    REQUIRE_FALSE(asr.is_initialized());
}
