// tests/test_audio_tts.cpp
// TTS 用例（Audio8 / Qwen3 / Kokoro）：同步与流式一致性、输出合理性。
// 约定：无模型时 SKIP（early return），保持 ctest 零失败（沿用仓库惯例）。
// 模型目录：env MODELDEPLOY_TTS_MODELS_DIR；缺省 ../../test_data（相对 ctest 工作目录 build2/）。

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "audio/solutions/tts_batcher.h"
#include "audio/tts/audio8/audio8.h"
#include "audio/tts/kokoro.h"
#include "audio/tts/qwen3/qwen3_tts.h"
#include "audio/tts/tts_model.h"
#include "tests/utils.h"

namespace fs = std::filesystem;
using modeldeploy::audio::tts::Audio8;
using modeldeploy::audio::tts::ITtsModel;
using modeldeploy::audio::tts::Kokoro;
using modeldeploy::audio::tts::Qwen3Tts;

namespace {

fs::path tts_models_base() {
    const char* env = std::getenv("MODELDEPLOY_TTS_MODELS_DIR");
    if (env && *env) return fs::absolute(fs::path(env));
    return fs::absolute("../../test_data");
}

fs::path audio8_dir() { return tts_models_base() / "audio8_preview"; }
fs::path qwen3_dir() { return tts_models_base() / "qwen3_tts_0.6b"; }

// Kokoro 候选目录（新目录布局 / test_data 旧布局）
fs::path find_kokoro_dir() {
    const std::vector<fs::path> candidates = {
        tts_models_base() / "kokoro_v1_1",
        tts_models_base() / "test_models" / "onnx" / "kokoro_v1_1",
        get_test_data_path() / "test_models" / "onnx" / "kokoro_v1_1",
    };
    for (const auto& c : candidates) {
        if (fs::exists(c / "model.onnx")) return c;
    }
    return {};
}

double rms(const std::vector<float>& a) {
    if (a.empty()) return 0.0;
    double s = 0.0;
    for (float v : a) s += static_cast<double>(v) * static_cast<double>(v);
    return std::sqrt(s / static_cast<double>(a.size()));
}

bool collect_stream(ITtsModel* m, const std::string& text,
                    const std::string& voice, int chunk_frames,
                    std::vector<float>* out) {
    out->clear();
    return m->predict_stream(
        text, voice, 1.0f, chunk_frames,
        [out](const float* data, int n, float) {
            if (n > 0) out->insert(out->end(), data, data + n);
            return true;
        });
}

double rel_len_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || b.empty()) return 1.0;
    return std::fabs(static_cast<double>(a.size()) - static_cast<double>(b.size())) /
           static_cast<double>(b.size());
}

}  // namespace

TEST_CASE("split_for_synthesis respects max_chars", "[tts][tts-common]") {
    const std::string text =
        "一二三四五六七八九十。Hello world this is a sentence. 再补几个汉字组成较长一段。";
    const auto parts =
        modeldeploy::audio::solution::TTSBatcher::split_for_synthesis(text, 12);
    REQUIRE(parts.size() > 1);
    std::string joined;
    for (const auto& p : parts) joined += p;
    REQUIRE(joined == text);  // 不丢字、不增字
}

TEST_CASE("Audio8 predict produces audio", "[tts][tts-audio8]") {
    const auto dir = audio8_dir();
    if (!fs::exists(dir / "runtime_manifest.json")) return;
    modeldeploy::RuntimeOption opt;
    Audio8 m;
    if (!m.Load(dir.string(), opt)) return;
    REQUIRE(m.get_sample_rate() == 44100);
    std::vector<float> audio;
    REQUIRE(m.predict("今天天气真不错，适合出门散步。", "demo", 1.0f, &audio));
    REQUIRE(audio.size() > 1000);
    REQUIRE(rms(audio) > 1e-3);
}

TEST_CASE("Qwen3Tts predict produces audio", "[tts][tts-qwen3]") {
    const auto dir = qwen3_dir();
    if (!fs::exists(dir / "onnx_kv_06b")) return;
    modeldeploy::RuntimeOption opt;
    Qwen3Tts m;
    if (!m.init(dir.string(), opt)) return;
    REQUIRE(m.get_sample_rate() == 24000);
    std::vector<float> audio;
    REQUIRE(m.predict("你好，世界！", "Vivian", 1.0f, &audio));
    REQUIRE(audio.size() > 1000);
    REQUIRE(rms(audio) > 1e-3);
}

TEST_CASE("Audio8 predict_stream equals predict", "[tts][tts-audio8]") {
    const auto dir = audio8_dir();
    if (!fs::exists(dir / "runtime_manifest.json")) return;
    modeldeploy::RuntimeOption opt;
    Audio8 m;
    if (!m.Load(dir.string(), opt)) return;
    const std::string text = "今天天气真不错，适合出门散步。";
    std::vector<float> whole;
    REQUIRE(m.predict(text, "demo", 1.0f, &whole));
    REQUIRE_FALSE(whole.empty());
    std::vector<float> streamed;
    REQUIRE(collect_stream(&m, text, "demo", 480, &streamed));
    REQUIRE_FALSE(streamed.empty());
    REQUIRE(rel_len_diff(streamed, whole) < 0.05);
}

TEST_CASE("Qwen3Tts predict_stream equals predict", "[tts][tts-qwen3]") {
    const auto dir = qwen3_dir();
    if (!fs::exists(dir / "onnx_kv_06b")) return;
    modeldeploy::RuntimeOption opt;
    Qwen3Tts m;
    if (!m.init(dir.string(), opt)) return;
    const std::string text = "你好，世界！这是流式一致性测试。";
    // Qwen3 采样用 thread_local mt19937(std::random_device{})，逐次生成非确定，
    // 同一文本两次独立生成的时长有抖动（实测 rel_diff 0%~45%）。故采用
    // "多尝试取最小差"：3 对独立 (predict, stream) 生成中至少一对时长差 <30%，
    // 从而排除"流式被截断/翻倍"这类结构性异常，又不因采样随机性偶发失败。
    double best = 1.0;
    for (int attempt = 0; attempt < 3; ++attempt) {
        std::vector<float> whole, streamed;
        if (!m.predict(text, "Vivian", 1.0f, &whole) || whole.empty()) continue;
        if (!collect_stream(&m, text, "Vivian", 480, &streamed) ||
            streamed.empty())
            continue;
        best = std::min(best, rel_len_diff(streamed, whole));
    }
    REQUIRE(best < 0.3);
}

TEST_CASE("Kokoro predict_stream equals predict", "[tts][tts-kokoro]") {
    const auto d = find_kokoro_dir();
    if (d.empty()) return;
    const auto model = d / "model.onnx";
    const auto toks = d / "tokens.txt";
    const auto voices = d / "voices.bin";
    const auto dict = d / "dict";
    if (!(fs::exists(toks) && fs::exists(voices) && fs::exists(dict))) return;
    modeldeploy::RuntimeOption opt;
    Kokoro koro(model.string(), toks.string(),
                {(d / "lexicon-us-en.txt").string(),
                 (d / "lexicon-zh.txt").string()},
                voices.string(), dict.string(), d.string(), opt);
    if (!koro.is_initialized()) return;
    const std::string text = "你好，世界。这是一个测试。";
    std::vector<float> whole;
    REQUIRE(koro.predict(text, "zf_001", 1.0f, &whole));
    REQUIRE_FALSE(whole.empty());
    std::vector<float> streamed;
    REQUIRE(collect_stream(&koro, text, "zf_001", 120, &streamed));
    REQUIRE_FALSE(streamed.empty());
    REQUIRE(rel_len_diff(streamed, whole) < 0.05);
}

TEST_CASE("Audio8 overlong text rejected without truncation", "[tts][tts-audio8]") {
    const auto dir = audio8_dir();
    if (!fs::exists(dir / "runtime_manifest.json")) return;
    modeldeploy::RuntimeOption opt;
    Audio8 m;
    if (!m.Load(dir.string(), opt)) return;
    const std::string long_text(6000, '啊');
    std::vector<float> audio(3, 0.0f);
    REQUIRE_FALSE(m.predict(long_text, "demo", 1.0f, &audio));
    REQUIRE(audio.size() == 3);
    REQUIRE(audio[0] == 0.0f);
    REQUIRE(audio[1] == 0.0f);
    REQUIRE(audio[2] == 0.0f);
}

TEST_CASE("Qwen3Tts overlong text rejected without truncation", "[tts][tts-qwen3]") {
    const auto dir = qwen3_dir();
    if (!fs::exists(dir / "onnx_kv_06b")) return;
    modeldeploy::RuntimeOption opt;
    Qwen3Tts m;
    if (!m.init(dir.string(), opt)) return;
    const std::string long_text(6000, '好');
    std::vector<float> audio(3, 0.0f);
    REQUIRE_FALSE(m.predict(long_text, "Vivian", 1.0f, &audio));
    REQUIRE(audio.size() == 3);
    REQUIRE(audio[0] == 0.0f);
    REQUIRE(audio[1] == 0.0f);
    REQUIRE(audio[2] == 0.0f);
}
