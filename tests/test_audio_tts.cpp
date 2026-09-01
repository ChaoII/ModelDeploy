// tests/test_audio_tts.cpp
// TTS 用例（Kokoro）：同步与流式一致性、输出合理性。
// 约定：无模型时 SKIP（early return），保持 ctest 零失败（沿用仓库惯例）。
// 模型目录：env MODELDEPLOY_TTS_MODELS_DIR；缺省 ../../test_data（相对 ctest 工作目录 build2/）。

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "audio/solutions/tts_batcher.h"
#include "audio/tts/kokoro.h"
#include "audio/tts/tts_model.h"
#include "tests/test_gpu_utils.h"
#include "tests/utils.h"

namespace fs = std::filesystem;
using modeldeploy::audio::tts::ITtsModel;
using modeldeploy::audio::tts::Kokoro;

namespace {

fs::path tts_models_base() {
    const char* env = std::getenv("MODELDEPLOY_TTS_MODELS_DIR");
    if (env && *env) return fs::absolute(fs::path(env));
    return fs::absolute("../../test_data");
}

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

// 分块统计：audio_callbacks 仅计 n>0 的音频回调。
struct StreamStats {
    bool ok = false;
    size_t audio_callbacks = 0;
    std::vector<float> audio;
    bool all_finite = true;
};

StreamStats collect_chunked(ITtsModel* m, const std::string& text,
                            const std::string& voice, int chunk_frames) {
    StreamStats st;
    st.ok = m->predict_stream(
        text, voice, 1.0f, chunk_frames,
        [&st](const float* data, int n, float) {
            if (n > 0) {
                ++st.audio_callbacks;
                st.audio.insert(st.audio.end(), data, data + n);
                for (int i = 0; i < n; ++i) {
                    if (!std::isfinite(data[i])) {
                        st.all_finite = false;
                        break;
                    }
                }
            }
            return true;
        });
    return st;
}

// 按 min_chars 个 UTF-8 字符生成重复长文本（中文字符 3 字节）。
std::string long_sentence(int min_chars) {
    auto n_chars = [](const std::string& s) {
        size_t n = 0;
        for (size_t i = 0; i < s.size(); ++i) {
            const unsigned char c = static_cast<unsigned char>(s[i]);
            if ((c & 0xC0) != 0x80) ++n;
        }
        return n;
    };
    std::string t;
    const char* sent = "今天天气真不错，适合出门散步，顺便听一首喜欢的歌，放松一下心情。";
    while (static_cast<int>(n_chars(t)) < min_chars) t += sent;
    return t;
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

TEST_CASE("Kokoro predict_stream real chunking", "[tts][tts-kokoro]") {
    // >120 字符 + chunk_frames=120 → split_for_synthesis 切成多个字符块 → 断言音频回调次数 >1。
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
    const std::string text = long_sentence(300);
    std::vector<float> whole;
    REQUIRE(koro.predict(text, "zf_001", 1.0f, &whole));
    REQUIRE_FALSE(whole.empty());
    auto st = collect_chunked(&koro, text, "zf_001", 120);
    REQUIRE(st.ok);
    REQUIRE(st.audio_callbacks > 1);
    REQUIRE(st.all_finite);
    REQUIRE_FALSE(st.audio.empty());
}

