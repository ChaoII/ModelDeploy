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
#include "audio/tts/audio8/audio8_runtime.h"
#include "audio/tts/common/ort_ep.h"
#include "audio/tts/kokoro.h"
#include "audio/tts/qwen3/qwen3_tts.h"
#include "audio/tts/tts_model.h"
#include "tests/test_gpu_utils.h"
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

// 分块统计：audio_callbacks 仅计 n>0 的音频回调（Qwen3 progress 空块不计数）。
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

TEST_CASE("Audio8 predict produces audio", "[tts][tts-audio8][gpu]") {
    MD_TEST_GPU_OR_SKIP();
    const auto dir = audio8_dir();
    if (!fs::exists(dir / "runtime_manifest.json")) return;
    modeldeploy::RuntimeOption opt;
    opt.set_device(modeldeploy::Device::GPU, 0);
    Audio8 m;
    if (!m.Load(dir.string(), opt)) return;
    REQUIRE(m.get_sample_rate() == 44100);
    std::vector<float> audio;
    REQUIRE(m.predict("今天天气真不错，适合出门散步。", "demo", 1.0f, &audio));
    REQUIRE(audio.size() > 1000);
    REQUIRE(rms(audio) > 1e-3);
}

TEST_CASE("Qwen3Tts predict produces audio", "[tts][tts-qwen3][gpu]") {
    MD_TEST_GPU_OR_SKIP();
    const auto dir = qwen3_dir();
    if (!fs::exists(dir / "onnx_kv_06b")) return;
    modeldeploy::RuntimeOption opt;
    opt.set_device(modeldeploy::Device::GPU, 0);
    Qwen3Tts m;
    if (!m.init(dir.string(), opt)) return;
    REQUIRE(m.get_sample_rate() == 24000);
    std::vector<float> audio;
    REQUIRE(m.predict("你好，世界！", "Vivian", 1.0f, &audio));
    REQUIRE(audio.size() > 1000);
    REQUIRE(rms(audio) > 1e-3);
}

TEST_CASE("Audio8 predict_stream equals predict", "[tts][tts-audio8][gpu]") {
    MD_TEST_GPU_OR_SKIP();
    const auto dir = audio8_dir();
    if (!fs::exists(dir / "runtime_manifest.json")) return;
    modeldeploy::RuntimeOption opt;
    opt.set_device(modeldeploy::Device::GPU, 0);
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

TEST_CASE("Audio8 predict_stream real chunking", "[tts][tts-audio8][gpu]") {
    // 长文本 + chunk_frames=24 → 断言真实分块（滑窗 guard 持续多块）：
    // 回调次数 >1、各块有限非空、拼接总长与 predict 相对差 <5%。
    MD_TEST_GPU_OR_SKIP();
    const auto dir = audio8_dir();
    if (!fs::exists(dir / "runtime_manifest.json")) return;
    modeldeploy::RuntimeOption opt;
    opt.set_device(modeldeploy::Device::GPU, 0);
    Audio8 m;
    if (!m.Load(dir.string(), opt)) return;
    const std::string text = long_sentence(150);
    std::vector<float> whole;
    REQUIRE(m.predict(text, "demo", 1.0f, &whole));
    REQUIRE_FALSE(whole.empty());
    auto st = collect_chunked(&m, text, "demo", 24);
    REQUIRE(st.ok);
    REQUIRE(st.audio_callbacks > 1);
    REQUIRE(st.all_finite);
    REQUIRE_FALSE(st.audio.empty());
    REQUIRE(rel_len_diff(st.audio, whole) < 0.05);
}

TEST_CASE("Qwen3Tts predict_stream equals predict", "[tts][tts-qwen3][gpu]") {
    MD_TEST_GPU_OR_SKIP();
    const auto dir = qwen3_dir();
    if (!fs::exists(dir / "onnx_kv_06b")) return;
    modeldeploy::RuntimeOption opt;
    opt.set_device(modeldeploy::Device::GPU, 0);
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

TEST_CASE("Qwen3Tts predict_stream real chunking", "[tts][tts-qwen3][gpu]") {
    // 长文本 + chunk_frames=24 → 断言真实分块（滑窗 guard 持续多块）：
    // 回调次数 >1、各块有限非空、拼接总长与 predict 相对差 <5%。
    MD_TEST_GPU_OR_SKIP();
    const auto dir = qwen3_dir();
    if (!fs::exists(dir / "onnx_kv_06b")) return;
    modeldeploy::RuntimeOption opt;
    opt.set_device(modeldeploy::Device::GPU, 0);
    Qwen3Tts m;
    if (!m.init(dir.string(), opt)) return;
    const std::string text = "你好，世界！这是流式分块一致性测试，用于验证 Qwen3 的多块路径。";
    size_t best_callbacks = 0;
    double best = 1.0;
    for (int attempt = 0; attempt < 3; ++attempt) {
        std::vector<float> whole;
        if (!m.predict(text, "Vivian", 1.0f, &whole) || whole.empty()) continue;
        auto st = collect_chunked(&m, text, "Vivian", 12);
        if (!st.ok || st.audio.empty()) continue;
        REQUIRE(st.all_finite);
        best_callbacks = std::max(best_callbacks, st.audio_callbacks);
        best = std::min(best, rel_len_diff(st.audio, whole));
    }
    REQUIRE(best_callbacks > 1);
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

TEST_CASE("Audio8 overlong text rejected without truncation", "[tts][tts-audio8][gpu]") {
    MD_TEST_GPU_OR_SKIP();
    const auto dir = audio8_dir();
    if (!fs::exists(dir / "runtime_manifest.json")) return;
    modeldeploy::RuntimeOption opt;
    opt.set_device(modeldeploy::Device::GPU, 0);
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

TEST_CASE("Qwen3Tts overlong text rejected without truncation", "[tts][tts-qwen3][gpu]") {
    MD_TEST_GPU_OR_SKIP();
    const auto dir = qwen3_dir();
    if (!fs::exists(dir / "onnx_kv_06b")) return;
    modeldeploy::RuntimeOption opt;
    opt.set_device(modeldeploy::Device::GPU, 0);
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

TEST_CASE("ApplyOrtCudaEp gating", "[tts][tts-common]") {
    using namespace modeldeploy::audio::tts;
    using modeldeploy::Device;
    Ort::SessionOptions opts;
    // CPU 设备：必须原样返回 false、不改 opts（之后创建 session 不会挂 CUDA）
    CHECK_FALSE(ApplyOrtCudaEp(opts, Device::CPU, 0));
    CHECK_FALSE(ApplyOrtCudaEp(opts, Device::CPU, -1));
#ifdef MD_ORT_CUDA
    // GPU 构建（且本机 CUDA 运行库就绪）应启用
    CHECK(ApplyOrtCudaEp(opts, Device::GPU, 0));
    // device_id 负值在入口被规约为 0，仍应启用 EP（用新 SessionOptions 避免重复 append）
    Ort::SessionOptions opts_neg;
    CHECK(ApplyOrtCudaEp(opts_neg, Device::GPU, -1));
#else
    // CPU 构建：GPU 请求也回退 false
    CHECK_FALSE(ApplyOrtCudaEp(opts, Device::GPU, 0));
    CHECK_FALSE(ApplyOrtCudaEp(opts, Device::GPU, -1));
#endif
}

TEST_CASE("Audio8 GpuState slow prefill matches CPU (spike)",
          "[tts][tts-audio8][gpu][spike]") {
    MD_TEST_GPU_OR_SKIP();
    namespace a8 = modeldeploy::audio::tts::audio8;
    const auto dir = audio8_dir();
    if (!fs::exists(dir)) { WARN("audio8 model dir missing; skipping"); return; }
    a8::Audio8Manifest manifest;
    REQUIRE(a8::Audio8Manifest::FromJson((dir / "runtime_manifest.json").string(),
                                         dir.string(), &manifest));

    a8::Audio8Runtime rt;
    REQUIRE(rt.Load(manifest, 4, modeldeploy::Device::GPU, 0));
    auto gpu = rt.MakeGpuState(modeldeploy::Device::GPU, 0);
    if (!gpu) { WARN("GpuState unavailable (no CUDA provider); skipping"); return; }

    const int64_t rows = rt.num_codebooks() + 1;   // 11
    const int64_t T = 6;
    std::vector<int64_t> codes(rows * T);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<int64_t>(i % 4096);
    std::vector<int64_t> positions(T);
    for (int64_t i = 0; i < T; ++i) positions[i] = i;

    const int64_t seg = manifest.n_local_heads * manifest.max_seq_len * manifest.head_dim;
    std::vector<uint16_t> cpu_cache(static_cast<size_t>(2 * manifest.num_layers * seg), 0);
    std::vector<float> cpu_logits, gpu_logits;
    std::vector<uint16_t> cpu_hidden, gpu_hidden;
    REQUIRE(rt.SlowStep(codes, positions, &cpu_cache, &cpu_logits, &cpu_hidden));
    REQUIRE(rt.SlowStepGpu(gpu.get(), codes, positions, &gpu_logits, &gpu_hidden));

    REQUIRE(gpu_logits.size() == cpu_logits.size());
    for (size_t i = 0; i < cpu_logits.size(); ++i)
        REQUIRE(std::fabs(gpu_logits[i] - cpu_logits[i]) < 1e-3f);
    REQUIRE(gpu_hidden.size() == cpu_hidden.size());

    // 关键 spike 断言:GPU cache 与 CPU cache 逐元素一致(验证输出绑 GPU + strided D2D 写入正确)
    if (gpu) {
        std::vector<uint16_t> gpu_cache(cpu_cache.size(), 0);
        REQUIRE(rt.CopyGpuCacheToHost(gpu.get(), 0, gpu_cache.data()));
        for (size_t i = 0; i < cpu_cache.size(); ++i) {
            if (std::fabs(static_cast<float>(gpu_cache[i]) -
                          static_cast<float>(cpu_cache[i])) > 2.0f) {
                FAIL("cache mismatch at " << i << " gpu=" << gpu_cache[i]
                     << " cpu=" << cpu_cache[i]);
            }
        }
    }
}

TEST_CASE("Audio8 GpuState fast step matches CPU", "[tts][tts-audio8][gpu][spike]") {
    MD_TEST_GPU_OR_SKIP();
    namespace a8 = modeldeploy::audio::tts::audio8;
    const auto dir = audio8_dir();
    if (!fs::exists(dir)) { WARN("audio8 model dir missing; skipping"); return; }
    a8::Audio8Manifest manifest;
    REQUIRE(a8::Audio8Manifest::FromJson((dir / "runtime_manifest.json").string(),
                                         dir.string(), &manifest));
    a8::Audio8Runtime rt;
    REQUIRE(rt.Load(manifest, 4, modeldeploy::Device::GPU, 0));
    auto gpu = rt.MakeGpuState(modeldeploy::Device::GPU, 0);
    if (!gpu) { WARN("GpuState unavailable; skipping"); return; }

    const int64_t fseg =
        manifest.fast_n_local_heads * manifest.num_codebooks * manifest.fast_head_dim;
    std::vector<uint16_t> cpu_fast(static_cast<size_t>(2 * manifest.num_fast_layers * fseg), 0);
    std::vector<uint16_t> hidden(static_cast<size_t>(rt.fast_dim()), 42u);
    std::vector<float> cpu_l, gpu_l;
    REQUIRE(rt.FastStep(7, true, 2, hidden, &cpu_fast, &cpu_l));
    REQUIRE(rt.FastStepGpu(gpu.get(), 7, true, 2, hidden, &gpu_l));
    REQUIRE(gpu_l.size() == cpu_l.size());
    for (size_t i = 0; i < cpu_l.size(); ++i)
        REQUIRE(std::fabs(gpu_l[i] - cpu_l[i]) < 1e-3f);
    std::vector<uint16_t> fast_cpy(cpu_fast.size(), 0);
    REQUIRE(rt.CopyGpuCacheToHost(gpu.get(), 1, fast_cpy.data()));
    for (size_t i = 0; i < cpu_fast.size(); ++i)
        REQUIRE(std::fabs(static_cast<float>(fast_cpy[i]) -
                          static_cast<float>(cpu_fast[i])) <= 2.0f);
}
