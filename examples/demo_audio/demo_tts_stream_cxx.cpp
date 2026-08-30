//
// Created by aichao on 2026/8/30.
//

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "csrc/audio/tts/utils.h"
#include "csrc/audio/tts/kokoro.h"
#include "csrc/audio/tts/audio8/audio8.h"
#include "csrc/audio/tts/qwen3/qwen3_tts.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

std::string models_dir() {
    const char* env = std::getenv("MODELDEPLOY_TTS_MODELS_DIR");
    if (env && *env) return std::string(env);
    return "../../test_data/test_models/tts";
}

template <typename ModelT>
bool run_stream(const std::string& name, ModelT* model, const std::string& text,
                const std::string& voice, int chunk_frames, const std::string& out_wav) {
    std::vector<float> accumulated;
    int audio_callbacks = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = model->predict_stream(
        text, voice, 1.0f, chunk_frames,
        [&](const float* samples, int n, float progress) -> bool {
            if (samples && n > 0) {
                ++audio_callbacks;
                accumulated.insert(accumulated.end(), samples, samples + n);
                std::cout << "  [" << name << "] progress=" << static_cast<int>(progress * 100.0f)
                          << "% chunk=" << n << " samples=" << accumulated.size() << std::endl;
            } else {
                // progress 空块（Qwen3 mode B AR 阶段）：跳过音频处理，仅记进度
                std::cout << "  [" << name << "] progress=" << static_cast<int>(progress * 100.0f)
                          << "% (no audio)" << std::endl;
            }
            return true;
        });
    const auto t1 = std::chrono::steady_clock::now();
    if (!ok || accumulated.empty()) {
        std::cerr << "FAILED: " << name << " predict_stream" << std::endl;
        return false;
    }
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    const double audio_secs =
        static_cast<double>(accumulated.size()) / static_cast<double>(model->get_sample_rate());
    std::cout << name << " stream ok: audio_callbacks=" << audio_callbacks
              << " samples_total=" << accumulated.size()
              << " audio_secs=" << audio_secs << " wall_time=" << wall << "s" << std::endl;
    modeldeploy::audio::tts::write_wave(out_wav, model->get_sample_rate(),
                                        accumulated.data(), accumulated.size());
    std::cout << "Saved " << out_wav << " (sample_rate=" << model->get_sample_rate() << ")"
              << std::endl;
    return true;
}

}  // namespace

int32_t main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    std::wcout.imbue(std::locale(""));
#endif
    // 各模型 chunk_frames 取较小值（单位模型相关：Audio8=AR 码帧 / Qwen3=vq 码帧 / Kokoro=字符），
    // 以便演示真实流式的多次音频回调。
    const std::string models_root = models_dir();

    // ---- Kokoro（chunk_frames=字符数，>120 字触发多块）----
    {
        const std::string kk = models_root + "/kokoro_v1_1";
        const std::string kokoro_onnx = kk + "/model.onnx";
        const std::string tokens = kk + "/tokens.txt";
        const std::vector<std::string> lexicons = {
            kk + "/lexicon-us-en.txt",
            kk + "/lexicon-zh.txt"};
        const std::string voice_bin = kk + "/voices.bin";
        const std::string jieba_dir = kk + "/dict/";
        const std::string text_normalization_dir = models_root + "/";
        modeldeploy::RuntimeOption option;
        modeldeploy::audio::tts::Kokoro kokoro(kokoro_onnx, tokens, lexicons, voice_bin,
                                               jieba_dir, text_normalization_dir, option);
        if (!kokoro.is_initialized()) {
            std::cerr << "FAILED: Kokoro load " << kokoro_onnx << std::endl;
            return 1;
        }
        std::cout << "Kokoro loaded, sample_rate=" << kokoro.get_sample_rate() << std::endl;
        const std::string long_text =
            "这是一段明显超过一百二十个字符的流式合成测试文本，用来验证 Kokoro 按字符分块时的多次音频回调。"
            "确保每个分块都能被独立回调并且拼接后得到完整音频。再补充一句让文本更长一些，以稳定触发分块路径。"
            "继续重复更多内容，确保这段文本长度确实超过一百二十个字符，从而触发多个音频回调，演示真实流式合成。";
        const bool ok = run_stream(
            "kokoro", &kokoro, long_text, "zf_001", 120, "out_stream_kokoro.wav");
        if (!ok) return 1;
    }

    // ---- Audio8 ----
    {
        const std::string model_dir = models_root + "/audio8_preview";
        modeldeploy::RuntimeOption option;
        modeldeploy::audio::tts::Audio8 audio8;
        if (!audio8.Load(model_dir, option)) {
            std::cerr << "FAILED: Audio8 load " << model_dir << std::endl;
            return 1;
        }
        std::cout << "Audio8 loaded, sample_rate=" << audio8.get_sample_rate() << std::endl;
        run_stream("audio8", &audio8, "你好，世界。这是流式合成的测试音频。", "demo",
                   24, "out_stream_audio8.wav");
    }

    // ---- Qwen3 ----
    {
        const std::string model_dir = models_root + "/qwen3_tts_0.6b";
        modeldeploy::RuntimeOption option;
        modeldeploy::audio::tts::Qwen3Tts qwen3(model_dir, option);
        std::cout << "Qwen3Tts loaded, sample_rate=" << qwen3.get_sample_rate() << std::endl;
        const bool ok = run_stream("qwen3", &qwen3, "你好，世界。这是流式合成的测试音频。",
                                   "Vivian", 12, "out_stream_qwen3.wav");
        if (!ok) return 1;
    }

    std::cout << "All three stream TTS finished." << std::endl;
    return 0;
}
