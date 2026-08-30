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
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = model->predict_stream(
        text, voice, 1.0f, chunk_frames,
        [&](const float* samples, int n, float progress) -> bool {
            accumulated.insert(accumulated.end(), samples, samples + n);
            std::cout << "  [" << name << "] progress=" << static_cast<int>(progress * 100.0f)
                      << "% chunk=" << n << " accumulated=" << accumulated.size() << std::endl;
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
    std::cout << name << " stream ok: chunks_total=" << accumulated.size()
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
    const int chunk_frames = 480;
    const std::string models_root = models_dir();

    // ---- Kokoro ----
    {
        const std::string kokoro_onnx = "../../test_data/test_models/onnx/kokoro_v1_1/model.onnx";
        const std::string tokens = "../../test_data/test_models/onnx/kokoro_v1_1/tokens.txt";
        const std::vector<std::string> lexicons = {
            "../../test_data/test_models/onnx/kokoro_v1_1/lexicon-us-en.txt",
            "../../test_data/test_models/onnx/kokoro_v1_1/lexicon-zh.txt"};
        const std::string voice_bin = "../../test_data/test_models/onnx/kokoro_v1_1/voices.bin";
        const std::string jieba_dir = "../../test_data/test_models/onnx/kokoro_v1_1/dict/";
        const std::string text_normalization_dir = "../../test_data/";
        modeldeploy::RuntimeOption option;
        modeldeploy::audio::tts::Kokoro kokoro(kokoro_onnx, tokens, lexicons, voice_bin,
                                               jieba_dir, text_normalization_dir, option);
        std::cout << "Kokoro loaded, sample_rate=" << kokoro.get_sample_rate() << std::endl;
        run_stream("kokoro", &kokoro, "大家好，这是流式合成的测试音频。", "zf_001",
                   chunk_frames, "out_stream_kokoro.wav");
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
                   chunk_frames, "out_stream_audio8.wav");
    }

    // ---- Qwen3 ----
    {
        const std::string model_dir = models_root + "/qwen3_tts_0.6b";
        modeldeploy::RuntimeOption option;
        modeldeploy::audio::tts::Qwen3Tts qwen3(model_dir, option);
        std::cout << "Qwen3Tts loaded, sample_rate=" << qwen3.get_sample_rate() << std::endl;
        const bool ok = run_stream("qwen3", &qwen3, "你好，世界。这是流式合成的测试音频。",
                                   "Vivian", chunk_frames, "out_stream_qwen3.wav");
        if (!ok) return 1;
    }

    std::cout << "All three stream TTS finished." << std::endl;
    return 0;
}
