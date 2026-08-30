//
// Created by aichao on 2026/8/30.
//

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "csrc/audio/tts/utils.h"
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

}  // namespace

int32_t main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    std::wcout.imbue(std::locale(""));
#endif
    const std::string model_dir = models_dir() + "/qwen3_tts_0.6b";
    std::string ref_audio;
    std::string ref_text;
    std::string lang = "auto";
    for (int i = 1; i + 2 < argc; ++i) {
        if (std::string(argv[i]) == "--clone") {
            ref_audio = argv[i + 1];
            ref_text = argv[i + 2];
            if (i + 3 < argc) lang = argv[i + 3];
            break;
        }
    }
    std::cout << "Qwen3Tts model_dir: " << model_dir << std::endl;

    modeldeploy::RuntimeOption option;
    modeldeploy::audio::tts::Qwen3Tts qwen3(model_dir, option);
    const size_t spk_count = qwen3.get_supported_speakers().size();
    std::cout << "Qwen3Tts loaded, preset speakers=" << spk_count
              << ", sample_rate=" << qwen3.get_sample_rate() << std::endl;

    const std::string speaker = "Vivian";
    std::vector<float> out_data;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = qwen3.predict("你好，世界。", speaker, 1.0f, &out_data);
    const auto t1 = std::chrono::steady_clock::now();
    if (!ok || out_data.empty()) {
        std::cerr << "FAILED: Qwen3Tts predict" << std::endl;
        return 1;
    }
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    const double audio_secs = static_cast<double>(out_data.size()) / qwen3.get_sample_rate();
    std::cout << "Qwen3Tts predict ok (" << speaker << "): samples=" << out_data.size()
              << " audio_secs=" << audio_secs << " wall_time=" << wall << "s" << std::endl;
    modeldeploy::audio::tts::write_wave(std::string("out_qwen3.wav"),
                                        qwen3.get_sample_rate(), out_data.data(),
                                        out_data.size());
    std::cout << "Saved out_qwen3.wav" << std::endl;

    if (!ref_audio.empty()) {
        std::cout << "Clone: ref_audio=" << ref_audio << ", ref_text=" << ref_text
                  << ", lang=" << lang << std::endl;
        std::vector<float> clone_data;
        const auto t2 = std::chrono::steady_clock::now();
        const bool cok = qwen3.clone("你好，这是声音克隆的演示。", ref_audio,
                                     ref_text, lang, &clone_data);
        const auto t3 = std::chrono::steady_clock::now();
        if (!cok || clone_data.empty()) {
            std::cerr << "FAILED: Qwen3Tts clone" << std::endl;
            return 1;
        }
        const double clone_wall = std::chrono::duration<double>(t3 - t2).count();
        const double clone_secs =
            static_cast<double>(clone_data.size()) / qwen3.get_sample_rate();
        std::cout << "Qwen3Tts clone ok: samples=" << clone_data.size()
                  << " audio_secs=" << clone_secs << " wall_time=" << clone_wall << "s"
                  << std::endl;
        modeldeploy::audio::tts::write_wave(std::string("out_qwen3_clone.wav"),
                                            qwen3.get_sample_rate(), clone_data.data(),
                                            clone_data.size());
        std::cout << "Saved out_qwen3_clone.wav" << std::endl;
    }
    return 0;
}
