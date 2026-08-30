//
// Created by aichao on 2026/8/30.
//

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "csrc/audio/tts/utils.h"
#include "csrc/audio/tts/audio8/audio8.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

// 模型根：环境变量 MODELDEPLOY_TTS_MODELS_DIR 或仓库内 test_data 下的 tt 子目录。
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
    const std::string voice = argc > 1 ? argv[1] : "demo";
    const std::string model_dir = models_dir() + "/audio8_preview";
    std::cout << "Audio8 model_dir: " << model_dir << ", voice: " << voice << std::endl;

    modeldeploy::RuntimeOption option;
    modeldeploy::audio::tts::Audio8 audio8;
    const bool loaded = audio8.Load(model_dir, option);
    if (!loaded) {
        std::cerr << "FAILED: Audio8 load failed: " << model_dir << std::endl;
        return 1;
    }
    std::cout << "Audio8 sample_rate: " << audio8.get_sample_rate() << std::endl;

    const std::string text = "你好，世界。";
    std::vector<float> out_data;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = audio8.predict(text, voice, 1.0f, &out_data);
    const auto t1 = std::chrono::steady_clock::now();
    if (!ok || out_data.empty()) {
        std::cerr << "FAILED: Audio8 predict" << std::endl;
        return 1;
    }
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    const double audio_secs = static_cast<double>(out_data.size()) / audio8.get_sample_rate();
    std::cout << "Audio8 predict ok: samples=" << out_data.size()
              << " audio_secs=" << audio_secs
              << " wall_time=" << wall << "s" << std::endl;

    modeldeploy::audio::tts::write_wave(std::string("out_audio8.wav"),
                                        audio8.get_sample_rate(), out_data.data(),
                                        out_data.size());
    std::cout << "Saved out_audio8.wav" << std::endl;
    return 0;
}
