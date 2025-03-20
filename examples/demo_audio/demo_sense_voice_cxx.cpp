//
// Created by aichao on 2025/2/24.
//
#include <chrono>  // NOLINT
#include <iostream>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif
#include "csrc/audio/sherpa-onnx/csrc/cxx-api.h"

int32_t main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    using namespace sherpa_onnx::cxx; // NOLINT
    OfflineRecognizerConfig config;

    config.model_config.sense_voice.model =
        "../../test_data/test_models/sense-voice-zh-en-ja-ko-yue/model.int8.onnx";
    config.model_config.sense_voice.use_itn = true;
    config.model_config.sense_voice.language = "auto";
    config.model_config.tokens =
        "../../test_data/test_models/sense-voice-zh-en-ja-ko-yue/tokens.txt";
    config.model_config.num_threads = 1;
    std::cout << "Loading model\n";
    OfflineRecognizer recognizer = OfflineRecognizer::Create(config);
    if (!recognizer.Get()) {
        std::cerr << "Please check your config\n";
        return -1;
    }
    std::cout << "Loading model done\n";
    std::string wave_filename =
        "../../test_data/test_models/sense-voice-zh-en-ja-ko-yue/test_wavs/vad.wav";
    Wave wave = ReadWave(wave_filename);
    if (wave.samples.empty()) {
        std::cerr << "Failed to read: '" << wave_filename << "'\n";
        return -1;
    }
    std::cout << "Start recognition\n";
    const auto begin = std::chrono::steady_clock::now();

    OfflineStream stream = recognizer.CreateStream();
    stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                          wave.samples.size());
    recognizer.Decode(&stream);
    OfflineRecognizerResult result = recognizer.GetResult(&stream);
    const auto end = std::chrono::steady_clock::now();
    const float elapsed_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
        .count() / 1000.f;
    const float duration = wave.samples.size() / static_cast<float>(wave.sample_rate);
    const float rtf = elapsed_seconds / duration;
    std::cout << "text: " << result.text << "\n";
    printf("Number of threads: %d\n", config.model_config.num_threads);
    printf("Duration: %.3fs\n", duration);
    printf("Elapsed seconds: %.3fs\n", elapsed_seconds);
    printf("(Real time factor) RTF = %.3f / %.3f = %.3f\n", elapsed_seconds,
           duration, rtf);
    return 0;
}
