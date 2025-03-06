//
// Created by aichao on 2025/2/26.
//
#include <chrono>  // NOLINT
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#endif

#include "capi/audio/asr/sense_voice_capi.h"

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    MDStatusCode ret;
    MDModel model;
    MDSenseVoiceParameters parameters = {
        "../tests/test_models/sense-voice-zh-en-ja-ko-yue/model.int8.onnx",
        1,
        "auto",
        "../tests/test_models/sense-voice-zh-en-ja-ko-yue/tokens.txt",
        8,
        0,
    };
    if ((ret = md_create_sense_voice_model(&model, &parameters)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    MDASRResult result;
    if ((ret = md_sense_voice_model_predict(&model, "../tests/vad_example.wav", &result)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    md_free_sense_voice_result(&result);
    md_free_sense_voice_model(&model);
    return ret;
}
