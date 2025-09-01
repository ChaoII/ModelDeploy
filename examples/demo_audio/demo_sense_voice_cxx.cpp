//
// Created by aichao on 2025/5/19.
//

#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include "csrc/utils/wave_helper.h"
#include "csrc/audio/asr_pipeline.h"


void onAsr(const std::string& asr) {
    std::cout << "asr:" << asr << "\n------------" << std::endl;
}


int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try {
        // 初始化推理引擎
        std::string asr_onnx = "../../test_data/test_models/sense_voice/model.int8.onnx";
        std::string tokens = "../../test_data/test_models/sense_voice/tokens.txt";
        std::string vad_onnx = "../../test_data/test_models/sense_voice/silero_vad.onnx";
        const std::string wav = "../../test_data/test_models/sense_voice/test_wavs/self_test.wav";

        const auto asr = std::make_unique<modeldeploy::audio::AAsr>(asr_onnx, tokens, vad_onnx);
        asr->on_asr_ = onAsr;
        std::vector<float> data;
        int32_t sampling_rate = 16000;
        load_wav_file(wav.c_str(), &sampling_rate, data);

        std::vector<float> tmp;
        for (auto i : data) {
            tmp.push_back(i * 37268);
        }
        std::string result_raw;
        asr->sense_voice_->predict(tmp, &result_raw);
        std::cout << "AsrResult:" << result_raw << std::endl;
        for (int i = 0; i < data.size() / 512; ++i) {
            std::vector tmp(data.begin() + i * 512, data.begin() + i * 512 + 512);
            asr->push_data(tmp, sampling_rate);
        }
        asr->wait_finish();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
