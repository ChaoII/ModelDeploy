//
// Created by aichao on 2025/5/19.
//


#include <algorithm>
#include <iostream>
#include <samplerate/include/samplerate.h>
#include "csrc/core/md_log.h"
#include "csrc/audio//asr_pipeline.h"

namespace modeldeploy {
    static std::vector<float> resample(const std::vector<float>& input, const int inputRate) {
        constexpr int outputRate = 16000;
        if (inputRate == outputRate) {
            return input;
        }


        if (input.empty()) return {};
        if (inputRate <= 0) {
            MD_LOG_ERROR << "Sample rates must be positive" << std::endl;
        }

        const double ratio = static_cast<double>(outputRate) / inputRate;
        std::vector<float> output(static_cast<size_t>(input.size() * ratio + 0.5));

        SRC_DATA src_data;
        src_data.data_in = const_cast<float*>(input.data());
        src_data.input_frames = static_cast<long>(input.size());
        src_data.data_out = output.data();
        src_data.output_frames = static_cast<long>(output.size());
        src_data.src_ratio = ratio;
        src_data.end_of_input = 1;

        constexpr int converterType = 2;
        if (const int error = src_simple(&src_data, converterType, 1)) {
            MD_LOG_ERROR << "Resampling failed: " << src_strerror(error) << std::endl;
        }
        output.resize(static_cast<size_t>(src_data.output_frames_gen));
        return output;
    }

    AAsr::AAsr(const std::string& asr_onnx,
               const std::string& tokens,
               const std::string& vad_onnx) {
        _running = true;
        _sence_voice = std::make_unique<SenseVoice>(asr_onnx, tokens);
        _vad = std::make_unique<SileroVAD>(vad_onnx);
        _th = std::thread(&AAsr::run, this);
    }

    AAsr::~AAsr() {
        _running = false;
        _th.join();
    }

    void AAsr::push_data(const std::vector<float>& data, int inputRate) {
        std::lock_guard lk(_mx);
        const auto out = resample(data, inputRate);
        for (const auto& d : out) {
            _deque.push_back(d);
        }
    }


    void AAsr::wait_finish() {
        std::unique_lock lk(_mx);
        _cv.wait(lk, [this] { return _deque.size() <= 512; });
        _running.store(true);
    }

    void AAsr::run() {
        int idx = 0;
        while (_running.load()) {
            std::vector<float> data;
            {
                std::lock_guard lk(_mx);
                data.clear();
                if (_deque.size() >= 512) {
                    for (int i = 0; i < 512; ++i) {
                        data.push_back(_deque.front());
                        _deque.pop_front();
                    }
                }
                else {
                    _cv.notify_all();
                }
            }
            if (data.size() == 512) {
                std::string trigger;
                _vad->predict(data, &trigger);
                if (trigger != "none") {
                    std::cout << 512 * idx++ << " " << trigger << std::endl;
                }
                // for asr detect
                std::transform(data.begin(), data.end(), data.begin(),
                               [](const float x) { return x * 32768.0f; });
                if (trigger == "start") {
                    // detect voice
                    _curWav.insert(_curWav.end(), data.begin(), data.end());
                }
                else if (trigger == "end") {
                    // detect silence
                    _curWav.insert(_curWav.end(), data.begin(), data.end());
                    std::string result;
                    _sence_voice->predict(_curWav, &result);
                    if (_onAsr) {
                        _onAsr(result);
                    }
                    _curWav.clear();
                }
                else if (!_curWav.empty()) {
                    _curWav.insert(_curWav.end(), data.begin(), data.end());
                }
            }
        }

        if (!_curWav.empty()) {
            std::string result;
            _sence_voice->predict(_curWav, &result);
            if (_onAsr) {
                _onAsr(result);
            }
        }
        std::cout << "Asr run exit" << std::endl;
    }
}
