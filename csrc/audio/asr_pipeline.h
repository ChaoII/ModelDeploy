//
// Created by aichao on 2025/5/19.
//

#pragma once
#include <string>
#include <memory>
#include <deque>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "csrc/base_model.h"
#include "csrc/audio/asr/sense_voice.h"
#include "csrc/audio/vad/silero_vad.h"

namespace modeldeploy {
    class MODELDEPLOY_CXX_EXPORT AAsr {
    public:
        AAsr(const std::string& asr_onnx, const std::string& tokens, const std::string& vad_onnx);
        ~AAsr();
        void push_data(const std::vector<float>& data, int sampleRate);
        void run();
        std::atomic<bool> _running;
        std::thread _th;
        std::unique_ptr<SenseVoice> _sence_voice;
        std::function<void(const std::string& asr)> _onAsr;
        void wait_finish();

    private:
        std::unique_ptr<SileroVAD> _vad;

        std::deque<float> _deque;
        std::vector<float> _curWav;
        std::mutex _mx;
        std::condition_variable _cv;
    };
}
