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

namespace modeldeploy::audio {
    class MODELDEPLOY_CXX_EXPORT AAsr {
    public:
        AAsr(const std::string& asr_onnx, const std::string& tokens, const std::string& vad_onnx);
        ~AAsr();
        void push_data(const std::vector<float>& data, int sampleRate);
        void run();
        std::atomic<bool> running_;
        std::thread th_;
        std::unique_ptr<asr::SenseVoice> sense_voice_;
        std::function<void(const std::string& asr)> on_asr_;
        void wait_finish();

    private:
        std::unique_ptr<vad::SileroVAD> vad_;
        std::deque<float> deque_;
        std::vector<float> cur_wav_;
        std::mutex mutex_;
        std::condition_variable cv_;
    };
}
