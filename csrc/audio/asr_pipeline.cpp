//
// Created by aichao on 2025/5/19.
//


#include <algorithm>
#include <iostream>
#include <samplerate/include/samplerate.h>
#include "core/md_log.h"
#include "audio//asr_pipeline.h"

namespace modeldeploy::audio {
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
        running_ = true;
        modeldeploy::RuntimeOption option;
        option.use_gpu(0);
        sense_voice_ = std::make_unique<asr::SenseVoice>(asr_onnx, tokens, option);
        vad_ = std::make_unique<vad::SileroVAD>(vad_onnx, option);
        th_ = std::thread(&AAsr::run, this);
    }

    AAsr::~AAsr() {
        running_ = false;
        th_.join();
    }

    void AAsr::push_data(const std::vector<float>& data, int inputRate) {
        std::lock_guard lk(mutex_);
        const auto out = resample(data, inputRate);
        for (const auto& d : out) {
            deque_.push_back(d);
        }
    }


    void AAsr::wait_finish() {
        std::unique_lock lk(mutex_);
        cv_.wait(lk, [this] { return deque_.size() <= 512; });
        running_.store(true);
    }

    void AAsr::run() {
        int idx = 0;
        while (running_.load()) {
            std::vector<float> data;
            {
                std::lock_guard lk(mutex_);
                data.clear();
                if (deque_.size() >= 512) {
                    for (int i = 0; i < 512; ++i) {
                        data.push_back(deque_.front());
                        deque_.pop_front();
                    }
                }
                else {
                    cv_.notify_all();
                }
            }
            if (data.size() == 512) {
                std::string trigger;
                vad_->predict(data, &trigger);
                if (trigger != "none") {
                    std::cout << 512 * idx++ << " " << trigger << std::endl;
                }
                // for asr detect
                std::transform(data.begin(), data.end(), data.begin(),
                               [](const float x) { return x * 32768.0f; });
                if (trigger == "start") {
                    // detect voice
                    cur_wav_.insert(cur_wav_.end(), data.begin(), data.end());
                }
                else if (trigger == "end") {
                    // detect silence
                    cur_wav_.insert(cur_wav_.end(), data.begin(), data.end());
                    std::string result;
                    sense_voice_->predict(cur_wav_, &result);
                    if (on_asr_) {
                        on_asr_(result);
                    }
                    cur_wav_.clear();
                }
                else if (!cur_wav_.empty()) {
                    cur_wav_.insert(cur_wav_.end(), data.begin(), data.end());
                }
            }
        }

        if (!cur_wav_.empty()) {
            std::string result;
            sense_voice_->predict(cur_wav_, &result);
            if (on_asr_) {
                on_asr_(result);
            }
        }
        std::cout << "Asr run exit" << std::endl;
    }
}
