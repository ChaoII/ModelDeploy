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

    AAsr::AAsr(const std::string& asr_onnx,
               const std::string& tokens,
               const std::string& vad_onnx,
               const std::string& stream_encoder,
               const std::string& stream_decoder,
               const std::string& stream_tokens,
               float offline_conf_threshold) {
        running_ = true;
        conf_threshold_ = offline_conf_threshold;
        modeldeploy::RuntimeOption option;
        option.use_gpu(0);
        sense_voice_ = std::make_unique<asr::SenseVoice>(asr_onnx, tokens, option);
        vad_ = std::make_unique<vad::SileroVAD>(vad_onnx, option);
        if (!stream_encoder.empty() && !stream_decoder.empty() && !stream_tokens.empty()) {
            streaming_ = std::make_unique<asr::ParaformerStreamingAsr>(
                stream_encoder, stream_decoder, stream_tokens, 16000, 2);
        }
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

    void AAsr::emit_final() {
        std::string final;
        float conf = 0.f;
        if (streaming_) {
            // 句末：flush 末尾短块，取累积流式置信度与全文
            streaming_->input_finished();
            asr::StreamingAsrResult r;
            streaming_->decode(true, &r);
            conf = r.confidence;
        }
        if (streaming_ && streaming_->is_initialized() &&
            conf >= conf_threshold_ && !streaming_->text().empty()) {
            final = streaming_->text();          // 流式足够置信 → 免离线
        } else {
            sense_voice_->predict(cur_wav_, &final);  // 离线精修
        }
        if (on_asr_) {
            on_asr_(final);
        }
        cur_wav_.clear();
    }

    void AAsr::run() {
        int idx = 0;
        bool in_segment = false;
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
                // for asr detect: int16 量纲
                std::transform(data.begin(), data.end(), data.begin(),
                               [](const float x) { return x * 32768.0f; });

                if (trigger == "start") {
                    in_segment = true;
                    cur_wav_.clear();
                    if (streaming_) streaming_->reset();
                }

                if (in_segment) {
                    cur_wav_.insert(cur_wav_.end(), data.begin(), data.end());
                    // 流式实时部分结果
                    if (streaming_) {
                        streaming_->accept_waveform(data);
                        asr::StreamingAsrResult part;
                        streaming_->decode(false, &part);
                        if (on_asr_partial_ && !part.text.empty()) {
                            on_asr_partial_(streaming_->text());
                        }
                    }
                }

                if (trigger == "end") {
                    emit_final();
                    in_segment = false;
                }
            }
        }

        if (in_segment && !cur_wav_.empty()) {
            emit_final();
        }
        std::cout << "Asr run exit" << std::endl;
    }
}
