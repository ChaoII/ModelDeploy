//
// Created by aichao on 2025/5/19.
//

#include "csrc/audio/vad/silero_vad.h"
#include <fstream>
#include <csrc/utils/utils.h>
#include <kaldi-native-fbank/csrc/online-feature.h>

namespace modeldeploy {
    SileroVAD::SileroVAD(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = initialize();
    }

    bool SileroVAD::initialize() {
        if (!init_runtime()) {
            std::cerr << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        auto Sample_rate = SampleRate::SR_16K;
        auto window_frame_ms = FrameMS::WS_32;
        const auto& min_silence_duration_ms = std::chrono::milliseconds(200);
        const auto& speech_pad_ms = std::chrono::milliseconds(30);
        const auto& min_speech_duration_ms = std::chrono::milliseconds(64);
        const auto& max_speech_duration_s = std::chrono::seconds(std::numeric_limits<int64_t>::max());

        threshold = 0.65f;
        sample_rate = static_cast<uint32_t>(Sample_rate);
        const uint32_t sr_per_ms = sample_rate / 1000;
        window_size_samples = static_cast<uint32_t>(window_frame_ms) * sr_per_ms;
        min_speech_samples = sr_per_ms * min_speech_duration_ms.count();
        speech_pad_samples = sr_per_ms * speech_pad_ms.count();
        min_silence_samples = sr_per_ms * min_silence_duration_ms.count();

        input.resize(window_size_samples);
        input_node_shape_ = {1, window_size_samples};

        _h.resize(size_hc);
        _c.resize(size_hc);
        sr.resize(1);
        sr[0] = sample_rate;
        return true;
    }


    bool SileroVAD::predict(const std::vector<float>& data, std::string* result) {
        if (!preprocess(data, &reused_input_tensors_)) {
            std::cerr << "Failed to preprocess the input image." << std::endl;
            return false;
        }
        for (int i = 0; i < reused_input_tensors_.size(); i++) {
            reused_input_tensors_[i].set_name(get_input_info(i).name);
        }
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            std::cerr << "Failed to inference by runtime." << std::endl;
            return false;
        }
        if (!postprocess(reused_output_tensors_, result)) {
            std::cerr << "Failed to postprocess the inference results by runtime."
                << std::endl;
            return false;
        }
        return true;
    }


    bool SileroVAD::preprocess(const std::vector<float>& data, std::vector<Tensor>* outputs) {
        input.assign(data.begin(), data.end());
        outputs->resize(4); // input,sr,h,c
        //input
        (*outputs)[0] = std::move(Tensor(input.data(), input_node_shape_, DataType::FP32));
        //sr
        (*outputs)[1] = std::move(Tensor(sr.data(), sr_node_shape_, DataType::INT64));
        //h
        (*outputs)[2] = std::move(Tensor(_h.data(), hc_node_shape_, DataType::FP32));
        //c
        (*outputs)[3] = std::move(Tensor(_c.data(), hc_node_shape_, DataType::FP32));
    }

    bool SileroVAD::postprocess(std::vector<Tensor>& infer_result, std::string* result) {
        // output
        // hn
        // cn
        const float speech_prob = static_cast<float*>(infer_result[0].data())[0];
        const float* hn = static_cast<float*>(infer_result[1].data());
        std::memcpy(_h.data(), hn, size_hc * sizeof(float));
        const float* cn = static_cast<float*>(infer_result[2].data());
        std::memcpy(_c.data(), cn, size_hc * sizeof(float));
        // Push forward sample index
        current_sample += window_size_samples;
        if (speech_prob >= threshold) {
            temp_end = 0;
        }
        if (speech_prob >= threshold && triggered == false) {
            triggered = true;
            //int speech_start = current_sample - speech_pad_samples - window_size_samples;
            *result = "start";
            return true;
        }
        if (speech_prob < threshold - 0.15 && triggered) {
            if (temp_end == 0) {
                temp_end = current_sample;
            }
            if (current_sample - temp_end < min_silence_samples) {
                *result = "none";
                return true;
            }
            else {
                temp_end = 0;
                triggered = false;
                *result = "end";
                return true;
            }
        }
        *result = "none";
        return true;
    }
}
