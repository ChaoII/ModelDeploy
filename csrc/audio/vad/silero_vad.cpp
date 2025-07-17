//
// Created by aichao on 2025/5/19.
//

#include "audio/vad/silero_vad.h"
#include <fstream>
#include <utils/utils.h>
#include <kaldi-native-fbank/csrc/online-feature.h>

namespace modeldeploy::audio::vad {
    SileroVAD::SileroVAD(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool SileroVAD::initialize() {
        if (!init_runtime()) {
            std::cerr << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        auto Sample_rate = SampleRate::SR_16K;
        auto window_frame_ms = FrameMS::WS_32;
        const auto& min_silence_duration_ms = std::chrono::milliseconds(200);
        const auto& speech_pad_ms = std::chrono::milliseconds(30);
        const auto& min_speech_duration_ms = std::chrono::milliseconds(64);
        const auto& max_speech_duration_s = std::chrono::seconds(std::numeric_limits<int64_t>::max());

        threshold_ = 0.65f;
        sample_rate_ = static_cast<uint32_t>(Sample_rate);
        const uint32_t sr_per_ms = sample_rate_ / 1000;
        window_size_samples_ = static_cast<uint32_t>(window_frame_ms) * sr_per_ms;
        min_speech_samples_ = sr_per_ms * min_speech_duration_ms.count();
        speech_pad_samples_ = sr_per_ms * speech_pad_ms.count();
        min_silence_samples_ = sr_per_ms * min_silence_duration_ms.count();

        input_.resize(window_size_samples_);
        input_node_shape_ = {1, window_size_samples_};

        h_.resize(size_hc_);
        c_.resize(size_hc_);
        sr_.resize(1);
        sr_[0] = sample_rate_;
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
        if (data.empty()) {
            MD_LOG_ERROR << "The input data is empty." << std::endl;
            return false;
        }
        input_.assign(data.begin(), data.end());
        outputs->resize(4); // input,sr,h,c
        //input
        (*outputs)[0] = std::move(Tensor(input_.data(), input_node_shape_, DataType::FP32));
        //sr
        (*outputs)[1] = std::move(Tensor(sr_.data(), sr_node_shape_, DataType::INT64));
        //h
        (*outputs)[2] = std::move(Tensor(h_.data(), hc_node_shape_, DataType::FP32));
        //c
        (*outputs)[3] = std::move(Tensor(c_.data(), hc_node_shape_, DataType::FP32));
        return true;
    }

    bool SileroVAD::postprocess(std::vector<Tensor>& infer_result, std::string* result) {
        if (infer_result.size() != 3) {
            MD_LOG_ERROR << "The size of infer_result is not equal to 3." << std::endl;
            return false;
        }
        // output // hn// cn
        const float speech_prob = static_cast<float*>(infer_result[0].data())[0];
        const float* hn = static_cast<float*>(infer_result[1].data());
        std::memcpy(h_.data(), hn, size_hc_ * sizeof(float));
        const float* cn = static_cast<float*>(infer_result[2].data());
        std::memcpy(c_.data(), cn, size_hc_ * sizeof(float));
        // Push forward sample index
        current_sample_ += window_size_samples_;
        if (speech_prob >= threshold_) {
            temp_end_ = 0;
        }
        if (speech_prob >= threshold_ && triggered_ == false) {
            triggered_ = true;
            //int speech_start = current_sample - speech_pad_samples - window_size_samples;
            *result = "start";
            return true;
        }
        if (speech_prob < threshold_ - 0.15 && triggered_) {
            if (temp_end_ == 0) {
                temp_end_ = current_sample_;
            }
            if (current_sample_ - temp_end_ < min_silence_samples_) {
                *result = "none";
                return true;
            }
            temp_end_ = 0;
            triggered_ = false;
            *result = "end";
            return true;
        }
        *result = "none";
        return true;
    }
}
