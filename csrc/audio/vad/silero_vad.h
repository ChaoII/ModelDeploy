//
// Created by aichao on 2025/5/19.
//

#pragma once


#include "csrc/base_model.h"

namespace modeldeploy {
    class MODELDEPLOY_CXX_EXPORT SileroVAD : public BaseModel {
    public:
        enum class SampleRate: uint32_t {
            SR_16K = 16000,
            SR_8K = 8000
        };

        enum class FrameMS : uint32_t {
            WS_32 = 32,
            WS_64 = 64,
            WS_96 = 96
        };


        explicit SileroVAD(const std::string& model_file,
                           const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "SileroVAD"; }

        virtual bool predict(const std::vector<float>& data, std::string* result);

    protected:
        bool initialize();

        bool preprocess(const std::vector<float>& data, std::vector<Tensor>* outputs);

        bool postprocess(std::vector<Tensor>& infer_result, std::string* result);

        // model config
        uint32_t window_size_samples{}; // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
        uint32_t sample_rate{}; //Assign when init support 16000 or 8000
        float threshold{};
        uint32_t min_silence_samples{}; // sr_per_ms * #ms
        uint32_t min_speech_samples{}; // sr_per_ms * #ms
        uint32_t speech_pad_samples{}; // usually a
        uint32_t audio_length_samples{};

        // model states
        bool triggered = false;
        unsigned int temp_end = 0;
        unsigned int current_sample = 0;
        // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes
        int prev_end{};
        int next_start = 0;

        std::vector<float> input;
        std::vector<int64_t> sr;
        unsigned int size_hc = 2 * 1 * 64; // It's FIXED.
        std::vector<float> _h;
        std::vector<float> _c;

        std::vector<int64_t> input_node_shape_ = {};
        std::vector<int64_t> sr_node_shape_ = {1};
        std::vector<int64_t> hc_node_shape_ = {2, 1, 64};
    };
} // namespace detection
