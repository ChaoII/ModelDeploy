//
// Created by aichao on 2025/5/19.
//

#pragma once


#include "csrc/base_model.h"

namespace modeldeploy::audio::asr {
    class MODELDEPLOY_CXX_EXPORT SenseVoice : public BaseModel {
    public:
        SenseVoice(const std::string& model_file,
                   const std::string& token_path_str,
                   const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "SenseVoice"; }

        bool predict(const std::vector<float>& data, std::string* result);

    protected:
        bool initialize();

        bool preprocess(const std::vector<float>& data, std::vector<Tensor>* outputs);

        bool postprocess(std::vector<Tensor>& infer_result, std::string* result);

    private:
        int32_t window_size_{};
        int32_t window_shift_{};
        int32_t with_itn_{};
        int32_t without_itn_{};

        std::string token_path_str_;
        std::vector<float> neg_mean_;
        std::vector<float> inv_stddev_;
        std::map<std::string, int32_t> lang_id_;
        std::map<std::string, std::string> tokens_;
    };
} // namespace detection
