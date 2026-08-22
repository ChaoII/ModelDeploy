#pragma once

#include "csrc/base_model.h"

namespace modeldeploy::audio::speaker_verify {
    /*! @brief ECAPA-TDNN speaker embedding model (SOTA + lightweight).
     *  输入 float PCM(16k)，输出 192-d 说话人 embedding。
     */
    class MODELDEPLOY_CXX_EXPORT SpeakerVerify : public BaseModel {
    public:
        SpeakerVerify(const std::string& model_file,
                      const RuntimeOption& custom_option = RuntimeOption());
        [[nodiscard]] std::string name() const override { return "SpeakerVerify"; }
        // 输入 float PCM(16k)；输出 embedding（len 由模型决定，默认 192）
        bool predict(const std::vector<float>& data, std::vector<float>* embedding);
        [[nodiscard]] bool is_initialized() const;
        [[nodiscard]] std::unique_ptr<SpeakerVerify> clone() const;
    protected:
        bool initialize();
        bool preprocess(const std::vector<float>& data, std::vector<Tensor>* outputs);
        bool postprocess(std::vector<Tensor>& infer_result, std::vector<float>* embedding);
    private:
        explicit SpeakerVerify() = default;   // 供 clone()
        int32_t mel_bins_{80};
        int32_t embedding_dim_{-1};
        std::vector<float> window_;  // 备用（无需时保持空）
    };
} // namespace modeldeploy::audio::speaker_verify
