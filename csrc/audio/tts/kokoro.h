//
// Created by aichao on 2025/5/20.
//

#pragma once

#include "csrc/base_model.h"
#include "cppjieba/Jieba.hpp"
#include "csrc/audio/text_normalize/text_normalization.h"


namespace modeldeploy::audio::tts
{
    class MODELDEPLOY_CXX_EXPORT Kokoro : public BaseModel {
    public:
        Kokoro(const std::string& model_file_path, const std::string& token_path_str,
               const std::vector<std::string>& lexicons, const std::string& voices_bin,
               const std::string& jieba_dir,
               const std::string& text_normalization_dir,
               const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "Kokoro"; }

        virtual bool predict(const std::string& text, const std::string& voice, float speed,
                             std::vector<float>* out_audio);

        [[nodiscard]] int32_t get_sample_rate() const { return sample_rate_; }

        void set_sample_rate(const int32_t sample_rate) { sample_rate_ = sample_rate; }

    protected:
        bool initialize();

        bool preprocess(const std::string& text, const std::string& voice, float speed, std::vector<Tensor>* outputs);

        bool postprocess(std::vector<Tensor>& infer_result, std::vector<float>* out_audio);

    private:
        void load_tokens(const std::string&);
        void load_lexicons(const std::vector<std::string>&);
        bool load_voices(const std::vector<std::string>& speaker_names,
                         std::vector<int64_t>& dims, const std::string& voices_bin);
        [[nodiscard]] std::vector<std::string> split_ch_eng(const std::string& text) const;

        std::unique_ptr<cppjieba::Jieba> jieba_;
        std::set<char> punc_set_;
        int32_t sample_rate_{};
        int32_t max_len_{};
        std::string token_path_str_;
        std::vector<std::string> lexicons_;
        std::string voices_bin_;
        std::string jieba_dir_;
        std::string text_normalization_dir_;
        std::map<std::string, int32_t> token2id_;
        std::map<std::string, std::vector<std::string>> word2token_;
        std::map<std::string, std::vector<float>> voices_; // voice -> 510 x 1 x 256
        std::vector<int64_t> style_dims_;
        std::unique_ptr<TextNormalizer> text_normalizer_;
    };
} // namespace detection
