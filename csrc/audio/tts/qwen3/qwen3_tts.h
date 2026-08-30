// csrc/audio/tts/qwen3/qwen3_tts.h
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "audio/tts/qwen3/qwen3_tts_model.h"
#include "audio/tts/qwen3/qwen3_tts_tokenizer.h"
#include "audio/tts/tts_model.h"
#include "runtime/runtime_option.h"

namespace modeldeploy::audio::tts {

// Qwen3-TTS-Tokenizer-12Hz 0.6B（多子模型 LLM 管线），24kHz 输出。
// 派生自 ITtsModel（chunk_frames==0 表示一次性合成）。
class Qwen3Tts : public ITtsModel {
public:
    Qwen3Tts() = default;
    Qwen3Tts(const std::string& model_dir, const RuntimeOption& opt);

    bool init(const std::string& model_dir, const RuntimeOption& opt);

    [[nodiscard]] std::string name() const override { return "Qwen3Tts"; }

    bool predict(const std::string& text, const std::string& voice, float speed,
                 std::vector<float>* out) override;
    bool predict_stream(const std::string& text, const std::string& voice,
                        float speed, int chunk_frames,
                        const std::function<bool(const float*, int, float)>& cb) override;

    // 声音克隆（Base 模型）：ref_audio 提说话人 embedding + 参考音频编码 + ICL prompt。
    // lang 为 codec_language_id 白名单（含 "auto"）；非法返回 false。
    bool clone(const std::string& text, const std::string& ref_audio,
               const std::string& ref_text, const std::string& lang,
               std::vector<float>* out);

    [[nodiscard]] int32_t get_sample_rate() const override { return 24000; }

    // 预置说话人（CustomVoice 模型名单，Base 模型仅用于默认名解析）
    [[nodiscard]] std::vector<std::string> get_supported_speakers() const;
    // 语言白名单（clone 的 lang 校验）
    [[nodiscard]] std::vector<std::string> get_supported_languages() const;

private:
    struct GenConfig {
        float temperature = 0.9f;
        int32_t top_k = 50;
        float top_p = 1.0f;
        float rep_penalty = 1.05f;
        int32_t max_new_tokens = 2048;
        float sub_temperature = 0.9f;
        int32_t sub_top_k = 50;
        float sub_top_p = 1.0f;
        int32_t chunk_frames = 0;
    };

    struct ClonePrompt {
        bool enabled = false;
        std::vector<std::vector<int64_t>> ref_codes;  // [ref_len][16]
        std::vector<float> ref_spk_embed;             // [hidden_size]
        std::vector<int64_t> ref_ids;                 // 参考文本 body ids
        int64_t language_id = -1;                     // -1 = auto
    };

    // 统一生成入口：一次性（cb==nullptr）或流式（cb!=nullptr，无 decode_stream → 退化）。
    // all_codes 为输出表单帧码（clone 用）或 nullptr。
    bool GenerateTalker(const std::vector<int64_t>& input_ids,
                        const GenConfig& gc, const ClonePrompt* clone_prompt,
                        const std::function<bool(const float*, int, float)>& cb,
                        std::vector<float>* audio,
                        std::vector<std::vector<int64_t>>* all_codes);

    Ort::Value RunTextProjectHelper(const std::vector<int64_t>& ids) const;
    Ort::Value RunCodecEmbedHelper(const std::vector<int64_t>& ids) const;

    int64_t SampleFromLogits(const Ort::Value& logits_tensor, int32_t vocab_size,
                             float temperature, int32_t top_k, float top_p,
                             float repetition_penalty,
                             const std::vector<int64_t>& generated,
                             int64_t suppress_start, int64_t suppress_end,
                             int64_t suppress_exception,
                             bool suppress_eos) const;

    std::vector<float> DecodeFrames(
        const std::vector<std::vector<int64_t>>& codes) const;

    // 参考音频 → [ref_len][16] 码（tokenizer12hz_encode + lengths 截断）
    bool EncodeRefAudio(const std::vector<float>& audio_24k,
                        std::vector<std::vector<int64_t>>* codes) const;

    // 参考音频 → 说话人 embedding（mel + speaker_encoder）
    bool ExtractSpeakerEmbedding(const std::vector<float>& audio_24k,
                                 std::vector<float>* spk) const;

    // 加载并归一化 WAV（支持 16/24/32 位整数与 float；转单声道、重采样到 24k）
    bool LoadRefAudio(const std::string& ref_audio,
                      std::vector<float>* audio_24k) const;

    Qwen3TtsModel model_;
    Qwen3TtsTokenizer tokenizer_;
    bool tokenizer_ok_ = false;
    bool init_ok_ = false;
};

}  // namespace modeldeploy::audio::tts
