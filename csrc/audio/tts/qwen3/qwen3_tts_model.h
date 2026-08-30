// csrc/audio/tts/qwen3/qwen3_tts_model.h
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>  // NOLINT
#include "runtime/runtime_option.h"

namespace modeldeploy::audio::tts {

// 主 talker 自回归循环的 KV-cache 状态（28 层 key/value）。
struct Qwen3TalkerState {
    std::vector<Ort::Value> kv_cache;
};

// 从 config.json 读取/默认的模型配置。
struct Qwen3TtsConfig {
    int32_t num_code_groups = 16;
    int32_t hidden_size = 1024;
    int32_t talker_vocab_size = 3072;
    int32_t code_predictor_vocab_size = 2048;
    int32_t num_hidden_layers = 28;
    int32_t text_vocab_size = 151936;

    int64_t tts_bos_token_id = 151672;
    int64_t tts_eos_token_id = 151673;
    int64_t tts_pad_token_id = 151671;

    int64_t codec_bos_id = 2149;
    int64_t codec_eos_token_id = 2150;
    int64_t codec_pad_id = 2148;
    int64_t codec_nothink_id = 2155;
    int64_t codec_think_id = 2154;
    int64_t codec_think_bos_id = 2156;
    int64_t codec_think_eos_id = 2157;

    // codec_language_id 映射（克隆模式的 lang 白名单来源）
    std::map<std::string, int64_t> codec_language_id;
    bool clone_supported = false;  // speaker_encoder + tokenizer12hz_encode 是否就绪
};

// Qwen3-TTS 12Hz 多子模型封装：9 个 ORT session。
// onnx 全部位于 {model_dir}/onnx_kv_06b/，tokenizer 位于 {model_dir}/models/...。
class Qwen3TtsModel {
public:
    Qwen3TtsModel();
    ~Qwen3TtsModel();

    Qwen3TtsModel(const Qwen3TtsModel&) = delete;
    Qwen3TtsModel& operator=(const Qwen3TtsModel&) = delete;

    bool Load(const std::string& model_dir, const RuntimeOption& opt);

    [[nodiscard]] bool loaded() const;

    // Run 接口（形状/输入输出名与本地 onnx 一一对应）
    Ort::Value RunTextProject(Ort::Value input_ids) const;
    Ort::Value RunCodecEmbed(Ort::Value input_ids) const;
    Ort::Value RunCodePredictorEmbed(Ort::Value input_ids,
                                     Ort::Value generation_step) const;
    Ort::Value RunCodePredictor(Ort::Value inputs_embeds,
                                Ort::Value generation_step) const;

    struct TalkerPrefillResult {
        Ort::Value logits;
        Ort::Value last_hidden;
        Qwen3TalkerState state;
    };
    TalkerPrefillResult RunTalkerPrefill(Ort::Value inputs_embeds,
                                         Ort::Value attention_mask) const;

    struct TalkerDecodeResult {
        Ort::Value logits;
        Ort::Value last_hidden;
        Qwen3TalkerState state;
    };
    TalkerDecodeResult RunTalkerDecode(Ort::Value inputs_embeds,
                                       Ort::Value attention_mask,
                                       Qwen3TalkerState state) const;

    Ort::Value RunSpeakerEncoder(Ort::Value mels) const;

    struct Tokenizer12hzEncodeResult {
        Ort::Value audio_codes;
        Ort::Value lengths;
    };
    Tokenizer12hzEncodeResult RunTokenizer12hzEncode(
        Ort::Value input_values, Ort::Value padding_mask) const;

    struct Tokenizer12hzDecodeResult {
        Ort::Value audio_values;
        Ort::Value lengths;
    };
    Tokenizer12hzDecodeResult RunTokenizer12hzDecode(
        Ort::Value audio_codes) const;

    [[nodiscard]] bool HasTokenizer12hzDecodeStream() const;

    [[nodiscard]] const Qwen3TtsConfig& GetConfig() const;
    [[nodiscard]] const std::string& tokenizer_dir() const;
    // 生成前确保 AR 相关 session 就绪；解码前释放大 session 腾内存。
    bool EnsureGenerationModels();
    void ReleaseGenerationModels();
    OrtAllocator* Allocator() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// 计算 speaker_encoder 所需 mel [frames, 128]（log 压缩，slaney 滤波组）。
// 输入为 24kHz 单声道 float 音频（[-1,1]）。成功返回 true。
bool ComputeSpeakerMel(const std::vector<float>& audio_24k,
                       std::vector<float>* mel_out, int* frames_out);

}  // namespace modeldeploy::audio::tts
