// csrc/audio/tts/audio8/audio8_manifest.h
#pragma once

#include <cstdint>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "core/md_log.h"

namespace modeldeploy::audio::tts::audio8 {

// Audio8-TTS-Preview-0.6B runtime_manifest.json 的 C++ 视图。
// 字段缺失/类型不匹配时回退默认值并输出 MD_LOG_WARN（与 brief 约定一致）。
// JSON 使用 vendored nlohmann/json（third_party/nlohmann/json.hpp）。
struct Audio8Manifest {
    std::string model_dir;
    std::string voices_dir;
    std::string tokenizer_path;

    std::string slow_logits_layout = "semantic_then_eos";
    int64_t slow_logits_size = 4097;
    int64_t max_seq_len = 2048;
    int64_t num_layers = 24;
    int64_t num_fast_layers = 4;
    int64_t num_codebooks = 10;
    int64_t n_local_heads = 2;
    int64_t fast_n_local_heads = 2;
    int64_t head_dim = 64;
    int64_t fast_head_dim = 64;
    int64_t fast_dim = 896;
    int64_t vocab_size = 155776;
    int64_t codebook_size = 4096;
    int64_t semantic_begin_id = 151678;
    int64_t semantic_end_id = 155773;
    int64_t eos_token_id = 151645;
    int64_t pad_token_id = 151643;
    int64_t im_end_id = 151645;
    int64_t codec_sample_rate = 44100;
    int64_t sample_rate = 44100;
    int64_t codec_frame_size = 2048;
    int64_t codec_hop_length = 2048;
    int64_t stream_context_frames = 128;
    int64_t stream_guard_frames = 1;
    std::string default_precision = "int4";
    std::string default_codec_precision = "fp16";
    std::string decoder_provider = "cpu";
    std::string slow_model;
    std::string fast_model;
    std::string codec_model = "codec_decoder_fp16.onnx";
    std::string model_fingerprint;

    static bool FromJson(const std::string& manifest_path, const std::string& model_dir,
                         Audio8Manifest* out);
};

inline bool Audio8Manifest::FromJson(const std::string& manifest_path,
                                     const std::string& model_dir, Audio8Manifest* out) {
    out->model_dir = model_dir;
    out->voices_dir = model_dir + "/voices";
    out->tokenizer_path = model_dir + "/tokenizer/tokenizer.json";

    try {
        std::ifstream ifs(manifest_path);
        if (!ifs.is_open()) {
            MD_LOG_WARN << "audio8: cannot open manifest " << manifest_path << std::endl;
            return false;
        }
        const nlohmann::json root = nlohmann::json::parse(ifs);
        if (!root.is_object()) {
            MD_LOG_WARN << "audio8: manifest is not an object" << std::endl;
            return false;
        }
        auto get_str = [&root](const char* key, std::string* field) {
            auto it = root.find(key);
            if (it != root.end() && it->is_string()) {
                *field = it->get<std::string>();
            } else {
                MD_LOG_WARN << "audio8: manifest missing '" << key << "', use default"
                            << std::endl;
            }
        };
        auto get_int = [&root](const char* key, int64_t* field) {
            auto it = root.find(key);
            if (it != root.end() && it->is_number_integer()) {
                *field = it->get<int64_t>();
            } else {
                MD_LOG_WARN << "audio8: manifest missing '" << key << "', use default"
                            << std::endl;
            }
        };

        get_str("slow_logits_layout", &out->slow_logits_layout);
        get_int("slow_logits_size", &out->slow_logits_size);
        get_int("max_seq_len", &out->max_seq_len);
        get_int("num_layers", &out->num_layers);
        get_int("num_fast_layers", &out->num_fast_layers);
        get_int("num_codebooks", &out->num_codebooks);
        get_int("n_local_heads", &out->n_local_heads);
        get_int("fast_n_local_heads", &out->fast_n_local_heads);
        get_int("head_dim", &out->head_dim);
        get_int("fast_head_dim", &out->fast_head_dim);
        get_int("fast_dim", &out->fast_dim);
        get_int("vocab_size", &out->vocab_size);
        get_int("codebook_size", &out->codebook_size);
        get_int("semantic_begin_id", &out->semantic_begin_id);
        get_int("semantic_end_id", &out->semantic_end_id);
        get_int("eos_token_id", &out->eos_token_id);
        get_int("pad_token_id", &out->pad_token_id);
        get_int("im_end_id", &out->im_end_id);
        get_int("codec_sample_rate", &out->codec_sample_rate);
        get_int("sample_rate", &out->sample_rate);
        get_int("codec_frame_size", &out->codec_frame_size);
        get_int("codec_hop_length", &out->codec_hop_length);
        get_int("stream_context_frames", &out->stream_context_frames);
        get_int("stream_guard_frames", &out->stream_guard_frames);
        get_str("default_precision", &out->default_precision);
        get_str("default_codec_precision", &out->default_codec_precision);
        get_str("decoder_provider", &out->decoder_provider);
        get_str("model_fingerprint", &out->model_fingerprint);

        const auto codec_models = root.find("codec_models");
        if (codec_models != root.end() && codec_models->is_object()) {
            auto it = codec_models->find(out->default_codec_precision);
            if (it != codec_models->end() && it->is_string()) {
                out->codec_model = it->get<std::string>();
            } else {
                MD_LOG_WARN << "audio8: codec_models missing precision '"
                            << out->default_codec_precision << "', use default" << std::endl;
            }
        } else {
            MD_LOG_WARN << "audio8: manifest missing 'codec_models', use default" << std::endl;
        }

        out->slow_model = "slow_ar_" + out->default_precision + ".onnx";
        out->fast_model = "fast_ar_" + out->default_precision + ".onnx";
    } catch (const std::exception& e) {
        MD_LOG_WARN << "audio8: parse manifest failed: " << e.what() << std::endl;
        return false;
    }

    MD_LOG_INFO << "Audio8Manifest: slow=" << out->slow_model << " fast=" << out->fast_model
                << " codec=" << out->codec_model << " sr=" << out->sample_rate
                << " max_seq=" << out->max_seq_len << " layers=" << out->num_layers
                << " fast_layers=" << out->num_fast_layers
                << " codebooks=" << out->num_codebooks << " layout=" << out->slow_logits_layout
                << std::endl;
    return true;
}

}  // namespace modeldeploy::audio::tts::audio8
