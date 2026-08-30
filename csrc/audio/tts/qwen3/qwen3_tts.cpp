// csrc/audio/tts/qwen3/qwen3_tts.cpp
#include "qwen3_tts.h"

#include <algorithm>
#include <array>
#include <chrono>  // NOLINT
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <utility>

#include "audio/tts/common/sampling.h"
#include "core/md_log.h"

namespace modeldeploy::audio::tts {
namespace {

constexpr int32_t kSamplesPerFrame = 1920;

int64_t ShapeNumel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
}

// 预置说话人（Qwen3-TTS 官方 CustomVoice 模型名单，见 README）
const std::vector<std::string>& kPresetSpeakers() {
    static const std::vector<std::string> spks = {
        "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
        "Ryan",   "Aiden",  "Ono_Anna", "Sohee"};
    return spks;
}

std::string Lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// 参考音频 → 单声道 float（[-1,1]），线性重采样到 24kHz
bool LoadWavTo24k(const std::string& path, std::vector<float>* out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        MD_LOG_ERROR << "cannot open " << path << std::endl;
        return false;
    }
    auto rd = [&](void* dst, size_t n) { ifs.read(reinterpret_cast<char*>(dst), n); };

    char riff[4];
    rd(riff, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0) {
        MD_LOG_ERROR << "not a RIFF file: " << path << std::endl;
        return false;
    }
    uint32_t riff_size = 0;
    rd(&riff_size, 4);
    char wave[4];
    rd(wave, 4);
    if (std::memcmp(wave, "WAVE", 4) != 0) {
        MD_LOG_ERROR << "not a WAVE file: " << path << std::endl;
        return false;
    }

    uint16_t audio_format = 0, num_channels = 0, bits = 0;
    uint32_t sample_rate = 0;
    bool have_fmt = false;
    while (ifs.good()) {
        char id[4];
        uint32_t sz = 0;
        rd(id, 4);
        rd(&sz, 4);
        if (!ifs.good()) break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            uint16_t fmt_code = 0;
            uint16_t ch = 0;
            uint32_t sr = 0;
            uint16_t bits_per = 0;
            rd(&fmt_code, 2);
            rd(&ch, 2);
            rd(&sr, 4);
            uint32_t byte_rate = 0;
            uint16_t block = 0;
            rd(&byte_rate, 4);
            rd(&block, 2);
            rd(&bits_per, 2);
            audio_format = fmt_code;
            num_channels = ch;
            sample_rate = sr;
            bits = bits_per;
            have_fmt = true;
            if (sz > 16) ifs.seekg(sz - 16, std::ios::cur);
        } else if (std::memcmp(id, "data", 4) == 0) {
            std::vector<uint8_t> raw(sz);
            rd(raw.data(), sz);
            if (!have_fmt) {
                MD_LOG_ERROR << "no fmt chunk before data: " << path << std::endl;
                return false;
            }
            if (num_channels == 0 || sample_rate == 0 || (bits != 16 && bits != 24 &&
                bits != 32 && bits != 8)) {
                MD_LOG_ERROR << "unsupported wav fmt: ch=" << num_channels
                             << " sr=" << sample_rate << " bits=" << bits
                             << std::endl;
                return false;
            }
            const size_t bytes_per = (bits + 7) / 8;
            const size_t n_frames = sz / (bytes_per * num_channels);
            std::vector<float> mono(n_frames);
            const bool is_float = (audio_format == 3) && (bits == 32);
            for (size_t f = 0; f < n_frames; ++f) {
                float sample = 0.0f;
                const uint8_t* p = raw.data() + f * bytes_per * num_channels;
                if (is_float) {
                    float v;
                    std::memcpy(&v, p, 4);
                    sample = v;
                } else if (bits == 16) {
                    int16_t v;
                    std::memcpy(&v, p, 2);
                    sample = static_cast<float>(v) / 32768.0f;
                } else if (bits == 24) {
                    int32_t v = (static_cast<int32_t>(p[0]) |
                                 (static_cast<int32_t>(p[1]) << 8) |
                                 (static_cast<int32_t>(p[2]) << 16));
                    if (v & 0x800000) v |= ~0xFFFFFF;  // 符号扩展
                    sample = static_cast<float>(v) / 8388608.0f;
                } else if (bits == 8) {
                    sample = (static_cast<float>(p[0]) - 128.0f) / 128.0f;
                }
                if (f == 0) {
                    mono[f] = sample;
                } else {
                    // 多声道取平均
                    float acc = sample;
                    for (uint16_t c = 1; c < num_channels; ++c) {
                        int16_t ch16 = 0;
                        if (is_float) {
                            float v;
                            std::memcpy(&v, p + c * 4, 4);
                            acc += v;
                        } else {
                            const uint8_t* pp = p + c * bytes_per;
                            if (bits == 16) {
                                std::memcpy(&ch16, pp, 2);
                                acc += static_cast<float>(ch16) / 32768.0f;
                            } else if (bits == 24) {
                                int32_t v = (static_cast<int32_t>(pp[0]) |
                                             (static_cast<int32_t>(pp[1]) << 8) |
                                             (static_cast<int32_t>(pp[2]) << 16));
                                if (v & 0x800000) v |= ~0xFFFFFF;
                                acc += static_cast<float>(v) / 8388608.0f;
                            } else {
                                acc += (static_cast<float>(pp[0]) - 128.0f) / 128.0f;
                            }
                        }
                    }
                    mono[f] = acc / static_cast<float>(num_channels);
                }
            }

            // 重采样到 24k（线性插值）
            if (sample_rate == 24000) {
                *out = std::move(mono);
            } else if (sample_rate > 0 && !mono.empty()) {
                const double ratio = static_cast<double>(24000) / sample_rate;
                const size_t out_n =
                    static_cast<size_t>(static_cast<double>(mono.size()) * ratio);
                std::vector<float> res(out_n);
                for (size_t i = 0; i < out_n; ++i) {
                    const double pos = static_cast<double>(i) / ratio;
                    const size_t i0 = static_cast<size_t>(pos);
                    const size_t i1 = std::min(i0 + 1, mono.size() - 1);
                    const double frac = pos - static_cast<double>(i0);
                    res[i] = static_cast<float>(mono[i0] * (1.0 - frac) +
                                                mono[i1] * frac);
                }
                *out = std::move(res);
            } else {
                MD_LOG_ERROR << "empty wav: " << path << std::endl;
                return false;
            }
            return true;
        } else {
            ifs.seekg(sz, std::ios::cur);  // 跳过其它 chunk
        }
    }
    MD_LOG_ERROR << "no data chunk in " << path << std::endl;
    return false;
}

std::string PathJoin(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    const char last = dir.back();
    if (last == '/' || last == '\\') return dir + name;
    return dir + "/" + name;
}

}  // namespace

Qwen3Tts::Qwen3Tts() = default;
Qwen3Tts::~Qwen3Tts() = default;

Qwen3Tts::Qwen3Tts(const std::string& model_dir, const RuntimeOption& opt) {
    init(model_dir, opt);
}

bool Qwen3Tts::init(const std::string& model_dir, const RuntimeOption& opt) {
    init_ok_ = false;
    if (model_dir.empty()) {
        MD_LOG_ERROR << "Qwen3Tts: empty model_dir" << std::endl;
        return false;
    }
    if (!model_.Load(model_dir, opt) || !model_.loaded()) {
        MD_LOG_ERROR << "Qwen3Tts: failed to load onnx models from " << model_dir
                     << std::endl;
        return false;
    }
    const std::string tdir = model_.tokenizer_dir();
    tokenizer_ = Qwen3TtsTokenizer(tdir);
    tokenizer_ok_ = tokenizer_.loaded();
    if (!tokenizer_ok_) {
        MD_LOG_ERROR << "Qwen3Tts: failed to load tokenizer from " << tdir
                     << std::endl;
        return false;
    }
    init_ok_ = true;
    return true;
}

std::vector<std::string> Qwen3Tts::get_supported_speakers() const {
    return kPresetSpeakers();
}

std::vector<std::string> Qwen3Tts::get_supported_languages() const {
    const auto& cfg = model_.GetConfig();
    std::vector<std::string> langs;
    for (const auto& kv : cfg.codec_language_id) langs.push_back(kv.first);
    langs.insert(langs.begin(), "auto");
    return langs;
}

bool Qwen3Tts::predict(const std::string& text, const std::string& voice,
                       float speed, std::vector<float>* out) {
    if (!init_ok_) {
        MD_LOG_ERROR << "Qwen3Tts not initialized" << std::endl;
        return false;
    }
    (void)voice;  // Base 模型无预置说话人，仅解析默认名
    (void)speed;  // 官方/impl 无 speed 支持：预留，返回一致结果

    if (out == nullptr || text.empty()) {
        MD_LOG_ERROR << "Qwen3Tts::predict invalid args" << std::endl;
        return false;
    }

    const std::string formatted = "<|im_start|>assistant\n" + text +
                                  "<|im_end|>\n<|im_start|>assistant\n";
    const auto input_ids = tokenizer_.Encode(formatted);
    if (input_ids.size() < 9) {
        MD_LOG_ERROR << "Qwen3Tts tokenized too short (" << input_ids.size()
                     << ")" << std::endl;
        return false;
    }
    if (input_ids.size() > 2048) {
        MD_LOG_ERROR << "Qwen3Tts input too long: " << input_ids.size()
                     << " tokens > 2048, refusing to truncate" << std::endl;
        return false;
    }

    GenConfig gc;
    return GenerateTalker(input_ids, gc, nullptr, nullptr, out, nullptr);
}

bool Qwen3Tts::predict_stream(
    const std::string& text, const std::string& voice, float speed,
    int chunk_frames,
    const std::function<bool(const float*, int, float)>& cb) {
    if (!init_ok_ || !cb) return false;
    (void)voice;
    (void)speed;
    if (chunk_frames <= 0) {
        MD_LOG_ERROR << "Qwen3Tts::predict_stream requires chunk_frames > 0"
                     << std::endl;
        return false;
    }
    const std::string formatted = "<|im_start|>assistant\n" + text +
                                  "<|im_end|>\n<|im_start|>assistant\n";
    const auto input_ids = tokenizer_.Encode(formatted);
    if (input_ids.size() < 9 || input_ids.size() > 2048) {
        MD_LOG_ERROR << "Qwen3Tts::predict_stream input invalid/long: "
                     << input_ids.size() << std::endl;
        return false;
    }
    GenConfig gc;
    gc.chunk_frames = chunk_frames;
    std::vector<float> audio;
    const bool ok = GenerateTalker(input_ids, gc, nullptr, cb, &audio, nullptr);
    return ok;
}

bool Qwen3Tts::clone(const std::string& text, const std::string& ref_audio,
                     const std::string& ref_text, const std::string& lang,
                     std::vector<float>* out) {
    if (!init_ok_) {
        MD_LOG_ERROR << "Qwen3Tts not initialized" << std::endl;
        return false;
    }
    if (out == nullptr || text.empty() || ref_audio.empty() || ref_text.empty()) {
        MD_LOG_ERROR << "Qwen3Tts::clone invalid args" << std::endl;
        return false;
    }
    if (!model_.GetConfig().clone_supported) {
        MD_LOG_ERROR << "Qwen3Tts::clone unsupported (missing speaker_encoder/"
                        "tokenizer12hz_encode)"
                     << std::endl;
        return false;
    }

    // lang 白名单校验
    const std::string lang_l = Lower(lang);
    const auto& cfg = model_.GetConfig();
    int64_t language_id = -1;
    if (lang_l != "auto" && !lang_l.empty()) {
        auto it = cfg.codec_language_id.find(lang_l);
        if (it == cfg.codec_language_id.end()) {
            MD_LOG_ERROR << "Qwen3Tts::clone unsupported lang '" << lang << "'"
                         << std::endl;
            return false;
        }
        language_id = it->second;
    }

    // 参考音频
    std::vector<float> audio_24k;
    if (!LoadRefAudio(ref_audio, &audio_24k) || audio_24k.empty()) {
        MD_LOG_ERROR << "Qwen3Tts::clone failed to load ref audio" << std::endl;
        return false;
    }
    if (audio_24k.size() < static_cast<size_t>(kSamplesPerFrame)) {
        MD_LOG_ERROR << "Qwen3Tts::clone ref audio too short" << std::endl;
        return false;
    }

    ClonePrompt cp;
    cp.enabled = true;
    cp.language_id = language_id;
    if (!EncodeRefAudio(audio_24k, &cp.ref_codes) || cp.ref_codes.empty()) {
        MD_LOG_ERROR << "Qwen3Tts::clone failed to encode ref audio" << std::endl;
        return false;
    }
    if (!ExtractSpeakerEmbedding(audio_24k, &cp.ref_spk_embed) ||
        cp.ref_spk_embed.empty()) {
        MD_LOG_ERROR << "Qwen3Tts::clone failed to extract speaker embedding"
                     << std::endl;
        return false;
    }

    // 参考文本 token（<|im_start|>assistant\n{ref}<|im_end|>\n → [3:-2]）
    const std::string ref_formatted =
        "<|im_start|>assistant\n" + ref_text + "<|im_end|>\n";
    const auto ref_ids_all = tokenizer_.Encode(ref_formatted);
    if (ref_ids_all.size() < 5) {
        MD_LOG_ERROR << "Qwen3Tts::clone ref_text tokenized too short"
                     << std::endl;
        return false;
    }
    cp.ref_ids.assign(ref_ids_all.begin() + 3, ref_ids_all.end() - 2);

    const std::string formatted = "<|im_start|>assistant\n" + text +
                                  "<|im_end|>\n<|im_start|>assistant\n";
    const auto input_ids = tokenizer_.Encode(formatted);
    if (input_ids.size() < 9 || input_ids.size() > 2048) {
        MD_LOG_ERROR << "Qwen3Tts::clone input invalid/long: " << input_ids.size()
                     << std::endl;
        return false;
    }

    GenConfig gc;
    std::vector<std::vector<int64_t>> all_codes;
    const bool ok =
        GenerateTalker(input_ids, gc, &cp, nullptr, out, &all_codes);
    return ok;
}

// =====================================================================
// 生成主流程（移植自 sherpa-onnx offline-tts-qwen3-impl.cc）
// =====================================================================

Ort::Value Qwen3Tts::RunTextProjectHelper(const std::vector<int64_t>& ids) const {
    auto alloc = model_.Allocator();
    std::array<int64_t, 2> shape = {1, static_cast<int64_t>(ids.size())};
    Ort::Value input = Ort::Value::CreateTensor(
        alloc, shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    std::copy(ids.begin(), ids.end(), input.GetTensorMutableData<int64_t>());
    return model_.RunTextProject(std::move(input));
}

Ort::Value Qwen3Tts::RunCodecEmbedHelper(const std::vector<int64_t>& ids) const {
    auto alloc = model_.Allocator();
    std::array<int64_t, 2> shape = {1, static_cast<int64_t>(ids.size())};
    Ort::Value input = Ort::Value::CreateTensor(
        alloc, shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    std::copy(ids.begin(), ids.end(), input.GetTensorMutableData<int64_t>());
    return model_.RunCodecEmbed(std::move(input));
}

int64_t Qwen3Tts::SampleFromLogits(
    const Ort::Value& logits_tensor, int32_t vocab_size, float temperature,
    int32_t top_k, float top_p, float repetition_penalty,
    const std::vector<int64_t>& generated, int64_t suppress_start,
    int64_t suppress_end, int64_t suppress_exception, bool suppress_eos) const {
    const auto shape = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const float* logits_data = logits_tensor.GetTensorData<float>();
    const int64_t total = ShapeNumel(shape);
    const int32_t V = total >= vocab_size ? vocab_size : static_cast<int32_t>(total);
    const float* src = logits_data + (total - V);

    thread_local std::mt19937 rng(std::random_device{}());
    return sampling::SampleFromLogits(src, V, temperature, top_k, top_p,
                                      repetition_penalty, generated,
                                      suppress_start, suppress_end,
                                      suppress_exception, suppress_eos, &rng);
}

std::vector<float> Qwen3Tts::DecodeFrames(
    const std::vector<std::vector<int64_t>>& codes) const {
    if (codes.empty()) return {};
    const auto& cfg = model_.GetConfig();
    const int32_t num_frames = static_cast<int32_t>(codes.size());
    const int32_t num_groups = cfg.num_code_groups;
    auto alloc = model_.Allocator();

    std::array<int64_t, 3> shape = {1, num_frames, num_groups};
    Ort::Value audio_codes = Ort::Value::CreateTensor(
        alloc, shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto* dst = audio_codes.GetTensorMutableData<int64_t>();
    for (int32_t t = 0; t < num_frames; ++t)
        for (int32_t g = 0; g < num_groups; ++g)
            dst[t * num_groups + g] = codes[t][g];

    auto result = model_.RunTokenizer12hzDecode(std::move(audio_codes));
    const float* audio_data = result.audio_values.GetTensorData<float>();
    const auto audio_shape =
        result.audio_values.GetTensorTypeAndShapeInfo().GetShape();
    const int64_t total = ShapeNumel(audio_shape);
    int64_t valid = total;
    if (result.lengths.IsTensor()) {
        const auto* len = result.lengths.GetTensorData<int64_t>();
        valid = len[0];
        if (valid > total) valid = total;
    }
    if (valid < 0) valid = 0;
    return std::vector<float>(audio_data, audio_data + valid);
}

bool Qwen3Tts::EncodeRefAudio(const std::vector<float>& audio_24k,
                              std::vector<std::vector<int64_t>>* codes) const {
    auto alloc = model_.Allocator();
    const int64_t n = static_cast<int64_t>(audio_24k.size());
    std::array<int64_t, 2> shape = {1, n};
    Ort::Value input_values = Ort::Value::CreateTensor(
        alloc, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::copy(audio_24k.begin(), audio_24k.end(),
              input_values.GetTensorMutableData<float>());
    Ort::Value padding_mask = Ort::Value::CreateTensor(
        alloc, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    std::fill(padding_mask.GetTensorMutableData<int64_t>(),
              padding_mask.GetTensorMutableData<int64_t>() + n, 1LL);

    auto res = model_.RunTokenizer12hzEncode(std::move(input_values),
                                             std::move(padding_mask));
    const auto codes_shape = res.audio_codes.GetTensorTypeAndShapeInfo().GetShape();
    if (codes_shape.size() != 3) {
        MD_LOG_ERROR << "unexpected audio_codes rank" << std::endl;
        return false;
    }
    const int64_t frames_avail = codes_shape[1];
    const int64_t groups = codes_shape[2];
    const auto* code_data = res.audio_codes.GetTensorData<int64_t>();
    int64_t len = frames_avail;
    if (res.lengths.IsTensor()) {
        const auto* l = res.lengths.GetTensorData<int64_t>();
        len = l[0];
        if (len > frames_avail) len = frames_avail;
    }
    codes->clear();
    for (int64_t t = 0; t < len && t < frames_avail; ++t) {
        std::vector<int64_t> row(groups);
        for (int64_t g = 0; g < groups; ++g)
            row[g] = code_data[t * groups + g];
        codes->push_back(std::move(row));
    }
    return !codes->empty();
}

bool Qwen3Tts::ExtractSpeakerEmbedding(const std::vector<float>& audio_24k,
                                       std::vector<float>* spk) const {
    std::vector<float> mel;
    int frames = 0;
    if (!ComputeSpeakerMel(audio_24k, &mel, &frames) || frames <= 0) return false;

    auto alloc = model_.Allocator();
    std::array<int64_t, 3> shape = {1, frames, 128};
    Ort::Value mels = Ort::Value::CreateTensor(
        alloc, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::copy(mel.begin(), mel.end(), mels.GetTensorMutableData<float>());

    auto out = model_.RunSpeakerEncoder(std::move(mels));
    const auto os = out.GetTensorTypeAndShapeInfo().GetShape();
    const int64_t total = ShapeNumel(os);
    const float* data = out.GetTensorData<float>();
    spk->assign(data, data + total);
    return !spk->empty();
}

bool Qwen3Tts::LoadRefAudio(const std::string& ref_audio,
                            std::vector<float>* audio_24k) const {
    return LoadWavTo24k(ref_audio, audio_24k);
}

bool Qwen3Tts::GenerateTalker(
    const std::vector<int64_t>& input_ids, const GenConfig& gc,
    const ClonePrompt* clone_prompt,
    const std::function<bool(const float*, int, float)>& cb,
    std::vector<float>* audio, std::vector<std::vector<int64_t>>* all_codes) {
    const auto& cfg = model_.GetConfig();
    const int32_t num_code_groups = cfg.num_code_groups;
    const int32_t D = cfg.hidden_size;
    auto alloc = model_.Allocator();

    // 生成前确保 AR 相关 session 已加载（可能被上次生成释放）
    if (!model_.EnsureGenerationModels()) {
        MD_LOG_ERROR << "Qwen3Tts: failed to (re)load generation models"
                     << std::endl;
        return false;
    }

    const int32_t chunk_frames = gc.chunk_frames;
    const bool streaming = (chunk_frames > 0) && (cb != nullptr);

    // ---- 预构建 embedding ----
    const std::vector<int64_t> special_ids = {cfg.tts_bos_token_id,
                                              cfg.tts_eos_token_id,
                                              cfg.tts_pad_token_id};
    auto special_embed = RunTextProjectHelper(special_ids);  // [1,3,D]

    const std::vector<int64_t> role_ids(input_ids.begin(), input_ids.begin() + 3);
    auto role_embed = RunTextProjectHelper(role_ids);  // [1,3,D]

    // codec prefix（clone 时可带语言 id）
    std::vector<int64_t> codec_prefix_ids;
    const int64_t language_id =
        clone_prompt ? clone_prompt->language_id : -1;
    if (language_id < 0) {
        codec_prefix_ids = {cfg.codec_nothink_id, cfg.codec_think_bos_id,
                            cfg.codec_think_eos_id};
    } else {
        codec_prefix_ids = {cfg.codec_think_id, cfg.codec_think_bos_id,
                            language_id, cfg.codec_think_eos_id};
    }
    auto codec_emb0 = RunCodecEmbedHelper(codec_prefix_ids);   // [1, P0, D]
    auto codec_emb1 = RunCodecEmbedHelper({cfg.codec_pad_id, cfg.codec_bos_id});
    const int32_t p0 = static_cast<int32_t>(codec_prefix_ids.size());
    const bool has_spk = clone_prompt != nullptr;
    const int32_t n_codec = p0 + (has_spk ? 1 : 0) + 2;
    const int32_t base_rows = n_codec - 1;

    const float* spd = special_embed.GetTensorData<float>();
    const float* rld = role_embed.GetTensorData<float>();
    const float* c0d = codec_emb0.GetTensorData<float>();
    const float* c1d = codec_emb1.GetTensorData<float>();

    // 组装 codec_input 行（行式：[codec_emb0..., spk? , pad, bos]）
    std::vector<float> codec_input(n_codec * D);
    {
        int32_t r = 0;
        for (int32_t i = 0; i < p0; ++i, ++r)
            std::copy(c0d + i * D, c0d + (i + 1) * D, codec_input.data() + r * D);
        if (has_spk) {
            std::copy(clone_prompt->ref_spk_embed.begin(),
                      clone_prompt->ref_spk_embed.end(),
                      codec_input.data() + r * D);
            ++r;
        }
        std::copy(c1d, c1d + D, codec_input.data() + r * D);  // pad
        ++r;
        std::copy(c1d + D, c1d + 2 * D, codec_input.data() + r * D);  // bos
    }

    // 文本 body
    const int32_t text_start = 3;
    const int32_t text_end =
        static_cast<int32_t>(input_ids.size()) - 5;
    std::vector<int64_t> body_text_ids;
    if (text_end > text_start)
        body_text_ids.assign(input_ids.begin() + text_start,
                             input_ids.begin() + text_end);

    // 前向 prefill 缓冲：role(3) + base_rows + 后续（text_first 或 icl）
    std::vector<float> prefill_data;
    prefill_data.reserve(static_cast<size_t>(3 + base_rows + input_ids.size()) * D);

    auto append_row = [&](const float* row) {
        prefill_data.insert(prefill_data.end(), row, row + D);
    };

    for (int32_t i = 0; i < 3; ++i) append_row(rld + i * D);

    for (int32_t i = 0; i < base_rows; ++i) {
        const float* tts_row = (i < base_rows - 1) ? (spd + 2 * D) : (spd + 0 * D);
        const float* cc_row = codec_input.data() + i * D;
        std::vector<float> row(D);
        for (int32_t d = 0; d < D; ++d) row[d] = tts_row[d] + cc_row[d];
        prefill_data.insert(prefill_data.end(), row.begin(), row.end());
    }

    // trailing 文本（AR 每步喂入）
    std::vector<std::vector<float>> trailing;
    const std::vector<float> tts_pad_vec(spd + 2 * D, spd + 3 * D);

    const bool icl = clone_prompt && !clone_prompt->ref_codes.empty();
    if (icl) {
        // ---- 官方 generate_icl_prompt(non_streaming_mode=False) ----
        std::vector<int64_t> text_ids = clone_prompt->ref_ids;
        text_ids.insert(text_ids.end(), body_text_ids.begin(), body_text_ids.end());
        auto text_embed = RunTextProjectHelper(text_ids);  // [1,T1,D]
        const float* td = text_embed.GetTensorData<float>();
        const int32_t T1 = static_cast<int32_t>(text_ids.size());

        const auto& ref_codes = clone_prompt->ref_codes;
        const int32_t T2 = static_cast<int32_t>(ref_codes.size());

        // 每帧 16 组码求和
        std::vector<float> codec_sum(static_cast<size_t>(T2) * D);
        {
            std::vector<int64_t> col(T2);
            for (int32_t t = 0; t < T2; ++t) col[t] = ref_codes[t][0];
            auto emb0 = RunCodecEmbedHelper(col);  // [1,T2,D]
            std::copy(emb0.GetTensorData<float>(),
                      emb0.GetTensorData<float>() + (size_t)T2 * D,
                      codec_sum.begin());
            for (int32_t j = 1; j < num_code_groups; ++j) {
                for (int32_t t = 0; t < T2; ++t) col[t] = ref_codes[t][j];
                std::array<int64_t, 2> cs = {1, T2};
                Ort::Value ids = Ort::Value::CreateTensor(
                    alloc, cs.data(), cs.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
                std::copy(col.begin(), col.end(),
                          ids.GetTensorMutableData<int64_t>());
                std::array<int64_t, 1> gs = {1};
                Ort::Value step = Ort::Value::CreateTensor(
                    alloc, gs.data(), gs.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
                step.GetTensorMutableData<int64_t>()[0] = j - 1;
                auto emb = model_.RunCodePredictorEmbed(std::move(ids), std::move(step));
                const float* ed = emb.GetTensorData<float>();
                for (int32_t i = 0; i < T2 * D; ++i) codec_sum[i] += ed[i];
            }
        }

        // codec 行：bos + 每帧求和
        const int32_t T2c = T2 + 1;  // bos + frames
        std::vector<float> codec_rows(static_cast<size_t>(T2c) * D);
        std::copy(c1d + D, c1d + 2 * D, codec_rows.data());  // bos
        std::copy(codec_sum.begin(), codec_sum.end(),
                  codec_rows.begin() + D);

        // text 行：text_embed + tts_eos
        const int32_t T1p = T1 + 1;
        std::vector<float> text_rows(static_cast<size_t>(T1p) * D);
        for (int32_t i = 0; i < T1; ++i)
            std::copy(td + i * D, td + (i + 1) * D, text_rows.data() + (size_t)i * D);
        std::copy(spd + D, spd + 2 * D, text_rows.data() + (size_t)T1 * D);  // eos

        if (T1p > T2c) {
            for (int32_t i = 0; i < T2c; ++i) {
                std::vector<float> row(D);
                const float* tr = text_rows.data() + (size_t)i * D;
                const float* cr = codec_rows.data() + (size_t)i * D;
                for (int32_t d = 0; d < D; ++d) row[d] = tr[d] + cr[d];
                prefill_data.insert(prefill_data.end(), row.begin(), row.end());
            }
            for (int32_t i = T2c; i < T1p; ++i) {
                std::vector<float> row(text_rows.data() + (size_t)i * D,
                                       text_rows.data() + (size_t)(i + 1) * D);
                trailing.push_back(std::move(row));
            }
        } else {
            // text 短于 codec：pad 到 T2c
            for (int32_t i = 0; i < T2c; ++i) {
                std::vector<float> row(D);
                const float* tr =
                    (i < T1p) ? (text_rows.data() + (size_t)i * D)
                              : tts_pad_vec.data();
                const float* cr = codec_rows.data() + (size_t)i * D;
                for (int32_t d = 0; d < D; ++d) row[d] = tr[d] + cr[d];
                prefill_data.insert(prefill_data.end(), row.begin(), row.end());
            }
            trailing.push_back(tts_pad_vec);
        }
    } else {
        // 非 clone（text_first + codec_bos）
        {
            auto first_embed = RunTextProjectHelper({body_text_ids[0]});
            const float* fd = first_embed.GetTensorData<float>();
            const float* bos_row = codec_input.data() + (n_codec - 1) * D;
            std::vector<float> row(D);
            for (int32_t d = 0; d < D; ++d) row[d] = fd[d] + bos_row[d];
            prefill_data.insert(prefill_data.end(), row.begin(), row.end());
        }
        if (body_text_ids.size() > 1) {
            const std::vector<int64_t> trail_ids(body_text_ids.begin() + 1,
                                                 body_text_ids.end());
            auto te = RunTextProjectHelper(trail_ids);
            const float* tdd = te.GetTensorData<float>();
            const int32_t n = static_cast<int32_t>(trail_ids.size());
            for (int32_t i = 0; i < n; ++i)
                trailing.emplace_back(tdd + i * D, tdd + (i + 1) * D);
        }
        trailing.emplace_back(spd + D, spd + 2 * D);  // tts_eos
    }

    const int32_t prefill_len = static_cast<int32_t>(prefill_data.size()) / D;

    // ---- Step 2: talker prefill ----
    std::array<int64_t, 3> ps = {1, prefill_len, D};
    Ort::Value prefill_embeds = Ort::Value::CreateTensor(
        alloc, ps.data(), ps.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::copy(prefill_data.begin(), prefill_data.end(),
              prefill_embeds.GetTensorMutableData<float>());

    std::array<int64_t, 2> ms = {1, prefill_len};
    Ort::Value attn_mask = Ort::Value::CreateTensor(
        alloc, ms.data(), ms.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    std::fill(attn_mask.GetTensorMutableData<int64_t>(),
              attn_mask.GetTensorMutableData<int64_t>() + prefill_len, 1LL);

    auto pr = model_.RunTalkerPrefill(std::move(prefill_embeds),
                                      std::move(attn_mask));
    Ort::Value logits = std::move(pr.logits);
    Ort::Value last_hidden = std::move(pr.last_hidden);
    Qwen3TalkerState talker_state = std::move(pr.state);
    int64_t total_seq_len = prefill_len;

    const int32_t suppress_start = cfg.talker_vocab_size - 1024;
    const int32_t suppress_end = cfg.talker_vocab_size;
    constexpr int32_t kMinNewTokens = 2;

    bool stop_requested = false;

    // ---- Step 3: AR generate loop ----
    std::vector<std::vector<int64_t>> codes;
    std::vector<int64_t> generated_primary;

    for (int32_t step = 0; step < gc.max_new_tokens && !stop_requested; ++step) {
        const int64_t primary_code = SampleFromLogits(
            logits, cfg.talker_vocab_size, gc.temperature, gc.top_k, gc.top_p,
            gc.rep_penalty, generated_primary, suppress_start, suppress_end,
            cfg.codec_eos_token_id, step < kMinNewTokens);

        if (primary_code == cfg.codec_eos_token_id) {
            if (step < kMinNewTokens) {
                // 前 2 步已压制 EOS，不可能走到；防御
            }
            break;
        }

        generated_primary.push_back(primary_code);

        auto primary_embed = RunCodecEmbedHelper({primary_code});

        std::vector<int64_t> frame_codes(num_code_groups);
        frame_codes[0] = primary_code;

        const float* lh_data = last_hidden.GetTensorData<float>();
        const int32_t lh_seq = static_cast<int32_t>(
            last_hidden.GetTensorTypeAndShapeInfo().GetShape()[1]);
        const float* pe_data = primary_embed.GetTensorData<float>();

        std::vector<float> cp_ctx(2 * D);
        std::copy(lh_data + (lh_seq - 1) * D, lh_data + lh_seq * D,
                  cp_ctx.begin());
        std::copy(pe_data, pe_data + D, cp_ctx.begin() + D);

        std::vector<float> codec_sum(D);
        std::copy(pe_data, pe_data + D, codec_sum.begin());

        for (int32_t j = 0; j < num_code_groups - 1; ++j) {
            const int32_t cp_len = static_cast<int32_t>(cp_ctx.size()) / D;
            std::array<int64_t, 3> cp_shape = {1, cp_len, D};
            Ort::Value cp_emb = Ort::Value::CreateTensor(
                alloc, cp_shape.data(), cp_shape.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            std::copy(cp_ctx.begin(), cp_ctx.end(),
                      cp_emb.GetTensorMutableData<float>());

            std::array<int64_t, 1> gs_shape = {1};
            Ort::Value gen_step = Ort::Value::CreateTensor(
                alloc, gs_shape.data(), gs_shape.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
            gen_step.GetTensorMutableData<int64_t>()[0] = j;

            auto cp_logits = model_.RunCodePredictor(std::move(cp_emb),
                                                     std::move(gen_step));
            const int64_t res_code = SampleFromLogits(
                cp_logits, cfg.code_predictor_vocab_size, gc.sub_temperature,
                gc.sub_top_k, gc.sub_top_p, 1.0f, {}, -1, -1, -1, false);
            frame_codes[j + 1] = res_code;

            std::array<int64_t, 2> rid_shape = {1, 1};
            Ort::Value rid = Ort::Value::CreateTensor(
                alloc, rid_shape.data(), rid_shape.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
            rid.GetTensorMutableData<int64_t>()[0] = res_code;

            Ort::Value gs2 = Ort::Value::CreateTensor(
                alloc, gs_shape.data(), gs_shape.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
            gs2.GetTensorMutableData<int64_t>()[0] = j;

            auto res_emb = model_.RunCodePredictorEmbed(std::move(rid), std::move(gs2));
            const float* rd = res_emb.GetTensorData<float>();
            cp_ctx.insert(cp_ctx.end(), rd, rd + D);
            for (int32_t d = 0; d < D; ++d) codec_sum[d] += rd[d];
        }

        codes.push_back(frame_codes);

        // 4d: 下一帧输入 = codec_sum + （trailing 文本或 tts_pad）
        const std::vector<float>& txt_hidden =
            (step < static_cast<int32_t>(trailing.size())) ? trailing[step]
                                                           : tts_pad_vec;
        std::vector<float> next_in(D);
        for (int32_t d = 0; d < D; ++d) next_in[d] = codec_sum[d] + txt_hidden[d];

        std::array<int64_t, 3> ne_shape = {1, 1, D};
        Ort::Value next_emb = Ort::Value::CreateTensor(
            alloc, ne_shape.data(), ne_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        std::copy(next_in.begin(), next_in.end(),
                  next_emb.GetTensorMutableData<float>());

        total_seq_len++;
        std::array<int64_t, 2> nm_shape = {1, total_seq_len};
        Ort::Value new_mask = Ort::Value::CreateTensor(
            alloc, nm_shape.data(), nm_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        std::fill(new_mask.GetTensorMutableData<int64_t>(),
                  new_mask.GetTensorMutableData<int64_t>() + total_seq_len, 1LL);

        auto dr = model_.RunTalkerDecode(std::move(next_emb), std::move(new_mask),
                                         std::move(talker_state));
        logits = std::move(dr.logits);
        last_hidden = std::move(dr.last_hidden);
        talker_state = std::move(dr.state);

        // streaming：无 decode_stream → 仅发进度回调（mode B 退化）
        if (streaming &&
            static_cast<int32_t>(codes.size()) % chunk_frames == 0) {
            const float progress =
                static_cast<float>(codes.size()) / gc.max_new_tokens;
            if (!cb(nullptr, 0, progress)) {
                stop_requested = true;
            }
        }
    }

    if (all_codes) *all_codes = codes;

    // AR 阶段结束，释放 text_project/talker_prefill/codec 等大 session，
    // 为 tokenizer12hz_decode 的 ~GB 级中间缓冲腾出内存。
    model_.ReleaseGenerationModels();

    // ---- Step 4: 解码音频 ----
    std::vector<float> samples;
    if (clone_prompt && icl) {
        // clone：拼接 ref codes + 生成 codes，解码后裁掉 ref 部分
        std::vector<std::vector<int64_t>> combined = clone_prompt->ref_codes;
        combined.insert(combined.end(), codes.begin(), codes.end());
        const auto full = DecodeFrames(combined);
        const int64_t ref_len = static_cast<int64_t>(clone_prompt->ref_codes.size());
        const int64_t total_len = static_cast<int64_t>(combined.size());
        if (total_len > 0 && static_cast<int64_t>(full.size()) > 0) {
            const int64_t cut =
                static_cast<int64_t>(static_cast<double>(ref_len) /
                                     static_cast<double>(total_len) *
                                     static_cast<double>(full.size()));
            if (cut >= 0 && cut < static_cast<int64_t>(full.size()))
                samples.assign(full.begin() + cut, full.end());
            else
                samples.assign(full.begin(), full.end());
        } else {
            samples = full;
        }
    } else if (streaming) {
        // 假流式（mode B）：一次性解码，再按 chunk 回调分块投递
        const auto full = DecodeFrames(codes);
        const int32_t total_s = static_cast<int32_t>(full.size());
        const int32_t chunk_s = chunk_frames * kSamplesPerFrame;
        for (int32_t off = 0; off < total_s && !stop_requested; off += chunk_s) {
            const int32_t n = std::min(chunk_s, total_s - off);
            const float prog =
                static_cast<float>(off + n) / static_cast<float>(total_s);
            if (!cb(full.data() + off, n, prog)) stop_requested = true;
        }
        samples = full;
    } else {
        samples = DecodeFrames(codes);
        if (cb) cb(samples.data(), static_cast<int32_t>(samples.size()), 1.0f);
    }

    if (audio) *audio = std::move(samples);
    return !codes.empty();
}

}  // namespace modeldeploy::audio::tts
