// csrc/audio/tts/qwen3/qwen3_tts_model.cpp
#include "qwen3_tts_model.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>
#include "audio/tts/common/ort_ep.h"
#include "core/md_log.h"

namespace modeldeploy::audio::tts {
namespace {

std::string LoadFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

std::string JoinPath(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    const char last = dir.back();
    if (last == '/' || last == '\\') return dir + name;
    return dir + "/" + name;
}

std::vector<const char*> MakePtrs(const std::vector<std::string>& names,
                                  std::vector<const char*>* out_ptrs) {
    out_ptrs->clear();
    out_ptrs->reserve(names.size());
    for (const auto& n : names) out_ptrs->push_back(n.c_str());
    return *out_ptrs;
}

void GetSessionIoNames(Ort::Session* sess, std::vector<std::string>* in_names,
                       std::vector<const char*>* in_ptrs,
                       std::vector<std::string>* out_names,
                       std::vector<const char*>* out_ptrs) {
    Ort::AllocatorWithDefaultOptions alloc;
    in_names->clear();
    out_names->clear();
    const size_t n_in = sess->GetInputCount();
    const size_t n_out = sess->GetOutputCount();
    in_names->reserve(n_in);
    out_names->reserve(n_out);
    for (size_t i = 0; i < n_in; ++i) {
        auto name = sess->GetInputNameAllocated(i, alloc);
        in_names->emplace_back(name.get());
    }
    for (size_t i = 0; i < n_out; ++i) {
        auto name = sess->GetOutputNameAllocated(i, alloc);
        out_names->emplace_back(name.get());
    }
    MakePtrs(*in_names, in_ptrs);
    MakePtrs(*out_names, out_ptrs);
}

}  // namespace

class Qwen3TtsModel::Impl {
public:
    struct Sess {
        std::unique_ptr<Ort::Session> sess;
        std::vector<std::string> in_names;
        std::vector<const char*> in_ptrs;
        std::vector<std::string> out_names;
        std::vector<const char*> out_ptrs;
        std::string path;

        bool Load(Ort::Env& env, const Ort::SessionOptions& opts,
                  const std::string& model_path) {
            if (model_path.empty()) return false;
            std::ifstream ifs(model_path, std::ios::binary);
            if (!ifs) return false;
            std::vector<char> data((std::istreambuf_iterator<char>(ifs)),
                                   std::istreambuf_iterator<char>());
            if (data.empty()) return false;
            path = model_path;
            // 用内存模式创建 session（ORT 在 Windows 下路径需 wchar，避免转换）
            sess = std::make_unique<Ort::Session>(env, data.data(),
                                                  data.size(), opts);
            GetSessionIoNames(sess.get(), &in_names, &in_ptrs, &out_names,
                              &out_ptrs);
            return true;
        }

        // 释放权重内存：大模型（talker/文本投影等）在不使用阶段释放，
        // 给 tokenizer12hz_decode 的 ~GB 级中间缓冲腾出空间。
        void Release() { sess.reset(); }

        [[nodiscard]] bool Ready() const { return sess != nullptr; }

        bool EnsureLoaded(Ort::Env& env, const Ort::SessionOptions& opts) {
            return sess ? true : Load(env, opts, path);
        }
    };

    explicit Impl(const std::string& model_dir, const RuntimeOption& opt)
        : env_(ORT_LOGGING_LEVEL_ERROR) {
        auto opts = std::make_shared<Ort::SessionOptions>();
        // 内存受限环境：ALL 级优化会把大模型权重/中间结果复制成多份，
        // 累积 9 个 session 后被 tokenizer12hz_decode 的 ~GB 级 ConvTranspose
        // 中间缓冲挤垮（OOM）。降级到 BASIC。
        opts->SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        opts->SetLogSeverityLevel(3);
        if (opt.cpu_thread_num > 0) opts->SetIntraOpNumThreads(opt.cpu_thread_num);
        opts->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        ApplyOrtCudaEp(*opts, opt.device, opt.device_id);
        sess_opts_ = opts;
        const Ort::SessionOptions& so = *opts;

        const std::string onnx_dir = JoinPath(model_dir, "onnx_kv_06b");
        bool ok = text_project_.Load(env_, so, JoinPath(onnx_dir, "text_project.onnx"));
        ok = codec_embed_.Load(env_, so, JoinPath(onnx_dir, "codec_embed.onnx")) && ok;
        ok = code_predictor_embed_.Load(env_, so,
                                        JoinPath(onnx_dir, "code_predictor_embed.onnx")) && ok;
        ok = code_predictor_.Load(env_, so,
                                  JoinPath(onnx_dir, "code_predictor.onnx")) && ok;
        ok = talker_prefill_.Load(env_, so, JoinPath(onnx_dir, "talker_prefill.onnx")) && ok;
        ok = talker_decode_.Load(env_, so, JoinPath(onnx_dir, "talker_decode.onnx")) && ok;
        ok = speaker_encoder_.Load(env_, so,
                                   JoinPath(onnx_dir, "speaker_encoder.onnx")) && ok;
        ok = tokenizer12hz_encode_.Load(env_, so,
                                        JoinPath(onnx_dir, "tokenizer12hz_encode.onnx")) && ok;
        ok = tokenizer12hz_decode_.Load(env_, so,
                                        JoinPath(onnx_dir, "tokenizer12hz_decode.onnx")) && ok;
        // 可选 streaming decoder（本模型包没有，保持空）
        loaded_ = ok;

        ResolveTokenizerDir(model_dir);
        LoadConfigJson();
    }

    // 生成前确保 AR 阶段相关 session 已加载（可能被 ReleaseGenerationModels
    // 释放过）。decode 阶段不用的 1.7GB(talker_prefill/talker_decode)/
    // 1.2GB(text_project)/~0.6GB(codec*+code_predictor*) 大 session 释放后
    // 给 tokenizer12hz_decode 的 ~GB 级中间缓冲腾内存。
    bool EnsureGenerationModels() {
        bool ok = text_project_.EnsureLoaded(env_, *sess_opts_);
        ok = talker_prefill_.EnsureLoaded(env_, *sess_opts_) && ok;
        ok = talker_decode_.EnsureLoaded(env_, *sess_opts_) && ok;
        ok = codec_embed_.EnsureLoaded(env_, *sess_opts_) && ok;
        ok = code_predictor_.EnsureLoaded(env_, *sess_opts_) && ok;
        ok = code_predictor_embed_.EnsureLoaded(env_, *sess_opts_) && ok;
        return ok;
    }
    void ReleaseGenerationModels() {
        text_project_.Release();
        talker_prefill_.Release();
        talker_decode_.Release();
        codec_embed_.Release();
        code_predictor_.Release();
        code_predictor_embed_.Release();
    }

    void ResolveTokenizerDir(const std::string& model_dir) {
        // 优先固定路径，其次扫描 models/* 找含 vocab.json 的目录
        const std::string fixed =
            JoinPath(model_dir, "models/Qwen3-TTS-12Hz-0.6B-Base");
        if (!LoadFile(JoinPath(fixed, "vocab.json")).empty()) {
            tokenizer_dir_ = fixed;
            return;
        }
#ifdef _WIN32
        const std::string sep = "\\";
#else
        const std::string sep = "/";
#endif
        const std::string models_dir = JoinPath(model_dir, "models");
        std::ifstream probe;  // 不做完整目录扫描，固定地址即可
        (void)sep;
        (void)probe;
        tokenizer_dir_ = fixed;  // 固定地址；找不到时 tokenizer 构造会报错
    }

    void LoadConfigJson() {
        const std::string config_file = JoinPath(tokenizer_dir_, "config.json");
        const std::string blob = LoadFile(config_file);
        if (blob.empty()) {
            MD_LOG_WARN << "qwen3: config.json missing, use defaults"
                        << std::endl;
            return;
        }
        using nlohmann::json;
        json cfg;
        try {
            cfg = json::parse(blob);
        } catch (const std::exception& e) {
            MD_LOG_WARN << "qwen3: failed to parse " << config_file
                        << ": " << e.what() << std::endl;
            return;
        }
        if (!cfg.is_object()) {
            MD_LOG_WARN << "qwen3: invalid config.json" << std::endl;
            return;
        }
        if (cfg.contains("talker_config") &&
            cfg["talker_config"].is_object()) {
            const json& tc = cfg["talker_config"];
            tts_config_.num_code_groups =
                tc.value("num_code_groups", tts_config_.num_code_groups);
            tts_config_.hidden_size =
                tc.value("hidden_size", tts_config_.hidden_size);
            tts_config_.talker_vocab_size =
                tc.value("vocab_size", tts_config_.talker_vocab_size);
            tts_config_.num_hidden_layers =
                tc.value("num_hidden_layers", tts_config_.num_hidden_layers);
            tts_config_.text_vocab_size =
                tc.value("text_vocab_size", tts_config_.text_vocab_size);
            tts_config_.codec_bos_id =
                tc.value("codec_bos_id", tts_config_.codec_bos_id);
            tts_config_.codec_eos_token_id =
                tc.value("codec_eos_token_id",
                         tts_config_.codec_eos_token_id);
            tts_config_.codec_pad_id =
                tc.value("codec_pad_id", tts_config_.codec_pad_id);
            tts_config_.codec_nothink_id =
                tc.value("codec_nothink_id", tts_config_.codec_nothink_id);
            tts_config_.codec_think_id =
                tc.value("codec_think_id", tts_config_.codec_think_id);
            tts_config_.codec_think_bos_id =
                tc.value("codec_think_bos_id",
                         tts_config_.codec_think_bos_id);
            tts_config_.codec_think_eos_id =
                tc.value("codec_think_eos_id",
                         tts_config_.codec_think_eos_id);
            if (tc.contains("code_predictor_config") &&
                tc["code_predictor_config"].is_object()) {
                tts_config_.code_predictor_vocab_size =
                    tc["code_predictor_config"].value(
                        "vocab_size", tts_config_.code_predictor_vocab_size);
            }
            if (tc.contains("codec_language_id") &&
                tc["codec_language_id"].is_object()) {
                for (auto it = tc["codec_language_id"].begin();
                     it != tc["codec_language_id"].end(); ++it) {
                    if (it.value().is_number_integer())
                        tts_config_.codec_language_id[it.key()] =
                            it.value().get<int64_t>();
                }
            }
        }
        tts_config_.tts_bos_token_id = cfg.value(
            "tts_bos_token_id", tts_config_.tts_bos_token_id);
        tts_config_.tts_eos_token_id = cfg.value(
            "tts_eos_token_id", tts_config_.tts_eos_token_id);
        tts_config_.tts_pad_token_id = cfg.value(
            "tts_pad_token_id", tts_config_.tts_pad_token_id);

        tts_config_.clone_supported =
            speaker_encoder_.Ready() && tokenizer12hz_encode_.Ready();

        MD_LOG_INFO << "Qwen3TtsConfig: num_code_groups="
                    << tts_config_.num_code_groups
                    << " hidden_size=" << tts_config_.hidden_size
                    << " talker_vocab=" << tts_config_.talker_vocab_size
                    << " code_predictor_vocab="
                    << tts_config_.code_predictor_vocab_size
                    << " clone_supported=" << tts_config_.clone_supported
                    << std::endl;
    }

    Ort::Value RunTextProject(Ort::Value v) const {
        auto out = text_project_.sess->Run(
            {}, text_project_.in_ptrs.data(), &v, 1,
            text_project_.out_ptrs.data(), text_project_.out_ptrs.size());
        return std::move(out[0]);
    }

    Ort::Value RunCodecEmbed(Ort::Value v) const {
        auto out = codec_embed_.sess->Run(
            {}, codec_embed_.in_ptrs.data(), &v, 1,
            codec_embed_.out_ptrs.data(), codec_embed_.out_ptrs.size());
        return std::move(out[0]);
    }

    Ort::Value RunCodePredictorEmbed(Ort::Value ids, Ort::Value step) const {
        std::array<Ort::Value, 2> inputs = {std::move(ids), std::move(step)};
        auto out = code_predictor_embed_.sess->Run(
            {}, code_predictor_embed_.in_ptrs.data(), inputs.data(),
            inputs.size(), code_predictor_embed_.out_ptrs.data(),
            code_predictor_embed_.out_ptrs.size());
        return std::move(out[0]);
    }

    Ort::Value RunCodePredictor(Ort::Value embeds, Ort::Value step) const {
        std::array<Ort::Value, 2> inputs = {std::move(embeds), std::move(step)};
        auto out = code_predictor_.sess->Run(
            {}, code_predictor_.in_ptrs.data(), inputs.data(), inputs.size(),
            code_predictor_.out_ptrs.data(), code_predictor_.out_ptrs.size());
        return std::move(out[0]);
    }

    Ort::Value RunSpeakerEncoder(Ort::Value mels) const {
        auto out = speaker_encoder_.sess->Run(
            {}, speaker_encoder_.in_ptrs.data(), &mels, 1,
            speaker_encoder_.out_ptrs.data(), speaker_encoder_.out_ptrs.size());
        return std::move(out[0]);
    }

    Qwen3TtsModel::Tokenizer12hzEncodeResult RunTokenizer12hzEncode(
        Ort::Value input_values, Ort::Value padding_mask) const {
        std::array<Ort::Value, 2> inputs = {std::move(input_values),
                                            std::move(padding_mask)};
        auto out = tokenizer12hz_encode_.sess->Run(
            {}, tokenizer12hz_encode_.in_ptrs.data(), inputs.data(),
            inputs.size(), tokenizer12hz_encode_.out_ptrs.data(),
            tokenizer12hz_encode_.out_ptrs.size());
        Qwen3TtsModel::Tokenizer12hzEncodeResult r;
        r.audio_codes = std::move(out[0]);
        if (out.size() > 1) r.lengths = std::move(out[1]);
        return r;
    }

    Qwen3TtsModel::Tokenizer12hzDecodeResult RunTokenizer12hzDecode(
        Ort::Value audio_codes) const {
        auto out = tokenizer12hz_decode_.sess->Run(
            {}, tokenizer12hz_decode_.in_ptrs.data(), &audio_codes, 1,
            tokenizer12hz_decode_.out_ptrs.data(),
            tokenizer12hz_decode_.out_ptrs.size());
        Qwen3TtsModel::Tokenizer12hzDecodeResult r;
        r.audio_values = std::move(out[0]);
        if (out.size() > 1) r.lengths = std::move(out[1]);
        return r;
    }

    Qwen3TtsModel::TalkerPrefillResult RunTalkerPrefill(
        Ort::Value embeds, Ort::Value mask) const {
        std::array<Ort::Value, 2> inputs = {std::move(embeds), std::move(mask)};
        auto out = talker_prefill_.sess->Run(
            {}, talker_prefill_.in_ptrs.data(), inputs.data(), inputs.size(),
            talker_prefill_.out_ptrs.data(), talker_prefill_.out_ptrs.size());
        Qwen3TtsModel::TalkerPrefillResult r;
        r.logits = std::move(out[0]);
        r.last_hidden = std::move(out[1]);
        for (size_t i = 2; i < out.size(); ++i)
            r.state.kv_cache.push_back(std::move(out[i]));
        return r;
    }

    Qwen3TtsModel::TalkerDecodeResult RunTalkerDecode(
        Ort::Value embeds, Ort::Value mask, Qwen3TalkerState state) const {
        std::vector<Ort::Value> inputs;
        inputs.reserve(2 + state.kv_cache.size());
        inputs.push_back(std::move(embeds));
        inputs.push_back(std::move(mask));
        for (auto& kv : state.kv_cache) inputs.push_back(std::move(kv));
        auto out = talker_decode_.sess->Run(
            {}, talker_decode_.in_ptrs.data(), inputs.data(), inputs.size(),
            talker_decode_.out_ptrs.data(), talker_decode_.out_ptrs.size());
        Qwen3TtsModel::TalkerDecodeResult r;
        r.logits = std::move(out[0]);
        r.last_hidden = std::move(out[1]);
        for (size_t i = 2; i < out.size(); ++i)
            r.state.kv_cache.push_back(std::move(out[i]));
        return r;
    }

    const Qwen3TtsConfig& GetConfig() const { return tts_config_; }
    const std::string& tokenizer_dir() const { return tokenizer_dir_; }
    OrtAllocator* Allocator() { return allocator_; }
    bool loaded() const { return loaded_; }
    bool HasTokenizer12hzDecodeStream() const { return false; }

private:
    bool loaded_ = false;
    std::string tokenizer_dir_;
    Qwen3TtsConfig tts_config_;
    Ort::Env env_;
    std::shared_ptr<Ort::SessionOptions> sess_opts_;
    Ort::AllocatorWithDefaultOptions allocator_;

    Sess text_project_;
    Sess codec_embed_;
    Sess code_predictor_embed_;
    Sess code_predictor_;
    Sess talker_prefill_;
    Sess talker_decode_;
    Sess speaker_encoder_;
    Sess tokenizer12hz_encode_;
    Sess tokenizer12hz_decode_;
};

Qwen3TtsModel::Qwen3TtsModel() = default;

Qwen3TtsModel::~Qwen3TtsModel() = default;

bool Qwen3TtsModel::Load(const std::string& model_dir,
                         const RuntimeOption& opt) {
    impl_ = std::make_unique<Impl>(model_dir, opt);
    return impl_->loaded();
}

bool Qwen3TtsModel::loaded() const {
    return impl_ && impl_->loaded();
}

bool Qwen3TtsModel::EnsureGenerationModels() {
    return impl_ && impl_->EnsureGenerationModels();
}

void Qwen3TtsModel::ReleaseGenerationModels() {
    if (impl_) impl_->ReleaseGenerationModels();
}

Ort::Value Qwen3TtsModel::RunTextProject(Ort::Value v) const {
    return impl_->RunTextProject(std::move(v));
}
Ort::Value Qwen3TtsModel::RunCodecEmbed(Ort::Value v) const {
    return impl_->RunCodecEmbed(std::move(v));
}
Ort::Value Qwen3TtsModel::RunCodePredictorEmbed(Ort::Value ids,
                                                Ort::Value step) const {
    return impl_->RunCodePredictorEmbed(std::move(ids), std::move(step));
}
Ort::Value Qwen3TtsModel::RunCodePredictor(Ort::Value embeds,
                                           Ort::Value step) const {
    return impl_->RunCodePredictor(std::move(embeds), std::move(step));
}
Qwen3TtsModel::TalkerPrefillResult Qwen3TtsModel::RunTalkerPrefill(
    Ort::Value embeds, Ort::Value mask) const {
    return impl_->RunTalkerPrefill(std::move(embeds), std::move(mask));
}
Qwen3TtsModel::TalkerDecodeResult Qwen3TtsModel::RunTalkerDecode(
    Ort::Value embeds, Ort::Value mask, Qwen3TalkerState state) const {
    return impl_->RunTalkerDecode(std::move(embeds), std::move(mask),
                                  std::move(state));
}
Ort::Value Qwen3TtsModel::RunSpeakerEncoder(Ort::Value mels) const {
    return impl_->RunSpeakerEncoder(std::move(mels));
}
Qwen3TtsModel::Tokenizer12hzEncodeResult Qwen3TtsModel::RunTokenizer12hzEncode(
    Ort::Value input_values, Ort::Value padding_mask) const {
    return impl_->RunTokenizer12hzEncode(std::move(input_values),
                                         std::move(padding_mask));
}
Qwen3TtsModel::Tokenizer12hzDecodeResult Qwen3TtsModel::RunTokenizer12hzDecode(
    Ort::Value audio_codes) const {
    return impl_->RunTokenizer12hzDecode(std::move(audio_codes));
}
bool Qwen3TtsModel::HasTokenizer12hzDecodeStream() const {
    return impl_ && impl_->HasTokenizer12hzDecodeStream();
}
const Qwen3TtsConfig& Qwen3TtsModel::GetConfig() const {
    return impl_->GetConfig();
}
const std::string& Qwen3TtsModel::tokenizer_dir() const {
    return impl_->tokenizer_dir();
}
OrtAllocator* Qwen3TtsModel::Allocator() const { return impl_->Allocator(); }

// ===== mel（speaker_encoder 前置）：port 自 Qwen3-TTS mel_spectrogram =====

namespace {

// librosa slaney mel 滤波组（sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000）
constexpr int kMelNFft = 1024;
constexpr int kMelNMels = 128;
constexpr float kMelSr = 24000.0f;
constexpr float kMelFmin = 0.0f;
constexpr float kMelFmax = 12000.0f;
const float kSpkFSp = 200.0f / 3.0f;
const float kSpkMinLogHz = 1000.0f;
const float kSpkMinLogMel = (kSpkMinLogHz - kMelFmin) / kSpkFSp;
const float kSpkLogStep = std::log(6.4f) / 27.0f;

const std::vector<std::vector<float>>& MelFilterbank() {
    static const std::vector<std::vector<float>> fb = [] {
        constexpr int kNFreqs = kMelNFft / 2 + 1;  // 513

        auto hz_to_mel = [](float f) {
            if (f >= kSpkMinLogHz)
                return kSpkMinLogMel + std::log(f / kSpkMinLogHz) / kSpkLogStep;
            return (f - kMelFmin) / kSpkFSp;
        };
        auto mel_to_hz = [](float m) {
            if (m >= kSpkMinLogMel)
                return kSpkMinLogHz * std::exp(kSpkLogStep * (m - kSpkMinLogMel));
            return kMelFmin + kSpkFSp * m;
        };

        std::vector<float> mel_f(kMelNMels + 2);
        const float lo = hz_to_mel(kMelFmin);
        const float hi = hz_to_mel(kMelFmax);
        for (int i = 0; i < kMelNMels + 2; ++i)
            mel_f[i] = mel_to_hz(lo + (hi - lo) * static_cast<float>(i) /
                                          static_cast<float>(kMelNMels + 1));

        std::vector<float> fftfreqs(kNFreqs);
        for (int k = 0; k < kNFreqs; ++k) fftfreqs[k] = k * kMelSr / kMelNFft;

        std::vector<std::vector<float>> w(kMelNMels,
                                          std::vector<float>(kNFreqs, 0.0f));
        for (int i = 0; i < kMelNMels; ++i) {
            const float fdiff_l = mel_f[i + 1] - mel_f[i];
            const float fdiff_r = mel_f[i + 2] - mel_f[i + 1];
            for (int k = 0; k < kNFreqs; ++k) {
                const float lower = (fftfreqs[k] - mel_f[i]) / fdiff_l;
                const float upper = (mel_f[i + 2] - fftfreqs[k]) / fdiff_r;
                w[i][k] = std::max(0.0f, std::min(lower, upper));
            }
            // slaney 归一化：2 / (上界 - 下界)
            const float enorm = 2.0f / (mel_f[i + 2] - mel_f[i]);
            for (int k = 0; k < kNFreqs; ++k) w[i][k] *= enorm;
        }
        return w;
    }();
    return fb;
}

}  // namespace

bool ComputeSpeakerMel(const std::vector<float>& audio_24k,
                       std::vector<float>* mel_out, int* frames_out) {
    if (mel_out == nullptr || audio_24k.empty()) return false;

    constexpr int kHop = 256;
    constexpr float kClip = 1e-5f;

    const auto& fb = MelFilterbank();
    const int n_freqs = kMelNFft / 2 + 1;

    // hann window (periodic)
    std::vector<float> window(kMelNFft);
    for (int n = 0; n < kMelNFft; ++n)
        window[n] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f *
                                            static_cast<float>(n) /
                                            static_cast<float>(kMelNFft)));

    // reflect pad (padding, padding) = 384 each side
    const int padding = (kMelNFft - kHop) / 2;  // 384
    const size_t n = audio_24k.size();
    std::vector<float> padded(n + 2 * static_cast<size_t>(padding));
    const size_t n1 = n > 1 ? n - 1 : 0;
    for (size_t i = 0; i < static_cast<size_t>(padding); ++i)
        padded[i] = audio_24k[std::min<size_t>(padding - i, n1)];  // x[p],...,x[1]
    for (size_t i = 0; i < n; ++i) padded[padding + i] = audio_24k[i];
    for (size_t i = 0; i < static_cast<size_t>(padding); ++i)
        padded[padding + n + i] = audio_24k[std::max<size_t>(n1, 0) - std::min(i, n1)];

    const size_t L = padded.size();
    if (L < static_cast<size_t>(kMelNFft)) {
        mel_out->clear();
        if (frames_out) *frames_out = 0;
        return true;
    }
    const int frames = static_cast<int>(1 + (L - kMelNFft) / kHop);

    // 预计算 twiddle（半谱，只算 0..511 正频率 + nyquist）
    std::vector<double> tw_cos(kMelNFft * kMelNFft);
    std::vector<double> tw_sin(kMelNFft * kMelNFft);
    {
        constexpr double k2Pi = 2.0 * 3.14159265358979323846;
        for (int k = 0; k < n_freqs; ++k)
            for (int nid = 0; nid < kMelNFft; ++nid) {
                const double ang = k2Pi * static_cast<double>(k) *
                                   static_cast<double>(nid) / kMelNFft;
                tw_cos[k * kMelNFft + nid] = std::cos(ang);
                tw_sin[k * kMelNFft + nid] = -std::sin(ang);
            }
    }

    std::vector<float> spec(n_freqs);
    std::vector<float> mel(kMelNMels);
    std::vector<float> out;
    out.reserve(static_cast<size_t>(frames) * kMelNMels);
    for (int f = 0; f < frames; ++f) {
        std::vector<double> re(kMelNFft, 0.0);
        std::vector<double> im(kMelNFft, 0.0);
        const size_t off = static_cast<size_t>(f) * kHop;
        for (int nid = 0; nid < kMelNFft; ++nid)
            re[nid] = static_cast<double>(padded[off + nid]) * window[nid];
        for (int k = 0; k < n_freqs; ++k) {
            double r = 0.0;
            double ig = 0.0;
            for (int nid = 0; nid < kMelNFft; ++nid) {
                r += re[nid] * tw_cos[k * kMelNFft + nid] - im[nid] * tw_sin[k * kMelNFft + nid];
                ig += re[nid] * tw_sin[k * kMelNFft + nid] + im[nid] * tw_cos[k * kMelNFft + nid];
            }
            spec[k] = static_cast<float>(std::sqrt(r * r + ig * ig + 1e-9));
        }
        for (int m = 0; m < kMelNMels; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < n_freqs; ++k) sum += fb[m][k] * spec[k];
            const float clamped = std::max(sum, kClip);
            mel[m] = std::log(clamped);  // torch.log 自然对数
        }
        out.insert(out.end(), mel.begin(), mel.end());
    }

    if (frames_out) *frames_out = frames;
    *mel_out = std::move(out);
    if (mel_out->empty()) return false;
    return true;
}

}  // namespace modeldeploy::audio::tts
