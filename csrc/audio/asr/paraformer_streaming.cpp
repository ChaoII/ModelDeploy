//
// Created by aichao on 2025/5/19.
//

#include "csrc/audio/asr/paraformer_streaming.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

namespace modeldeploy::audio::asr {

    namespace {

        // tokens.txt: 每行一个符号，id = 行号（第 0 行一般是 <blank>）
        class BasicSymbolTable {
        public:
            bool Load(const std::string& path) {
                std::ifstream in(path);
                if (!in) return false;
                std::string line;
                while (std::getline(in, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty()) continue;
                    // 格式: "<symbol> <id>"（符号可能含空格？末列是整数 id）
                    auto sp = line.rfind(' ');
                    if (sp == std::string::npos) return false;
                    int32_t sym_id = std::stoi(line.substr(sp + 1));
                    syms_[sym_id] = line.substr(0, sp);
                }
                return !syms_.empty();
            }
            const std::string& operator[](int32_t id) const {
                static const std::string kEmpty;
                auto it = syms_.find(id);
                return it == syms_.end() ? kEmpty : it->second;
            }
        private:
            std::map<int32_t, std::string> syms_;
        };

        // 复刻 sherpa Convert()：token 以 "@@" 结尾表示与后段合并成词
        std::string TokensToText(const std::vector<int32_t>& ids,
                                 const BasicSymbolTable& table) {
            std::string text;
            bool mergeable = false;
            for (size_t i = 0; i != ids.size(); ++i) {
                std::string sym = table[ids[i]];
                if (sym.empty()) continue;
                bool double_at = (sym.size() > 2) &&
                                 (sym[sym.size() - 1] == '@') &&
                                 (sym[sym.size() - 2] == '@');
                if (!double_at) {
                    unsigned char p0 = static_cast<unsigned char>(sym[0]);
                    if (p0 < 0x80) {
                        if (mergeable) { mergeable = false; text.append(sym); }
                        else           { text.append(" "); text.append(sym); }
                    } else {
                        mergeable = false;
                        if (i > 0) {
                            const std::string& prev = table[ids[i - 1]];
                            if (!prev.empty() &&
                                static_cast<unsigned char>(prev[0]) < 0x80) {
                                text.append(" ");
                            }
                        }
                        text.append(sym);
                    }
                } else {
                    sym = std::string(sym.data(), sym.size() - 2);
                    if (mergeable) text.append(sym);
                    else { text.append(" "); text.append(sym); mergeable = true; }
                }
            }
            if (!text.empty() && text.front() == ' ') text.erase(text.begin());
            return text;
        }

        inline std::vector<std::string> ParseNames(Ort::Session& sess,
                                                   size_t count,
                                                   Ort::AllocatorWithDefaultOptions& alloc) {
            std::vector<std::string> names;
            names.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                auto ptr = sess.GetInputNameAllocated(i, alloc);
                names.emplace_back(ptr.get());
            }
            return names;
        }

        inline std::vector<std::string> ParseOutNames(Ort::Session& sess,
                                                      size_t count,
                                                      Ort::AllocatorWithDefaultOptions& alloc) {
            std::vector<std::string> names;
            names.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                auto ptr = sess.GetOutputNameAllocated(i, alloc);
                names.emplace_back(ptr.get());
            }
            return names;
        }

        // 把 const char* 名字数组指向上面解析出的 std::string 存储
        inline void MakePtrArray(const std::vector<std::string>& names,
                                 std::vector<const char*>* ptrs) {
            ptrs->clear();
            ptrs->reserve(names.size());
            for (const auto& n : names) ptrs->push_back(n.c_str());
        }

    } // namespace

    class ParaformerStreamingAsr::Impl {
    public:
        // 流式 paraformer 常量（对齐 sherpa online-paraformer）
        static constexpr int32_t kChunkSize = 61;
        static constexpr int32_t kLeftChunk = 5;
        static constexpr int32_t kRightChunk = 3;
        static constexpr int32_t kFeatDim = 80;

        Impl(const std::string& encoder_onnx,
             const std::string& decoder_onnx,
             const std::string& tokens_txt,
             int32_t sample_rate,
             int32_t num_threads,
             float threshold)
            : sample_rate_(sample_rate),
              threshold_(threshold),
              env_(ORT_LOGGING_LEVEL_WARNING, "MD-ParaformerStreaming") {
            sess_opts_.SetIntraOpNumThreads(num_threads > 0 ? num_threads : 1);
            sess_opts_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
            try {
                encoder_sess_ = std::make_unique<Ort::Session>(
                    env_, encoder_onnx.c_str(), sess_opts_);
                decoder_sess_ = std::make_unique<Ort::Session>(
                    env_, decoder_onnx.c_str(), sess_opts_);

                Ort::AllocatorWithDefaultOptions alloc;
                encoder_in_strings_ = ParseNames(*encoder_sess_, encoder_sess_->GetInputCount(), alloc);
                encoder_out_strings_ = ParseOutNames(*encoder_sess_, encoder_sess_->GetOutputCount(), alloc);
                decoder_in_strings_ = ParseNames(*decoder_sess_, decoder_sess_->GetInputCount(), alloc);
                decoder_out_strings_ = ParseOutNames(*decoder_sess_, decoder_sess_->GetOutputCount(), alloc);
                MakePtrArray(encoder_in_strings_, &encoder_in_names_);
                MakePtrArray(encoder_out_strings_, &encoder_out_names_);
                MakePtrArray(decoder_in_strings_, &decoder_in_names_);
                MakePtrArray(decoder_out_strings_, &decoder_out_names_);

                initialized_ = ReadMetadata() && token_table_.Load(tokens_txt);
                if (initialized_) ResetFbank();
            } catch (const std::exception& e) {
                std::cerr << "[ParaformerStreamingAsr] init error: " << e.what() << std::endl;
                initialized_ = false;
            }
        }

        ~Impl() = default;

        bool is_initialized() const { return initialized_; }

        bool ReadMetadata() {
            Ort::AllocatorWithDefaultOptions alloc;
            Ort::ModelMetadata meta = encoder_sess_->GetModelMetadata();

            auto lookup = [&](const char* key, std::string* out) -> bool {
                auto v = meta.LookupCustomMetadataMapAllocated(key, alloc);
                if (!v) return false;
                *out = v.get();
                return true;
            };

            auto read_int = [&lookup](const char* key, int32_t* out) -> bool {
                std::string s;
                if (!lookup(key, &s)) return false;
                try { *out = std::stoi(s); } catch (...) { return false; }
                return true;
            };
            auto read_vec = [&lookup](const char* key, std::vector<float>* out) -> bool {
                std::string s;
                if (!lookup(key, &s)) return false;
                std::vector<float> v;
                size_t pos = 0;
                while (pos < s.size()) {
                    size_t comma = s.find(',', pos);
                    std::string tok = s.substr(
                        pos, comma == std::string::npos ? std::string::npos : comma - pos);
                    try { v.push_back(std::stof(tok)); } catch (...) { return false; }
                    if (comma == std::string::npos) break;
                    pos = comma + 1;
                }
                *out = std::move(v);
                return true;
            };

            read_int("vocab_size", &vocab_size_);
            read_int("lfr_window_size", &lfr_window_size_);
            read_int("lfr_window_shift", &lfr_window_shift_);
            if (!read_int("encoder_output_size", &encoder_output_size_)) {
                std::cerr << "[ParaformerStreamingAsr] no encoder_output_size\n";
                return false;
            }
            read_int("decoder_num_blocks", &decoder_num_blocks_);
            read_int("decoder_kernel_size", &decoder_kernel_size_);
            if (!read_vec("neg_mean", &neg_mean_)) {
                std::cerr << "[ParaformerStreamingAsr] no neg_mean\n";
                return false;
            }
            if (!read_vec("inv_stddev", &inv_stddev_)) {
                std::cerr << "[ParaformerStreamingAsr] no inv_stddev\n";
                return false;
            }
            float scale = std::sqrt(static_cast<float>(encoder_output_size_));
            for (auto& f : inv_stddev_) f *= scale;
            if (decoder_num_blocks_ <= 0 || decoder_kernel_size_ <= 1) {
                std::cerr << "[ParaformerStreamingAsr] bad decoder meta\n";
                return false;
            }
            return true;
        }

        void ResetFbank() {
            knf::FbankOptions opts;
            opts.frame_opts.dither = 0.0f;
            opts.frame_opts.snip_edges = true;
            opts.frame_opts.window_type = "hamming";
            opts.frame_opts.samp_freq = static_cast<float>(sample_rate_);
            opts.frame_opts.frame_shift_ms = 10.0f;
            opts.frame_opts.frame_length_ms = 25.0f;
            opts.mel_opts.num_bins = kFeatDim;
            opts.mel_opts.low_freq = 20.0f;
            opts.mel_opts.high_freq = 0.0f;
            fbank_ = std::make_unique<knf::OnlineFbank>(opts);
        }

        void reset() {
            ResetFbank();
            frames_processed_ = 0;
            feat_cache_.clear();
            initial_hidden_.clear();
            states_.clear();
            all_tokens_.clear();
            alpha_cache_ = 0.f;
            fired_alpha_ = 0.f;
            running_confidence_ = 0.f;
        }

        void accept_waveform(const std::vector<float>& samples) {
            if (!initialized_ || samples.empty()) return;
            fbank_->AcceptWaveform(static_cast<float>(sample_rate_),
                                   samples.data(), static_cast<int32_t>(samples.size()));
        }

        void input_finished() { if (fbank_) fbank_->InputFinished(); }

        bool DecodeStep(bool is_final, std::vector<int32_t>& new_tokens,
                        float& confidence, bool& out_final) {
            if (!initialized_) return false;
            if (is_final && short_final_done_) {
                out_final = true;
                return false;
            }

            int32_t start_processed = frames_processed_;
            int32_t available = fbank_->NumFramesReady() - frames_processed_;
            // 非 final：帧数不足一个 chunk 时先等待（对齐 sherpa IsReady），
            // 不提前推进，避免小片喂入时误处理/越界。
            if (!is_final && available < kChunkSize) return false;
            bool short_final = is_final && available < kChunkSize;

            std::vector<float> frames =
                GetFrames(frames_processed_, short_final ? available : kChunkSize);

            if (short_final) {                frames.resize(static_cast<size_t>(kChunkSize) * kFeatDim, 0.0f);
                frames_processed_ += available;
                short_final_done_ = true;
            } else {
                frames_processed_ += kChunkSize - 1;
            }

            if (frames.empty()) return false;

            int32_t t_offset = start_processed / lfr_window_shift_;
            std::vector<float> lfr = ApplyLFR(frames);
            ApplyCMVN(lfr);
            PositionalEncoding(lfr, t_offset);

            int32_t feat_dim = static_cast<int32_t>(neg_mean_.size());
            if (feat_cache_.empty()) {
                feat_cache_.resize(
                    static_cast<size_t>(kLeftChunk + kRightChunk) * feat_dim, 0.0f);
            }
            lfr.insert(lfr.begin(), feat_cache_.begin(), feat_cache_.end());
            std::copy(lfr.end() - static_cast<ptrdiff_t>(feat_cache_.size()),
                      lfr.end(), feat_cache_.begin());

            int32_t num_frames = static_cast<int32_t>(lfr.size()) / feat_dim;
            auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

            std::array<int64_t, 3> x_shape{1, num_frames, feat_dim};
            Ort::Value x = Ort::Value::CreateTensor(
                mem, lfr.data(), lfr.size(), x_shape.data(), x_shape.size());
            std::array<int64_t, 1> len_shape{1};
            int32_t len_val = num_frames;
            Ort::Value x_len = Ort::Value::CreateTensor(
                mem, &len_val, 1, len_shape.data(), len_shape.size());

            std::array<Ort::Value, 2> enc_in = {std::move(x), std::move(x_len)};
            std::vector<Ort::Value> enc_out = encoder_sess_->Run(
                {}, encoder_in_names_.data(), enc_in.data(), enc_in.size(),
                encoder_out_names_.data(), encoder_out_names_.size());

            bool ok = RunDecoder(enc_out, new_tokens, confidence, out_final);
            confidence = running_confidence_;  // 累积置信度（末步无新 token 也保底）
            if (is_final) out_final = true;  // final 标记的调用即视为段结束
            return ok;
        }

        std::vector<float> GetFrames(int32_t frame_index, int32_t n) {
            std::vector<float> out;
            const int32_t ready = fbank_->NumFramesReady();
            if (frame_index >= ready) return out;
            int32_t m = std::min<int32_t>(n, ready - frame_index);
            out.reserve(static_cast<size_t>(m) * kFeatDim);
            for (int32_t i = 0; i < m; ++i) {
                const float* p = fbank_->GetFrame(frame_index + i);
                out.insert(out.end(), p, p + kFeatDim);
            }
            return out;
        }

        std::vector<float> ApplyLFR(const std::vector<float>& in) {
            int32_t in_frames = static_cast<int32_t>(in.size()) / kFeatDim;
            int32_t out_frames = (in_frames - lfr_window_size_) / lfr_window_shift_ + 1;
            int32_t out_dim = kFeatDim * lfr_window_size_;
            std::vector<float> out(static_cast<size_t>(out_frames) * out_dim);
            const float* p_in = in.data();
            float* p_out = out.data();
            for (int32_t i = 0; i < out_frames; ++i) {
                std::copy(p_in, p_in + out_dim, p_out);
                p_out += out_dim;
                p_in += lfr_window_shift_ * kFeatDim;
            }
            return out;
        }

        void ApplyCMVN(std::vector<float>& v) {
            int32_t dim = static_cast<int32_t>(neg_mean_.size());
            int32_t frames = static_cast<int32_t>(v.size()) / dim;
            for (int32_t t = 0; t < frames; ++t) {
                float* p = v.data() + static_cast<size_t>(t) * dim;
                for (int32_t d = 0; d < dim; ++d) {
                    p[d] = (p[d] + neg_mean_[d]) * inv_stddev_[d];
                }
            }
        }

        void PositionalEncoding(std::vector<float>& v, int32_t t_offset) {
            int32_t feat_dim = kFeatDim * lfr_window_size_;
            int32_t T = static_cast<int32_t>(v.size()) / feat_dim;
            constexpr float kScale = -0.03301197265941284f;
            for (int32_t t = 0; t < T; ++t) {
                float* p = v.data() + static_cast<size_t>(t) * feat_dim;
                int32_t offset = t + 1 + t_offset;
                for (int32_t d = 0; d < feat_dim / 2; ++d) {
                    float inv = static_cast<float>(offset) * std::exp(d * kScale);
                    p[d] += std::sin(inv);
                    p[d + feat_dim / 2] += std::cos(inv);
                }
            }
        }

        bool RunDecoder(std::vector<Ort::Value>& enc_out,
                        std::vector<int32_t>& new_tokens,
                        float& confidence, bool& out_final) {
            const float* p_enc = enc_out[0].GetTensorData<float>();
            auto enc_shape = enc_out[0].GetTensorTypeAndShapeInfo().GetShape();
            int32_t enc_frames = static_cast<int32_t>(enc_shape[1]);
            int32_t hidden = static_cast<int32_t>(enc_shape[2]);

            float* p_alpha = enc_out[2].GetTensorMutableData<float>();
            int32_t a_frames = static_cast<int32_t>(
                enc_out[2].GetTensorTypeAndShapeInfo().GetShape()[1]);
            std::fill(p_alpha, p_alpha + std::min<int32_t>(kLeftChunk, a_frames), 0.0f);
            if (a_frames > kRightChunk) {
                std::fill(p_alpha + a_frames - kRightChunk, p_alpha + a_frames, 0.0f);
            }

            if (initial_hidden_.empty()) initial_hidden_.resize(static_cast<size_t>(hidden));
            std::vector<float> acoustic;
            acoustic.reserve(static_cast<size_t>(a_frames) * hidden);

            float integrate = alpha_cache_;
            for (int32_t i = 0; i < a_frames; ++i) {
                const float* r = p_enc + static_cast<size_t>(i) * hidden;
                float a = p_alpha[i];
                if (integrate + a < threshold_) {
                    integrate += a;
                    for (int32_t d = 0; d < hidden; ++d)
                        initial_hidden_[d] += a * r[d];
                    continue;
                }
                float need = threshold_ - integrate;
                fired_alpha_ += need;
                for (int32_t d = 0; d < hidden; ++d)
                    initial_hidden_[d] += need * r[d];
                acoustic.insert(acoustic.end(), initial_hidden_.begin(), initial_hidden_.end());
                integrate += a - threshold_;
                for (int32_t d = 0; d < hidden; ++d)
                    initial_hidden_[d] = integrate * r[d];
            }
            alpha_cache_ = integrate;

            if (acoustic.empty()) return false;

            int32_t num_tokens = static_cast<int32_t>(acoustic.size()) / hidden;
            auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

            // 首次：初始化 decoder 每 block 的状态缓存（零），对齐 sherpa
            if (states_.empty()) {
                std::array<int64_t, 3> shp{1, encoder_output_size_,
                                           decoder_kernel_size_ - 1};
                int32_t nbytes = static_cast<int32_t>(shp[0] * shp[1] * shp[2]);
                Ort::AllocatorWithDefaultOptions alloc2;
                states_.reserve(static_cast<size_t>(decoder_num_blocks_));
                for (int32_t b = 0; b < decoder_num_blocks_; ++b) {
                    Ort::Value st = Ort::Value::CreateTensor<float>(
                        alloc2, shp.data(), shp.size());
                    memset(st.GetTensorMutableData<float>(), 0,
                           sizeof(float) * nbytes);
                    states_.push_back(std::move(st));
                }
            }

            std::array<int64_t, 3> ac_shape{1, num_tokens, hidden};
            Ort::Value ac = Ort::Value::CreateTensor(
                mem, acoustic.data(), acoustic.size(), ac_shape.data(), ac_shape.size());
            std::array<int64_t, 1> ac_len_shape{1};
            Ort::Value ac_len = Ort::Value::CreateTensor(
                mem, &num_tokens, 1, ac_len_shape.data(), ac_len_shape.size());

            // decoder 输入按顺序：encoder_out, encoder_out_len, acoustic_embedding,
            // acoustic_embedding_length, states...
            std::vector<Ort::Value> dec_in;
            dec_in.reserve(4 + states_.size());
            std::array<int64_t, 3> enc_shape2{0, 0, 0};
            std::copy_n(enc_shape.begin(),
                        std::min<size_t>(enc_shape.size(), 3), enc_shape2.begin());
            Ort::Value cout = Ort::Value::CreateTensor(
                mem, const_cast<float*>(p_enc), Numel(enc_shape),
                enc_shape2.data(), enc_shape2.size());
            std::array<int64_t, 1> len_shape{1};
            int32_t len_val = enc_frames;
            Ort::Value cout_len = Ort::Value::CreateTensor(
                mem, &len_val, 1, len_shape.data(), len_shape.size());
            dec_in.push_back(std::move(cout));
            dec_in.push_back(std::move(cout_len));
            dec_in.push_back(std::move(ac));
            dec_in.push_back(std::move(ac_len));
            for (auto& s : states_) dec_in.push_back(std::move(s));

            std::vector<Ort::Value> dec_out = decoder_sess_->Run(
                {}, decoder_in_names_.data(), dec_in.data(), dec_in.size(),
                decoder_out_names_.data(), decoder_out_names_.size());

            states_.clear();
            states_.reserve(static_cast<size_t>(decoder_num_blocks_));
            for (size_t i = 2; i < dec_out.size(); ++i) {
                states_.push_back(std::move(dec_out[i]));
            }

            const int64_t* sample_ids = dec_out[1].GetTensorData<int64_t>();
            new_tokens.clear();
            for (int32_t i = 0; i < num_tokens; ++i) {
                int32_t t = static_cast<int32_t>(sample_ids[i]);
                if (t == 0) continue;
                all_tokens_.push_back(t);
                new_tokens.push_back(t);
            }

            float denom = fired_alpha_ + alpha_cache_;
            running_confidence_ = (denom > 0.f) ? (fired_alpha_ / denom) : 0.f;
            confidence = running_confidence_;
            out_final = short_final_done_;
            return true;
        }

        static inline size_t Numel(const std::vector<int64_t>& shape) {
            size_t n = 1;
            for (auto d : shape) n *= static_cast<size_t>(std::max<int64_t>(d, 1));
            return n;
        }

        const std::vector<int32_t>& all_tokens() const { return all_tokens_; }
        const BasicSymbolTable& table() const { return token_table_; }
        int32_t vocab_size() const { return vocab_size_; }
        float threshold() const { return threshold_; }

    private:
        int32_t sample_rate_;
        float threshold_;
        Ort::Env env_;
        Ort::SessionOptions sess_opts_;
        std::unique_ptr<Ort::Session> encoder_sess_;
        std::unique_ptr<Ort::Session> decoder_sess_;
        bool initialized_ = false;
        bool short_final_done_ = false;

        int32_t vocab_size_ = 0;
        int32_t lfr_window_size_ = 7;
        int32_t lfr_window_shift_ = 6;
        int32_t encoder_output_size_ = 0;
        int32_t decoder_num_blocks_ = 1;
        int32_t decoder_kernel_size_ = 3;
        std::vector<float> neg_mean_;
        std::vector<float> inv_stddev_;

        std::unique_ptr<knf::OnlineFbank> fbank_;

        int32_t frames_processed_ = 0;
        std::vector<float> feat_cache_;
        std::vector<float> initial_hidden_;
        std::vector<Ort::Value> states_;
        std::vector<int32_t> all_tokens_;
        float alpha_cache_ = 0.f;
        float fired_alpha_ = 0.f;
        float running_confidence_ = 0.f;

        BasicSymbolTable token_table_;

        std::vector<const char*> encoder_in_names_;
        std::vector<const char*> encoder_out_names_;
        std::vector<const char*> decoder_in_names_;
        std::vector<const char*> decoder_out_names_;

        std::vector<std::string> encoder_in_strings_;
        std::vector<std::string> encoder_out_strings_;
        std::vector<std::string> decoder_in_strings_;
        std::vector<std::string> decoder_out_strings_;
    };

    ParaformerStreamingAsr::ParaformerStreamingAsr(
            const std::string& encoder_onnx,
            const std::string& decoder_onnx,
            const std::string& tokens_txt,
            int32_t sample_rate,
            int32_t num_threads,
            float threshold)
        : impl_(std::make_unique<Impl>(encoder_onnx, decoder_onnx, tokens_txt,
                                       sample_rate, num_threads, threshold)) {}

    ParaformerStreamingAsr::~ParaformerStreamingAsr() = default;

    bool ParaformerStreamingAsr::is_initialized() const { return impl_->is_initialized(); }

    void ParaformerStreamingAsr::reset() { impl_->reset(); }

    void ParaformerStreamingAsr::accept_waveform(const std::vector<float>& samples) {
        impl_->accept_waveform(samples);
    }

    void ParaformerStreamingAsr::input_finished() { impl_->input_finished(); }

    bool ParaformerStreamingAsr::decode(bool is_final, StreamingAsrResult* result) {
        std::vector<int32_t> new_tokens;
        float conf = 0.f;
        bool out_final = false;
        bool ok = impl_->DecodeStep(is_final, new_tokens, conf, out_final);
        if (result) {
            result->tokens = new_tokens;
            result->confidence = conf;
            result->is_final = out_final;
            result->text = TokensToText(new_tokens, impl_->table());
            if (result->text == " ") result->text.clear();
        }
        return ok;
    }

    std::string ParaformerStreamingAsr::text() const {
        return TokensToText(impl_->all_tokens(), impl_->table());
    }

    int32_t ParaformerStreamingAsr::vocab_size() const { return impl_->vocab_size(); }

    float ParaformerStreamingAsr::threshold() const { return impl_->threshold(); }

} // namespace modeldeploy::audio::asr
