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

#include "runtime/runtime.h"
#include "runtime/runtime_option.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

namespace modeldeploy::audio::asr {

    namespace {

        // tokens.txt: 每行 "<symbol> <id>"
        class BasicSymbolTable {
        public:
            bool Load(const std::string& path) {
                std::ifstream in(path);
                if (!in) return false;
                std::string line;
                while (std::getline(in, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty()) continue;
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

        // 自有 CPU Tensor + memcpy（对齐 SDK 惯例，避免 set_data 模板未实例化问题）
        inline Tensor MakeFloatTensor(const std::vector<float>& data,
                                      const std::vector<int64_t>& shape) {
            Tensor t(shape, DataType::FP32, Device::CPU);
            if (!data.empty()) {
                std::memcpy(t.data(), data.data(), data.size() * sizeof(float));
            }
            return t;
        }

        inline Tensor MakeInt32Tensor(int32_t value) {
            Tensor t({1}, DataType::INT32, Device::CPU);
            std::memcpy(t.data(), &value, sizeof(int32_t));
            return t;
        }

        constexpr int64_t kMelDim = 80;

    } // namespace

    class ParaformerStreamingAsr::Impl {
    public:
        static constexpr int32_t kChunkSize = 61;
        static constexpr int32_t kLeftChunk = 5;
        static constexpr int32_t kRightChunk = 3;

        Impl(const std::string& encoder_onnx,
             const std::string& decoder_onnx,
             const std::string& tokens_txt,
             int32_t sample_rate,
             int32_t num_threads,
             float threshold)
            : sample_rate_(sample_rate), threshold_(threshold) {
            RuntimeOption eopt, dopt;
            eopt.set_model_path(encoder_onnx);
            dopt.set_model_path(decoder_onnx);
            if (num_threads > 0) {
                eopt.set_cpu_thread_num(num_threads);
                dopt.set_cpu_thread_num(num_threads);
            }
            // 走 SDK 统一 Runtime：多后端可插拔（当前 ORT CPU）
            // 注意：Runtime::is_initialized() 对 ORT 不可靠（OrtBackend 遮蔽了
            // BaseBackend::initialized_），以 init() 返回值判定。
            const bool enc_ok = encoder_rt_.init(eopt);
            const bool dec_ok = decoder_rt_.init(dopt);
            if (!enc_ok || !dec_ok) {
                initialized_ = false;
                return;
            }

            initialized_ = ReadMetadata() && token_table_.Load(tokens_txt);
            if (initialized_) ResetFbank();
        }

        ~Impl() = default;

        [[nodiscard]] bool is_initialized() const { return initialized_; }

        // ---- preprocess 阶段：在线 FBank ----
        void ResetFbank() {
            knf::FbankOptions opts;
            opts.frame_opts.dither = 0.0f;
            opts.frame_opts.snip_edges = true;
            opts.frame_opts.window_type = "hamming";
            opts.frame_opts.samp_freq = static_cast<float>(sample_rate_);
            opts.frame_opts.frame_shift_ms = 10.0f;
            opts.frame_opts.frame_length_ms = 25.0f;
            opts.mel_opts.num_bins = kMelDim;
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
            short_final_done_ = false;
        }

        void accept_waveform(const std::vector<float>& samples) {
            if (!initialized_ || samples.empty()) return;
            fbank_->AcceptWaveform(static_cast<float>(sample_rate_),
                                   samples.data(),
                                   static_cast<int32_t>(samples.size()));
        }

        void input_finished() { if (fbank_) fbank_->InputFinished(); }

        // ---- 解码总入口 ----
        bool DecodeStep(bool is_final, std::vector<int32_t>& new_tokens,
                        float& confidence, bool& out_final) {
            if (!initialized_) return false;
            // 语义：调用者标记 is_final 即视为段结束（即使末块不解码出新 token）
            if (is_final) out_final = true;
            if (is_final && short_final_done_) return false;

            const int32_t start_processed = frames_processed_;
            const int32_t available = fbank_->NumFramesReady() - frames_processed_;
            if (!is_final && available < kChunkSize) return false;
            const bool short_final = is_final && available < kChunkSize;

            // 取帧 + LFR/CMVN/PositionalEncoding → 输入 Tensor
            std::vector<float> lfr = preprocess_feature(short_final, available);
            if (lfr.empty()) return false;

            const int32_t num_frames = static_cast<int32_t>(lfr.size()) /
                                       static_cast<int32_t>(neg_mean_.size());
            Tensor features = MakeFloatTensor(
                lfr, {1, num_frames, static_cast<int64_t>(neg_mean_.size())});
            Tensor features_len = MakeInt32Tensor(num_frames);

            // encode + CIF 聚合（把 candidate 合并成 acoustic_embedding）
            Tensor encoder_out, encoder_out_len, alpha;
            if (!RunEncoder(features, features_len, &encoder_out, &encoder_out_len, &alpha))
                return false;

            std::vector<float> acoustic;
            if (!CifAggregate(alpha, encoder_out, &acoustic)) return false;

            const int32_t num_tokens =
                static_cast<int32_t>(acoustic.size()) / static_cast<int32_t>(encoder_out.shape()[2]);
            Tensor ac = MakeFloatTensor(acoustic,
                                        {1, num_tokens, encoder_out.shape()[2]});
            Tensor ac_len = MakeInt32Tensor(num_tokens);

            // decode：encoder_out / len + acoustic + states
            if (!RunDecoder(encoder_out, encoder_out_len, ac, ac_len, new_tokens))
                return false;

            confidence = running_confidence_;
            return true;
        }

        // 取帧 + LFR + CMVN + PositionalEncoding（preprocess 后半段）
        std::vector<float> preprocess_feature(bool short_final, int32_t available) {
            std::vector<float> frames = GetFrames(
                frames_processed_, short_final ? available : kChunkSize);
            if (frames.empty() && !short_final) return {};
            if (short_final) {
                frames.resize(static_cast<size_t>(kChunkSize) * kMelDim, 0.0f);
                frames_processed_ += available;
                short_final_done_ = true;
            } else {
                frames_processed_ += kChunkSize - 1;
            }

            const int32_t t_offset = (frames_processed_ - (short_final ? available
                                                                       : (kChunkSize - 1))) /
                                     lfr_window_shift_;
            std::vector<float> lfr = ApplyLFR(frames);
            ApplyCMVN(lfr);
            PositionalEncoding(lfr, t_offset);

            const int32_t feat_dim = static_cast<int32_t>(neg_mean_.size());
            if (feat_cache_.empty()) {
                feat_cache_.resize(static_cast<size_t>(kLeftChunk + kRightChunk) * feat_dim, 0.0f);
            }
            lfr.insert(lfr.begin(), feat_cache_.begin(), feat_cache_.end());
            std::copy(lfr.end() - static_cast<ptrdiff_t>(feat_cache_.size()),
                      lfr.end(), feat_cache_.begin());
            return lfr;
        }

        std::vector<float> GetFrames(int32_t frame_index, int32_t n) {
            std::vector<float> out;
            const int32_t ready = fbank_->NumFramesReady();
            if (frame_index >= ready) return out;
            int32_t m = std::min<int32_t>(n, ready - frame_index);
            out.reserve(static_cast<size_t>(m) * kMelDim);
            for (int32_t i = 0; i < m; ++i) {
                const float* p = fbank_->GetFrame(frame_index + i);
                out.insert(out.end(), p, p + kMelDim);
            }
            return out;
        }

        std::vector<float> ApplyLFR(const std::vector<float>& in) {
            int32_t in_frames = static_cast<int32_t>(in.size()) / kMelDim;
            int32_t out_frames = (in_frames - lfr_window_size_) / lfr_window_shift_ + 1;
            int32_t out_dim = kMelDim * lfr_window_size_;
            std::vector<float> out(static_cast<size_t>(out_frames) * out_dim);
            const float* p_in = in.data();
            float* p_out = out.data();
            for (int32_t i = 0; i < out_frames; ++i) {
                std::copy(p_in, p_in + out_dim, p_out);
                p_out += out_dim;
                p_in += lfr_window_shift_ * kMelDim;
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
            int32_t feat_dim = kMelDim * lfr_window_size_;
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

        // ---- encode：走 encoder Runtime ----
        bool RunEncoder(const Tensor& features, const Tensor& features_len,
                        Tensor* encoder_out, Tensor* encoder_out_len, Tensor* alpha) {
            std::vector<Tensor> in = {features, features_len};
            std::vector<Tensor> out;
            if (!encoder_rt_.infer(in, &out) || out.size() < 3) return false;
            *encoder_out = out[0];
            *encoder_out_len = out[1];
            *alpha = out[2];
            return true;
        }

        // CIF 搜索：把 encoder_out 按 alpha 聚合为 acoustic_embedding
        bool CifAggregate(Tensor& alpha, const Tensor& encoder_out,
                          std::vector<float>* acoustic) {
            const int32_t hidden = static_cast<int32_t>(encoder_out.shape()[2]);
            const int32_t a_frames = static_cast<int32_t>(alpha.shape()[1]);
            float* p_alpha = static_cast<float*>(alpha.data());
            std::fill(p_alpha, p_alpha + std::min<int32_t>(kLeftChunk, a_frames), 0.0f);
            if (a_frames > kRightChunk) {
                std::fill(p_alpha + a_frames - kRightChunk, p_alpha + a_frames, 0.0f);
            }
            const float* p_enc = static_cast<const float*>(encoder_out.data());

            if (initial_hidden_.empty()) initial_hidden_.resize(static_cast<size_t>(hidden));
            std::vector<float>& ac = *acoustic;
            ac.clear();
            ac.reserve(static_cast<size_t>(a_frames) * hidden);

            float integrate = alpha_cache_;
            for (int32_t i = 0; i < a_frames; ++i) {
                const float* r = p_enc + static_cast<size_t>(i) * hidden;
                float a = p_alpha[i];
                if (integrate + a < threshold_) {
                    integrate += a;
                    for (int32_t d = 0; d < hidden; ++d) initial_hidden_[d] += a * r[d];
                    continue;
                }
                float need = threshold_ - integrate;
                fired_alpha_ += need;
                for (int32_t d = 0; d < hidden; ++d) initial_hidden_[d] += need * r[d];
                ac.insert(ac.end(), initial_hidden_.begin(), initial_hidden_.end());
                integrate += a - threshold_;
                for (int32_t d = 0; d < hidden; ++d) initial_hidden_[d] = integrate * r[d];
            }
            alpha_cache_ = integrate;
            return !ac.empty();
        }

        // ---- decode：走 decoder Runtime，状态 Tensor 往返 ----
        bool RunDecoder(const Tensor& encoder_out, const Tensor& encoder_out_len,
                        const Tensor& acoustic, const Tensor& acoustic_len,
                        std::vector<int32_t>& new_tokens) {
            const int32_t hidden = static_cast<int32_t>(encoder_out.shape()[2]);
            const int32_t num_tokens = static_cast<int32_t>(acoustic.shape()[1]);

            // 首次：初始化每 block 状态缓存（零）
            if (states_.empty()) {
                std::vector<int64_t> shp{1, encoder_output_size_,
                                         decoder_kernel_size_ - 1};
                states_.reserve(static_cast<size_t>(decoder_num_blocks_));
                for (int32_t b = 0; b < decoder_num_blocks_; ++b) {
                    Tensor st(shp, DataType::FP32, Device::CPU);
                    std::memset(st.data(), 0, st.byte_size());
                    states_.push_back(std::move(st));
                }
            }

            std::vector<Tensor> in;
            in.reserve(4 + states_.size());
            in.push_back(encoder_out);
            in.push_back(encoder_out_len);
            in.push_back(acoustic);
            in.push_back(acoustic_len);
            for (const auto& st : states_) in.push_back(st);

            std::vector<Tensor> out;
            if (!decoder_rt_.infer(in, &out)) return false;

            // 解析输出：末尾 decoder_num_blocks_ 个为状态，前一(int64)为 sample_ids
            const size_t n_out = out.size();
            const size_t n_state = static_cast<size_t>(decoder_num_blocks_);
            if (n_out < 2 + n_state) return false;

            Tensor& sample_ids_t = out[n_out - n_state - 1];
            new_tokens.clear();
            if (sample_ids_t.dtype() == DataType::INT64) {
                const int64_t* ids = static_cast<const int64_t*>(sample_ids_t.data());
                for (int32_t i = 0; i < num_tokens; ++i) {
                    int32_t t = static_cast<int32_t>(ids[i]);
                    if (t == 0) continue;
                    all_tokens_.push_back(t);
                    new_tokens.push_back(t);
                }
            }

            states_.clear();
            for (size_t i = n_out - n_state; i < n_out; ++i) {
                states_.push_back(std::move(out[i]));
            }

            float denom = fired_alpha_ + alpha_cache_;
            running_confidence_ = (denom > 0.f) ? (fired_alpha_ / denom) : 0.f;
            return true;
        }

        static inline size_t NumelShape(const std::vector<int64_t>& s) {
            size_t n = 1;
            for (auto d : s) n *= static_cast<size_t>(d);
            return n;
        }

        bool ReadMetadata() {
            auto map = encoder_rt_.get_custom_meta_data();
            auto read_int = [&map](const char* key, int32_t* out) -> bool {
                auto it = map.find(key);
                if (it == map.end()) return false;
                try { *out = std::stoi(it->second); } catch (...) { return false; }
                return true;
            };
            auto read_vec = [&map](const char* key, std::vector<float>* out) -> bool {
                auto it = map.find(key);
                if (it == map.end()) return false;
                std::vector<float> v;
                std::string s = it->second;
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
            if (!read_int("encoder_output_size", &encoder_output_size_)) return false;
            read_int("decoder_num_blocks", &decoder_num_blocks_);
            read_int("decoder_kernel_size", &decoder_kernel_size_);
            if (!read_vec("neg_mean", &neg_mean_)) return false;
            if (!read_vec("inv_stddev", &inv_stddev_)) return false;
            float scale = std::sqrt(static_cast<float>(encoder_output_size_));
            for (auto& f : inv_stddev_) f *= scale;
            if (decoder_num_blocks_ <= 0 || decoder_kernel_size_ <= 1) return false;
            return true;
        }

        const std::vector<int32_t>& all_tokens() const { return all_tokens_; }
        const BasicSymbolTable& table() const { return token_table_; }
        int32_t vocab_size() const { return vocab_size_; }
        float threshold() const { return threshold_; }

    private:
        int32_t sample_rate_;
        float threshold_;

        Runtime encoder_rt_;
        Runtime decoder_rt_;

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
        std::vector<Tensor> states_;
        std::vector<int32_t> all_tokens_;
        float alpha_cache_ = 0.f;
        float fired_alpha_ = 0.f;
        float running_confidence_ = 0.f;

        BasicSymbolTable token_table_;
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
