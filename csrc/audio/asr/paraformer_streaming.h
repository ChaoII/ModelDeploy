//
// Created by aichao on 2025/5/19.
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "csrc/base_model.h"

namespace knf {
    // OnlineFbank 是 OnlineGenericBaseFeature<FbankComputer> 的别名（见 kaldi-native-fbank
    // csrc/online-feature.h），别名无法前向声明，改为前向声明其底层模板与参数类型。
    template <typename Computer>
    class OnlineGenericBaseFeature;
    class FbankComputer;
}

namespace modeldeploy::audio::asr {

    // tokens.txt 符号表：每行 "<symbol> <id>"。Load() 实现在 .cpp。
    class MODELDEPLOY_CXX_EXPORT BasicSymbolTable {
    public:
        bool Load(const std::string& path);

        const std::string& operator[](int32_t id) const {
            static const std::string kEmpty;
            auto it = syms_.find(id);
            return it == syms_.end() ? kEmpty : it->second;
        }

    private:
        std::map<int32_t, std::string> syms_;
    };

    // 流式 ASR 的单步结果
    struct MODELDEPLOY_CXX_EXPORT StreamingAsrResult {
        std::string text;            // 该步新增文本（已做 BPE/@@ 解码）
        std::vector<int32_t> tokens; // 该步新增 token id（不含 <blank>=0）
        float confidence{0.f};       // 0..1 解码置信度代理（重连续 alpha 质量）
        bool is_final{false};        // 是否已 flush 到尾块
    };

    // Paraformer-streaming（中英双语 bixes，非 AR）
    // 参考: sherpa-onnx online-recognizer-paraformer-impl / online-paraformer-model
    //
    // 使用约定:
    //   - accept_waveform() 期望 int16 量纲样本（[-32768, 32767]），16k。
    //   - 构造后可先 accept 若干块样本, 反复调用 decode(false) 拿实时部分结果;
    //   - 全部喂完后调用 input_finished()，最后一次 decode(true) flush 尾部短块。
    //
    // 框架对齐: 继承 BaseModel。
    //   - encoder 走 BaseModel::runtime_（统一 Runtime，多后端可插拔，当前仅支持 ORT）;
    //   - decoder 用独立的第二个 Runtime（状态化解码，多后端未实现）。
    //   - preprocess / encode / CIF / decode / postprocess 分层作为模型成员方法，
    //     符合 ModelDeploy 的模型前后处理约定。
    class MODELDEPLOY_CXX_EXPORT ParaformerStreamingAsr : public BaseModel {
    public:
        static constexpr int32_t kChunkSize = 61;
        static constexpr int32_t kLeftChunk = 5;
        static constexpr int32_t kRightChunk = 3;

        ParaformerStreamingAsr(
                const std::string& encoder_onnx,
                const std::string& decoder_onnx,
                const std::string& tokens_txt,
                int32_t sample_rate = 16000,
                int32_t num_threads = 2,
                float threshold = 1.0f);

        // 与默认构造等价的运行时选项重载：调用方可传入全自定义 RuntimeOption（需设置 encoder
        // 模型路径与 use_ort_backend()）。流式 Paraformer 仅支持 ORT——非 ORT 后端会校验失败并明确报错。
        ParaformerStreamingAsr(
                const RuntimeOption& runtime_option,
                const std::string& decoder_onnx,
                const std::string& tokens_txt,
                int32_t sample_rate = 16000,
                int32_t num_threads = 2,
                float threshold = 1.0f);

        ~ParaformerStreamingAsr() override;

        [[nodiscard]] std::string name() const override { return "ParaformerStreamingAsr"; }

        // 复位整个流（特征、decoder 状态、已解 token 全部清空）
        void reset();

        // 接受一段 int16 量纲样本（16k）。可多次调用。
        void accept_waveform(const std::vector<float>& samples);

        // 告诉特征器输入结束（flush 最后几帧）。
        void input_finished();

        // 执行一次解码。is_final=true 时读取剩余短块并置 is_final。
        // 返回 false 表示该步无新 token（result.text 为空，可忽略）。
        bool decode(bool is_final, StreamingAsrResult* result);

        // 累计全文（含历史各步）。
        [[nodiscard]] std::string text() const;

        [[nodiscard]] int32_t vocab_size() const;
        [[nodiscard]] float threshold() const;

    protected:
        // 初始化：encoder 走 BaseModel::init_runtime()，再初始化 decoder Runtime。
        bool initialize();
        bool init_runtime() override;

    private:
        // 两个构造器的公共入口：承载 encoder 的 runtime_option + decoder/tokens 路径，校验后端并初始化。
        void init_from(const RuntimeOption& runtime_option,
                       const std::string& decoder_onnx, const std::string& tokens_txt,
                       int32_t sample_rate, int32_t num_threads, float threshold);
        // ---- preprocess 阶段：在线 FBank ----
        void ResetFbank();

        // ---- 解码总入口 ----
        bool DecodeStep(bool is_final, std::vector<int32_t>& new_tokens,
                        float& confidence, bool& out_final);

        // ---- 特征子步骤（preprocess）----
        std::vector<float> preprocess_feature(bool short_final, int32_t available);
        std::vector<float> GetFrames(int32_t frame_index, int32_t n);
        std::vector<float> ApplyLFR(const std::vector<float>& in);
        void ApplyCMVN(std::vector<float>& v);
        void PositionalEncoding(std::vector<float>& v, int32_t t_offset);

        // ---- encode：走 BaseModel::runtime_ ----
        bool RunEncoder(const Tensor& features, const Tensor& features_len,
                        Tensor* encoder_out, Tensor* encoder_out_len, Tensor* alpha);
        // ---- CIF 搜索：聚合 acoustic_embedding ----
        bool CifAggregate(Tensor& alpha, const Tensor& encoder_out,
                          std::vector<float>* acoustic);
        // ---- decode：走 decoder Runtime，状态 Tensor 往返 ----
        bool RunDecoder(const Tensor& encoder_out, const Tensor& encoder_out_len,
                        const Tensor& acoustic, const Tensor& acoustic_len,
                        std::vector<int32_t>& new_tokens);

        bool ReadMetadata();
        static inline size_t NumelShape(const std::vector<int64_t>& s);

        const std::vector<int32_t>& all_tokens() const { return all_tokens_; }

        // ---- 参数与配置 ----
        int32_t sample_rate_ = 16000;
        int32_t num_threads_ = 2;
        float threshold_ = 1.0f;
        std::string decoder_onnx_;
        std::string tokens_txt_;

        // ---- decoder 用的第二个 Runtime（encoder 走 BaseModel::runtime_）----
        Runtime decoder_rt_;

        // ---- 模型元数据 ----
        bool short_final_done_ = false;
        int32_t vocab_size_ = 0;
        int32_t lfr_window_size_ = 7;
        int32_t lfr_window_shift_ = 6;
        int32_t encoder_output_size_ = 0;
        int32_t decoder_num_blocks_ = 1;
        int32_t decoder_kernel_size_ = 3;
        std::vector<float> neg_mean_;
        std::vector<float> inv_stddev_;

        // ---- 在线特征器 ----
        std::unique_ptr<knf::OnlineGenericBaseFeature<knf::FbankComputer>> fbank_;

        // ---- 流状态 ----
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

} // namespace modeldeploy::audio::asr
