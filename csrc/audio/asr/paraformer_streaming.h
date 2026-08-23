//
// Created by aichao on 2025/5/19.
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "csrc/base_model.h"

namespace modeldeploy::audio::asr {

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
    class MODELDEPLOY_CXX_EXPORT ParaformerStreamingAsr {
    public:
        ParaformerStreamingAsr(
                const std::string& encoder_onnx,
                const std::string& decoder_onnx,
                const std::string& tokens_txt,
                int32_t sample_rate = 16000,
                int32_t num_threads = 2,
                float threshold = 1.0f);

        ~ParaformerStreamingAsr();

        [[nodiscard]] bool is_initialized() const;

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

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace modeldeploy::audio::asr
