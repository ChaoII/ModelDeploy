//
// Created by aichao on 2025/5/19.
//

#pragma once


#include "csrc/base_model.h"

namespace modeldeploy::audio::asr {
    // SenseVoice 复任务标签的结构化结果：
    // 模型输出的 <|zh|><|HAPPY|><|Speech|>...<|/Speech|>...<|withitn|> 会被解析成字段，
    // 同时把纯文本独立出来。
    struct MODELDEPLOY_CXX_EXPORT SenseVoiceResult {
        std::string text;      // 纯净识别文本（不含任何标签）
        std::string language;  // 语种标签："zh","en","ja","ko"...（无则空）
        std::string emotion;   // 情感："NEUTRAL","HAPPY","SAD"...（无则空）
        std::string event;     // 事件/类别："Speech","Music","BGM"...（无则空）
        std::string task;      // 任务标签："ASR","AED","SER","nospeech"（无则空）
        bool itn{false};       // 是否为 withitn（模型已做逆文本归一化）
        bool nospeech{false};  // 检测到 <|nospeech|>
    };

    class MODELDEPLOY_CXX_EXPORT SenseVoice : public BaseModel {
    public:
        SenseVoice(const std::string& model_file,
                   const std::string& token_path_str,
                   const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "SenseVoice"; }

        bool predict(const std::vector<float>& data, std::string* result);
        // 结构化版本：额外填 language/emotion/event/task/itn/nospeech
        bool predict(const std::vector<float>& data, SenseVoiceResult* result);

        // 深拷贝：复用已加载的 backend session（不重新加载模型/显存）
        [[nodiscard]] std::unique_ptr<SenseVoice> clone() const;

    protected:
        bool initialize();

        bool preprocess(const std::vector<float>& data, std::vector<Tensor>* outputs);

        bool postprocess(std::vector<Tensor>& infer_result, std::string* result);
        bool postprocess(std::vector<Tensor>& infer_result, SenseVoiceResult* result);

    private:
        // 供 clone() 使用：不触发 initialize
        explicit SenseVoice() = default;
        int32_t window_size_{};
        int32_t window_shift_{};
        int32_t with_itn_{};
        int32_t without_itn_{};

        std::string token_path_str_;
        std::vector<float> neg_mean_;
        std::vector<float> inv_stddev_;
        std::map<std::string, int32_t> lang_id_;
        std::map<std::string, std::string> tokens_;
    };
} // namespace detection
