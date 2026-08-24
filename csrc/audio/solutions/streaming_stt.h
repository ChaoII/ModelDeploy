#pragma once
#include <functional>
#include <string>
#include <vector>
#include "core/md_decl.h"
#include "audio/tools/vad_segment.h"
#include "audio/solutions/solution_base.h"
namespace modeldeploy::audio { namespace asr { class SenseVoice; } }
namespace modeldeploy::audio::solution {
class MODELDEPLOY_CXX_EXPORT StreamingSTT : public SolutionBase {
public:
    // 转写回调：输入一个 VAD 切出的语音段（16k float PCM）与采样率，返回识别文本。
    using TranscribeFn = std::function<std::string(const std::vector<float>&, int sr)>;

    explicit StreamingSTT(std::function<void(const std::string&)> on_text = nullptr,
                          TranscribeFn transcribe = nullptr,
                          int sample_rate = 16000);
    void push(const std::vector<float>& data, int sr);
    void set_on_text(std::function<void(const std::string&)> cb) { on_text_ = std::move(cb); }
    void set_transcribe(TranscribeFn cb) { transcribe_ = std::move(cb); }
    // 消费当前已就绪的语音段并触发 on_text
    void run_once();
    // 流结束：强制闭合并转写末尾未成段的语音
    void finish();

    // 把真实 SenseVoice 包成 TranscribeFn（data 为 16k float PCM）
    static TranscribeFn sense_voice(asr::SenseVoice& model);

private:
    std::function<void(const std::string&)> on_text_;
    TranscribeFn transcribe_;
    tool::VadSegment vad_;
    int sr_;
};
} // namespace modeldeploy::audio::solution
