#include "audio/solutions/streaming_stt.h"
#include "audio/asr/sense_voice.h"
namespace modeldeploy::audio::solution {
StreamingSTT::StreamingSTT(std::function<void(const std::string&)> on_text,
                           TranscribeFn transcribe, int sample_rate)
    : on_text_(std::move(on_text)),
      transcribe_(std::move(transcribe)),
      vad_(sample_rate),
      sr_(sample_rate) {}

void StreamingSTT::push(const std::vector<float>& data, int sr) {
    if (sr != sr_) {
        // VAD 按固定采样率运行；采样率不一致时置空分段结果而不是错误转换。
        (void)sr;
    }
    vad_.feed(data);
}

void StreamingSTT::run_once() {
    auto segs = vad_.consume();
    for (const auto& seg : segs) {
        std::string text;
        if (transcribe_ && !seg.samples.empty()) {
            text = transcribe_(seg.samples, sr_);
        }
        if (on_text_) on_text_(text);
    }
}

void StreamingSTT::finish() {
    vad_.finish();   // 追加静音使末尾语音闭合并被切出
    run_once();
}

StreamingSTT::TranscribeFn StreamingSTT::sense_voice(asr::SenseVoice& model) {
    return [&model](const std::vector<float>& data, int /*sr*/) {
        std::string text;
        if (model.is_initialized() && model.predict(data, &text)) return text;
        return std::string{};
    };
}
} // namespace modeldeploy::audio::solution
