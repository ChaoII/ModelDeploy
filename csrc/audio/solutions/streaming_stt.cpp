#include "audio/solutions/streaming_stt.h"
namespace modeldeploy::audio::solution {
StreamingSTT::StreamingSTT(std::function<void(const std::string&)> on_text)
    : on_text_(std::move(on_text)), vad_(16000) {}
void StreamingSTT::push(const std::vector<float>& data, int sr) { (void)sr; vad_.feed(data); }
void StreamingSTT::run_once() {
    auto segs = vad_.segments();
    if (!segs.empty() && on_text_) on_text_("");
}
} // namespace modeldeploy::audio::solution
