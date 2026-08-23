#include "audio/solutions/tts_batcher.h"
namespace modeldeploy::audio::solution {
TTSBatcher::TTSBatcher(std::function<std::vector<float>(const std::string&)> synth) : synth_(std::move(synth)) {}
void TTSBatcher::enqueue(const std::vector<std::string>& texts) {
    queue_.insert(queue_.end(), texts.begin(), texts.end());
}
std::vector<std::vector<float>> TTSBatcher::dequeue_all() {
    std::vector<std::vector<float>> out;
    for (const auto& t : queue_) {
        if (synth_) out.push_back(synth_(t)); else out.emplace_back();
    }
    queue_.clear();
    return out;
}
} // namespace modeldeploy::audio::solution
