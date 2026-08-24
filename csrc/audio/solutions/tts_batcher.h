#pragma once
#include <functional>
#include <string>
#include <vector>
#include "core/md_decl.h"
#include "audio/solutions/solution_base.h"
namespace modeldeploy::audio { namespace tts { class Kokoro; } }
namespace modeldeploy::audio::solution {
class MODELDEPLOY_CXX_EXPORT TTSBatcher : public SolutionBase {
public:
    // 合成回调：输入一段文本，返回 16k/24k float PCM 音频。
    using SynthFn = std::function<std::vector<float>(const std::string&)>;

    explicit TTSBatcher(SynthFn synth = nullptr);
    void enqueue(const std::vector<std::string>& texts);
    void enqueue(const std::string& text);
    std::vector<std::vector<float>> dequeue_all();
    size_t pending() const { return queue_.size(); }
    void set_synth(SynthFn synth) { synth_ = std::move(synth); }

    // 把真实 Kokoro 包成 SynthFn（model 需已初始化；长文本按句子分块再拼接，
    // 避免超出 Kokoro 最大 token 数被静默截断）。
    static SynthFn kokoro_synth(tts::Kokoro& model, const std::string& voice, float speed = 1.0f);

    // 单段文本合成：按字符上限分块，逐块 predict 并拼接。max_chars<=0 时不切块。
    static std::vector<float> synthesize_text(tts::Kokoro& model, const std::string& voice,
                                              float speed, const std::string& text, int max_chars = 120);
    // 把一段文本按标点/空白切成最长 max_chars（UTF-8 字符）的块。
    static std::vector<std::string> split_for_synthesis(const std::string& text, int max_chars);

private:
    SynthFn synth_;
    std::vector<std::string> queue_;
};
} // namespace modeldeploy::audio::solution
