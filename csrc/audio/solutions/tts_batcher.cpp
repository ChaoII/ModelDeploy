#include "audio/solutions/tts_batcher.h"
#include "audio/tts/kokoro.h"
#include <algorithm>
#include <cstdint>

namespace modeldeploy::audio::solution {
TTSBatcher::TTSBatcher(SynthFn synth) : synth_(std::move(synth)) {}

void TTSBatcher::enqueue(const std::vector<std::string>& texts) {
    queue_.insert(queue_.end(), texts.begin(), texts.end());
}

void TTSBatcher::enqueue(const std::string& text) { queue_.push_back(text); }

std::vector<std::vector<float>> TTSBatcher::dequeue_all() {
    std::vector<std::vector<float>> out;
    out.reserve(queue_.size());
    for (const auto& t : queue_) {
        if (synth_) {
            out.push_back(synth_(t));
        } else {
            out.emplace_back();
        }
    }
    queue_.clear();
    return out;
}

TTSBatcher::SynthFn TTSBatcher::kokoro_synth(tts::Kokoro& model, const std::string& voice, float speed) {
    return [&model, voice, speed](const std::string& text) {
        return synthesize_text(model, voice, speed, text);
    };
}

// 按 UTF-8 字符计数，把 text 切成长度不超过 max_chars 的块，尽量在标点/空白处断开。
std::vector<std::string> TTSBatcher::split_for_synthesis(const std::string& text, int max_chars) {
    std::vector<std::string> parts;
    if (text.empty() || max_chars <= 0) return {text};

    // 收集每个 UTF-8 字符的 [起始偏移, 长度]
    std::vector<std::pair<size_t, size_t>> chars;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = (unsigned char)text[i];
        size_t len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        chars.emplace_back(i, len);
        i += len;
    }
    if ((int)chars.size() <= max_chars) return {text};

    auto is_break = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '.' || c == ',' || c == ';' ||
               c == '!' || c == '?' || c == '。' || c == '，' || c == '；' ||
               c == '！' || c == '？' || c == '、' || c == '\n' || c == '\r';
    };

    size_t start = 0;
    while (start < chars.size()) {
        size_t end = std::min(start + (size_t)max_chars, chars.size());
        if (end < chars.size()) {
            // 尝试从 [start, end) 内最后一个可断点截断，否则在 end 硬切
            size_t cut = end;
            for (size_t k = end; k > start; --k) {
                if (is_break((unsigned char)text[chars[k - 1].first])) { cut = k; break; }
            }
            end = (cut > start) ? cut : end;
        }
        const size_t b = chars[start].first;
        const size_t e = (end < chars.size()) ? chars[end].first : text.size();
        parts.push_back(text.substr(b, e - b));
        start = end;
    }
    return parts;
}

std::vector<float> TTSBatcher::synthesize_text(tts::Kokoro& model, const std::string& voice,
                                               float speed, const std::string& text, int max_chars) {
    std::vector<float> out;
    if (!model.is_initialized()) return out;
    auto parts = split_for_synthesis(text, max_chars);
    for (const auto& p : parts) {
        std::vector<float> audio;
        if (model.predict(p, voice, speed, &audio)) {
            out.insert(out.end(), audio.begin(), audio.end());
        }
    }
    return out;
}
} // namespace modeldeploy::audio::solution
