// ModelDeploy demo_stream_stt —— 流式语音识别演示（无权重，VAD 分段骨架）。
// Usage: demo_stream_stt
//   StreamingSTT: push 音频 + set_on_text 回调 + run_once；缺 SenseVoice 权重仅做 VAD 分段。
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "audio/solutions/streaming_stt.h"

int main() {
    const int sr = 16000;
    modeldeploy::audio::solution::StreamingSTT stt([](const std::string& text) {
        printf("[STT] %s\n", text.empty() ? "(vad segments only)" : text.c_str());
    });

    // 合成两段语音（含静音）。
    std::vector<float> audio;
    auto tone = [&](float freq, int ms) {
        for (int i = 0; i < sr * ms / 1000; ++i) {
            float t = (float)i / sr;
            audio.push_back((float)(0.3 * std::sin(2 * 3.14159265 * freq * t)));
        }
    };
    tone(220.0f, 400);
    audio.insert(audio.end(), sr / 2, 0.0f);
    tone(330.0f, 400);

    // 分块推送（模拟流式输入）。
    const size_t chunk = sr / 4;
    for (size_t off = 0; off < audio.size(); off += chunk) {
        size_t n = std::min(chunk, audio.size() - off);
        stt.push(std::vector<float>(audio.begin() + off, audio.begin() + off + n), sr);
        stt.run_once();
    }
    stt.run_once();
    printf("streaming_stt demo done\n");
    return 0;
}
