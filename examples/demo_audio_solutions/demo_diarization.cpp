// ModelDeploy demo_diarization —— 说话人分段演示（无权重，VAD 切段）。
// Usage: demo_diarization [input.wav]
//   合成/读取音频 -> SpeakerDiarization::run -> 打印 (start_ms, end_ms[, speaker_id])。
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "audio/solutions/speaker_diarization.h"
#include "audio/tools/wav_io.h"

static std::vector<float> synthesize_multi_utterance(int sr) {
    // 模拟两段语音，中间隔 500ms 静音。
    std::vector<float> audio;
    auto tone = [&](float freq, int ms) {
        for (int i = 0; i < sr * ms / 1000; ++i) {
            float t = (float)i / sr;
            float env = (i < 200) ? (float)i / 200.0f : 1.0f;
            audio.push_back((float)(0.3 * env * std::sin(2 * 3.14159265 * freq * t)));
        }
    };
    tone(220.0f, 500);
    audio.insert(audio.end(), sr / 2, 0.0f);  // 500ms 静音
    tone(330.0f, 500);
    return audio;
}

int main(int argc, char** argv) {
    const int sr = 16000;
    std::vector<float> audio;
    if (argc > 1) {
        modeldeploy::audio::tool::WavData wd;
        if (!modeldeploy::audio::tool::read_wav(argv[1], &wd) || wd.samples.empty())
            { printf("cannot read %s; fallback to synthesis\n", argv[1]); audio = synthesize_multi_utterance(sr); }
        else
            audio = wd.samples;
    } else {
        audio = synthesize_multi_utterance(sr);
    }

    modeldeploy::audio::solution::SpeakerDiarization dia;
    std::vector<modeldeploy::audio::solution::Segment> segs;
    if (!dia.run(audio, &segs)) { printf("diarization failed\n"); return 1; }
    printf("diarization: %zu segment(s)\n", segs.size());
    for (auto& s : segs)
        printf("  [%d ms, %d ms] speaker=%d\n", s.start_ms, s.end_ms, s.speaker_id);
    return 0;
}
