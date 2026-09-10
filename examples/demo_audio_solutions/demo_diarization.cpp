// ModelDeploy demo_diarization —— 说话人日志（VAD 分段 + 可选的 ECAPA embedding 聚类）。
// 用法: demo_diarization [in.wav] [ecapa.onnx]
//   提供 ecapa.onnx 时对每段取 embedding 并聚类出说话人；缺省仅分段（speaker=-1）。
#include <cstdio>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "audio/solutions/speaker_diarization.h"
#include "audio/speaker_verify/ecapa.h"
#include "audio/tools/wav_io.h"

static std::vector<float> synthesize_multi_utterance(int sr) {
    std::vector<float> audio;
    auto tone = [&](float freq, int ms) {
        for (int i = 0; i < sr * ms / 1000; ++i) {
            float t = (float)i / sr;
            float env = (i < 200) ? (float)i / 200.0f : 1.0f;
            audio.push_back((float)(0.3 * env * std::sin(2 * 3.14159265f * freq * t)));
        }
    };
    tone(220.0f, 500);
    audio.insert(audio.end(), sr / 2, 0.0f);
    tone(330.0f, 500);
    return audio;
}

int main(int argc, char** argv) {
    const int sr = 16000;
    std::vector<float> audio;
    if (argc > 1) {
        modeldeploy::audio::tool::WavData wd;
        if (!modeldeploy::audio::tool::read_wav(argv[1], &wd) || wd.samples.empty()) {
            printf("cannot read %s; fallback to synthesis\n", argv[1]);
            audio = synthesize_multi_utterance(sr);
        } else {
            audio = wd.samples;
        }
    } else {
        audio = synthesize_multi_utterance(sr);
    }

    modeldeploy::audio::solution::SpeakerDiarization::EmbedFn embed;
    std::unique_ptr<modeldeploy::audio::speaker_verify::SpeakerVerify> sv;
    if (argc > 2) {
        modeldeploy::RuntimeOption opt;
        opt.set_device(modeldeploy::Device::CPU);
        sv = std::make_unique<modeldeploy::audio::speaker_verify::SpeakerVerify>(argv[2], opt);
        if (sv->is_initialized()) {
            embed = modeldeploy::audio::solution::SpeakerDiarization::ecapa_embedder(*sv);
        } else {
            printf("SpeakerVerify init failed; VAD-only\n");
        }
    }

    modeldeploy::audio::solution::SpeakerDiarization dia;
    std::vector<modeldeploy::audio::solution::Segment> segs;
    if (!dia.run(audio, &segs, embed)) { printf("diarization failed\n"); return 1; }
    printf("diarization: %zu segment(s)\n", segs.size());
    for (auto& s : segs)
        printf("  [%d ms, %d ms] speaker=%d\n", s.start_ms, s.end_ms, s.speaker_id);
    return 0;
}
