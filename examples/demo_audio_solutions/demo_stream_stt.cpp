// ModelDeploy demo_stream_stt —— 流式语音识别。
// 用法: demo_stream_stt [model.onnx] [tokens.txt] [in.wav]
//   提供 3 个参数时加载真实 SenseVoice，对 wav 分块推送 VAD 分段识别；
//   缺省时用合成音演示 VAD 分段框架（无权重）。
#include <cstdio>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "audio/solutions/streaming_stt.h"
#include "audio/asr/sense_voice.h"
#include "audio/tools/wav_io.h"

int main(int argc, char** argv) {
    const int sr = 16000;
    std::vector<float> audio;
    std::unique_ptr<modeldeploy::audio::asr::SenseVoice> sv;  // 需存活到识别结束（lambda 按引用捕获）

    modeldeploy::audio::solution::StreamingSTT stt(
        [](const std::string& text) {
            printf("[STT] %s\n", text.empty() ? "(vad segments only)" : text.c_str());
        });

    if (argc >= 4) {
        modeldeploy::RuntimeOption opt;
        opt.set_device(modeldeploy::Device::CPU);
        sv = std::make_unique<modeldeploy::audio::asr::SenseVoice>(argv[1], argv[2], opt);
        if (!sv->is_initialized()) {
            printf("SenseVoice init failed; fallback to VAD-only\n");
            sv.reset();
        } else {
            stt.set_transcribe(modeldeploy::audio::solution::StreamingSTT::sense_voice(*sv));
        }
        modeldeploy::audio::tool::WavData wd;
        if (modeldeploy::audio::tool::read_wav(argv[3], &wd) && !wd.samples.empty())
            audio = wd.samples;
        else {
            printf("cannot read %s; fallback to synthesis\n", argv[3]);
        }
    }

    if (audio.empty()) {  // 合成两段带间隙的语音
        auto tone = [&](float freq, int ms) {
            for (int i = 0; i < sr * ms / 1000; ++i) {
                float t = (float)i / sr;
                audio.push_back((float)(0.3 * std::sin(2 * 3.14159265f * freq * t)));
            }
        };
        tone(220.0f, 400);
        audio.insert(audio.end(), sr / 2, 0.0f);
        tone(330.0f, 400);
    }

    const size_t chunk = sr / 4;  // 250ms 一块，模拟流式输入
    for (size_t off = 0; off < audio.size(); off += chunk) {
        size_t n = std::min(chunk, audio.size() - off);
        stt.push(std::vector<float>(audio.begin() + (long)off, audio.begin() + (long)off + (long)n), sr);
        stt.run_once();
    }
    stt.finish();   // 流结束：闭合并转写末尾语音段
    printf("streaming_stt demo done\n");
    return 0;
}
