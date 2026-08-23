#include <cstdio>
#include <string>
#include <vector>

#include "runtime/runtime_option.h"
#include "csrc/utils/wave_helper.h"
#include "csrc/audio/speaker_gallery.h"
#include "csrc/audio/speaker_verify/ecapa.h"

// 加载 wav -> float PCM(16k)；坏/缺失 wav 时返回空并打印错误
static std::vector<float> load_wav(const std::string& path, const char* tag) {
    std::vector<float> pcm;
    int32_t sampling_rate = 16000;
    if (!load_wav_file(path.c_str(), &sampling_rate, pcm)) {
        std::cerr << "failed to read wav " << tag << ": " << path << "\n";
        return {};
    }
    if (pcm.empty()) {
        std::cerr << "wav " << tag << " is empty: " << path << "\n";
        return {};
    }
    std::cout << tag << " wav samples=" << pcm.size()
              << " sr=" << sampling_rate << "\n";
    return pcm;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: demo_speaker <model.onnx> <wavA> <wavB>\n";
        std::cerr << "  加载 SpeakerVerify 提取两支语音的说话人 embedding，\n";
        std::cerr << "  enroll(A) 后 match(B) -> (label, score)。支持额外 wav：可从 B 开始继续注册/比对。\n";
        return 1;
    }
    const std::string model_file = argv[1];

    modeldeploy::RuntimeOption option;
    option.use_ort_backend();
    option.use_cpu();

    modeldeploy::audio::speaker_verify::SpeakerVerify model(model_file, option);
    if (!model.is_initialized()) {
        std::cerr << "failed to init SpeakerVerify model: " << model_file << "\n";
        return 1;
    }

    auto embed = [&](const std::vector<float>& pcm, const char* tag) -> std::vector<float> {
        std::vector<float> emb;
        if (!model.predict(pcm, &emb) || emb.empty()) {
            std::cerr << "embedding extraction failed for " << tag << "\n";
            return {};
        }
        std::cout << tag << " embedding dim=" << emb.size() << "\n";
        return emb;
    };

    auto eA = embed(load_wav(argv[2], "A"), "A");
    if (eA.empty()) return 1;

    modeldeploy::audio::SpeakerGallery gallery;
    gallery.enroll("A", eA);
    std::cout << "gallery size=" << gallery.size() << "\n";

    // 可选：B 及其后 wav 均注册进 gallery，再各与 gallery 比对
    std::vector<std::string> others;
    for (int i = 3; i < argc; ++i) others.emplace_back(argv[i]);
    for (size_t i = 0; i < others.size(); ++i) {
        auto e = embed(load_wav(others[i], ("B" + std::to_string(i)).c_str()), ("B" + std::to_string(i)).c_str());
        if (e.empty()) continue;
        auto top = gallery.match(e, 1);
        for (auto& [label, score] : top) {
            std::cout << "match(B" << i << ", 1) -> label=" << label
                      << " score=" << score << "\n";
        }
        gallery.enroll("B" + std::to_string(i), e);
    }
    return 0;
}
