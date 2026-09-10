// ModelDeploy demo_tts_batch —— TTS 批处理。
// 用法: demo_tts_batch [kokoro_dir]
//   提供目录时加载真实 Kokoro（model.onnx/tokens.txt/lexicon-*.txt/voices.bin/dict/
//   + text_normalization 目录），对长文本分块合成并拼接；缺省时 mock synth 演示队列。
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "audio/solutions/tts_batcher.h"
#include "audio/tts/kokoro.h"

int main(int argc, char** argv) {
    modeldeploy::audio::solution::TTSBatcher batcher;
    const std::string text =
        "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。这是一段用于演示长文本自动分块合成的句子，"
        "它会超出单个碎片长度，从而验证 TTSBatcher 按标点切块并拼接音频的能力。Hello World!";
    std::unique_ptr<modeldeploy::audio::tts::Kokoro> kokoro;  // 存活到合成结束（lambda 按引用捕获）

    if (argc >= 2 && std::filesystem::exists(argv[1])) {
        namespace fs = std::filesystem;
        const fs::path d = argv[1];
        modeldeploy::RuntimeOption opt;
        opt.set_device(modeldeploy::Device::CPU);
        const std::vector<std::string> lexicons = {
            (d / "lexicon-us-en.txt").string(),
            (d / "lexicon-zh.txt").string(),
            (d / "lexicon-gb-en.txt").string()
        };
        kokoro = std::make_unique<modeldeploy::audio::tts::Kokoro>(
            (d / "model.onnx").string(), (d / "tokens.txt").string(),
            lexicons, (d / "voices.bin").string(),
            (d / "dict").string(), d.string(), opt);
        if (!kokoro->is_initialized()) {
            printf("Kokoro init failed; fallback to mock synth\n");
            kokoro.reset();
        } else {
            batcher.set_synth(modeldeploy::audio::solution::TTSBatcher::kokoro_synth(*kokoro, "zf_001", 1.0f));
        }
    }

    batcher.enqueue({text});
    printf("pending = %zu\n", batcher.pending());

    auto batches = batcher.dequeue_all();
    printf("dequeue_all: %zu batch(es)\n", batches.size());
    int i = 0;
    for (auto& b : batches) printf("  batch[%d] audio = %zu samples\n", i++, b.size());
    return 0;
}
