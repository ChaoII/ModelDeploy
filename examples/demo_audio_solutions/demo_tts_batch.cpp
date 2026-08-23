// ModelDeploy demo_tts_batch —— TTS 批处理队列演示（无权重，注入 mock 合成器）。
// Usage: demo_tts_batch
//   TTSBatcher: enqueue 多条文本 -> dequeue_all 打印每段音频长度；缺 Kokoro 权重用 mock synth。
#include <cstdio>
#include <string>
#include <vector>

#include "audio/solutions/tts_batcher.h"

int main() {
    // 无权重 mock 合成器：文本字符数 * 10ms @ 16k。
    modeldeploy::audio::solution::TTSBatcher batcher([](const std::string& text) {
        const int sr = 16000;
        std::vector<float> out(sr / 100, 0.0f);             // 100ms
        out.insert(out.end(), sr * (int)text.size() / 200, 0.0f);
        return out;
    });

    batcher.enqueue({"你好，世界。", "Hello World", "第三段中文文本"});
    printf("pending = %zu\n", batcher.pending());

    auto batches = batcher.dequeue_all();
    printf("dequeue_all: %zu batch(es)\n", batches.size());
    int i = 0;
    for (auto& b : batches) printf("  batch[%d] audio = %zu samples\n", i++, b.size());
    return 0;
}
