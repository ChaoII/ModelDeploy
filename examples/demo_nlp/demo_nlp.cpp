// ModelDeploy demo_nlp —— NLP 工具/分类演示。
// Usage: demo_nlp [bert_classifier.onnx] [text]
//   若无 onnx：演示 分词(Splitter)/分句/关键词(Keywords)/统计(Stats)，无需权重。
//   若有 onnx：TextClassifier classify 输出 (label, score)。
#include <cstdio>
#include <string>

#include "nlp/solutions/text_classifier.h"
#include "nlp/tools/keywords.h"
#include "nlp/tools/splitter.h"
#include "nlp/tools/stats.h"
#include "runtime/runtime_option.h"

static const char* kText =
    "ModelDeploy 是一个推理部署SDK。它支持视觉、音频与NLP任务。设计目标是快速、跨后端。";

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : "";
    const std::string text  = argc > 2 ? argv[2] : kText;

    // 无权重工具演示。
    auto sents = modeldeploy::nlp::tool::Splitter::split_sentences(text);
    printf("split_sentences: %zu 句\n", sents.size());
    for (auto& s : sents) printf("  [%s]\n", s.c_str());

    auto kw = modeldeploy::nlp::tool::Keywords::top(text, 5);
    printf("keywords(top5):\n");
    for (auto& kv : kw) printf("  %s : %d\n", kv.first.c_str(), kv.second);

    printf("stats: chars=%zu words=%zu sents=%zu\n",
           modeldeploy::nlp::tool::Stats::char_count(text),
           modeldeploy::nlp::tool::Stats::word_count(text),
           modeldeploy::nlp::tool::Stats::sentence_count(text));

    // 可选：TextClassifier 分类。
    if (!model.empty()) {
        modeldeploy::RuntimeOption opt; opt.use_ort_backend();
        modeldeploy::nlp::solution::TextClassifier clf(model, opt);
        if (!clf.is_initialized()) { printf("classifier init failed (missing weights?)\n"); return 2; }
        int label; float score;
        if (clf.predict(text, &label, &score))
            printf("classify: label=%d score=%.4f\n", label, score);
        else
            printf("classify failed\n");
    }
    return 0;
}
