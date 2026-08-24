#pragma once
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "core/md_decl.h"

namespace modeldeploy::audio::tool {

class ContextGraph;

// 命中热词的记录（用于调试/QA，非偏置核心）
struct MODELDEPLOY_CXX_EXPORT FoundHotword {
    std::string word;
    float weight{1.0f};
    std::size_t count{0};
};

// 一条候选识别结果（用于 n-best rescoring）
struct MODELDEPLOY_CXX_EXPORT Hypothesis {
    std::string text;
    float score{0.0f}; // 声学/语言模型得分（越高越好）
};

// 把一句话转为 token id 序列（在统一词表/编码器视角下）。
// 对 CJK，可默认用 tokenize_chars()（按 Unicode 码点）。
using TokenizeFn = std::function<std::vector<int32_t>(const std::string&)>;

// 热词 boosting：模型无关、可插拔。
// 偏置核心用的是行业标准的 Kaldi/sherpa-onnx ContextGraph（token 级上下文偏置：
// 命中热词路径加分、短语命中后解锁重置），不是朴素子串匹配。见 context_graph.h。
// scan()/highlight() 仅为 QA/可视化辅助。
class MODELDEPLOY_CXX_EXPORT HotwordContext {
public:
    HotwordContext() = default;

    void add(std::string word, float weight = 1.0f);
    void set_weight(const std::string& word, float weight);
    void clear();
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] bool empty() const;
    [[nodiscard]] const std::vector<std::pair<std::string, float>>& entries() const;
    [[nodiscard]] std::vector<std::string> words() const;

    // 在 text 中查找命中的热词（含次数；ASCII 按词边界）
    [[nodiscard]] std::vector<FoundHotword> scan(const std::string& text) const;
    // 把命中热词高亮（QA/可视化）
    [[nodiscard]] std::string highlight(const std::string& text, const std::string& left = "[",
                                        const std::string& right = "]") const;

private:
    std::vector<std::pair<std::string, float>> entries_;
};

// 默认字符级分词器：CJK 按 Unicode 码点输出 id（适合中文/英文混排的字符级词表）
MODELDEPLOY_CXX_EXPORT std::vector<int32_t> tokenize_chars(const std::string& text);

// 依据热词列表构建 ContextGraph（供解码/重打分偏置）
MODELDEPLOY_CXX_EXPORT std::shared_ptr<ContextGraph>
build_context_graph(const HotwordContext& ctx, const TokenizeFn& tokenize,
                    float context_score = 1.0f);

// 用 ContextGraph 对一段 token 化文本打分（偏置后得分）
MODELDEPLOY_CXX_EXPORT float score(const ContextGraph& graph, const TokenizeFn& tokenize,
                                   const std::string& text);

// 对 n-best 候选按 声学得分 + λ*热词偏置 重排序（就地排序后返回）
MODELDEPLOY_CXX_EXPORT std::vector<Hypothesis>
rescore(std::vector<Hypothesis> hyps, const HotwordContext& ctx, const TokenizeFn& tokenize,
        float lambda = 1.0f);

} // namespace modeldeploy::audio::tool
