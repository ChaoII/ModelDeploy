// context_graph.h —— Kaldi/sherpa-onnx ContextGraph 的移植（token 级上下文偏置）。
//
// 原始实现：sherpa-onnx/csrc/context-graph.h/.cc
//   Copyright (c) 2023 Xiaomi Corporation  (Apache License 2.0)
//   https://github.com/k2-fsa/sherpa-onnx
// 本文件按 ModelDeploy 命名空间/无外部宏适配，算法保持一致。
// 它在 ASR 解码时对“命中热词路径的 token”加分，热词完整命中后自动退回到根
// （context reset），是行业标准的热词 boosting 方法，优于朴素子串匹配。

#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "core/md_decl.h"

namespace modeldeploy::audio::tool {

class ContextGraph;
using ContextGraphPtr = std::shared_ptr<ContextGraph>;

struct MODELDEPLOY_CXX_EXPORT ContextState {
    int32_t token;
    float token_score;
    float node_score;
    float output_score;
    int32_t level;
    float ac_threshold;
    bool is_end;
    std::string phrase;
    std::unordered_map<int32_t, std::unique_ptr<ContextState>> next;
    const ContextState* fail = nullptr;
    const ContextState* output = nullptr;

    ContextState() = default;
    ContextState(const ContextState&) = delete;
    ContextState& operator=(const ContextState&) = delete;
    ContextState(ContextState&&) = default;
    ContextState& operator=(ContextState&&) = default;
    ContextState(int32_t token, float token_score, float node_score,
                 float output_score, int32_t level = 0, float ac_threshold = 0.0f,
                 bool is_end = false, const std::string& phrase = {})
        : token(token), token_score(token_score), node_score(node_score),
          output_score(output_score), level(level), ac_threshold(ac_threshold),
          is_end(is_end), phrase(phrase) {}
};

class MODELDEPLOY_CXX_EXPORT ContextGraph {
public:
    ContextGraph() : ContextGraph(std::vector<std::vector<int32_t>>(), 0.0f) {}
    ContextGraph(const std::vector<std::vector<int32_t>>& token_ids,
                 float context_score, float ac_threshold,
                 const std::vector<float>& scores = {},
                 const std::vector<std::string>& phrases = {},
                 const std::vector<float>& ac_thresholds = {})
        : context_score_(context_score), ac_threshold_(ac_threshold) {
        root_ = std::make_unique<ContextState>(-1, 0, 0, 0);
        root_->fail = root_.get();
        Build(token_ids, scores, phrases, ac_thresholds);
    }

    ContextGraph(const std::vector<std::vector<int32_t>>& token_ids,
                 float context_score, const std::vector<float>& scores = {})
        : ContextGraph(token_ids, context_score, 0.0f, scores,
                       std::vector<std::string>(), std::vector<float>()) {}

    // 前进一步：返回 (该步分数加成, 新状态, 命中的热词节点[可能为 null])
    std::tuple<float, const ContextState*, const ContextState*> ForwardOneStep(
        const ContextState* state, int32_t token_id,
        bool strict_mode = true) const;

    // 当前状态是否命中热词
    std::pair<bool, const ContextState*> IsMatched(const ContextState* state) const;

    // 序列结束时的收尾分数（用于 rescoring）
    std::pair<float, const ContextState*> Finalize(const ContextState* state) const;

    const ContextState* Root() const { return root_.get(); }

private:
    float context_score_ = 0.0f;
    float ac_threshold_ = 0.0f;
    std::unique_ptr<ContextState> root_;
    void Build(const std::vector<std::vector<int32_t>>& token_ids,
               const std::vector<float>& scores,
               const std::vector<std::string>& phrases,
               const std::vector<float>& ac_thresholds);
    void FillFailOutput();
};

} // namespace modeldeploy::audio::tool
