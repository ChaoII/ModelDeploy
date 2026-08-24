#include "audio/tools/hotword.h"
#include "audio/tools/context_graph.h"
#include <algorithm>
#include <cwctype>
#include <csrc/utils/utils.h>

namespace modeldeploy::audio::tool {
using modeldeploy::utf8_to_wstring;
using modeldeploy::wstring_to_string;

namespace {
bool is_ascii_word(const std::wstring& s) {
    return std::all_of(s.begin(), s.end(), [](wchar_t c) { return c < 128; });
}
bool is_word_char(wchar_t c) {
    return c < 128 && (iswalnum(c) || c == L'_'); // 仅 ASCII 视为词内，CJK 视为边界
}
bool match_at(const std::wstring& text, std::size_t pos, const std::wstring& word) {
    if (is_ascii_word(word)) {
        if (pos > 0 && is_word_char(text[pos - 1])) return false;
        if (pos + word.size() < text.size() && is_word_char(text[pos + word.size()])) return false;
    }
    return true;
}
} // namespace

void HotwordContext::add(std::string word, float weight) {
    if (word.empty()) return;
    for (auto& e : entries_) {
        if (e.first == word) { e.second = weight; return; }
    }
    entries_.emplace_back(std::move(word), weight);
}

void HotwordContext::set_weight(const std::string& word, float weight) {
    for (auto& e : entries_)
        if (e.first == word) { e.second = weight; return; }
}

void HotwordContext::clear() { entries_.clear(); }

std::size_t HotwordContext::size() const { return entries_.size(); }

bool HotwordContext::empty() const { return entries_.empty(); }

const std::vector<std::pair<std::string, float>>& HotwordContext::entries() const {
    return entries_;
}

std::vector<std::string> HotwordContext::words() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto& e : entries_) out.push_back(e.first);
    return out;
}

std::vector<FoundHotword> HotwordContext::scan(const std::string& text) const {
    const std::wstring ws = utf8_to_wstring(text);
    std::vector<FoundHotword> out;
    for (const auto& e : entries_) {
        if (e.second == 0.0f) continue;
        const std::wstring ww = utf8_to_wstring(e.first);
        std::size_t count = 0;
        std::size_t pos = 0;
        while ((pos = ws.find(ww, pos)) != std::wstring::npos) {
            if (match_at(ws, pos, ww)) ++count;
            pos += ww.size();
        }
        if (count > 0) out.push_back({e.first, e.second, count});
    }
    return out;
}

std::string HotwordContext::highlight(const std::string& text, const std::string& left,
                                      const std::string& right) const {
    const std::wstring ws = utf8_to_wstring(text);
    std::vector<std::size_t> len(ws.size(), 0);
    for (const auto& e : entries_) {
        const std::wstring ww = utf8_to_wstring(e.first);
        std::size_t pos = 0;
        while ((pos = ws.find(ww, pos)) != std::wstring::npos) {
            if (match_at(ws, pos, ww)) len[pos] = std::max(len[pos], ww.size());
            pos += ww.size();
        }
    }
    std::wstring out;
    std::size_t i = 0;
    while (i < ws.size()) {
        if (len[i] > 0) {
            out += utf8_to_wstring(left);
            out += ws.substr(i, len[i]);
            out += utf8_to_wstring(right);
            i += len[i];
        } else {
            out += ws[i++];
        }
    }
    return wstring_to_string(out);
}

std::vector<int32_t> tokenize_chars(const std::string& text) {
    const std::wstring ws = utf8_to_wstring(text);
    std::vector<int32_t> ids;
    ids.reserve(ws.size());
    for (wchar_t c : ws) ids.push_back(static_cast<int32_t>(c));
    return ids;
}

std::shared_ptr<ContextGraph> build_context_graph(const HotwordContext& ctx,
                                                  const TokenizeFn& tokenize,
                                                  float context_score) {
    std::vector<std::vector<int32_t>> token_ids;
    std::vector<float> scores;
    std::vector<std::string> phrases;
    token_ids.reserve(ctx.size());
    scores.reserve(ctx.size());
    phrases.reserve(ctx.size());
    for (const auto& e : ctx.entries()) {
        auto ids = tokenize(e.first);
        if (ids.empty()) continue;
        token_ids.push_back(std::move(ids));
        scores.push_back(e.second);
        phrases.push_back(e.first);
    }
    if (token_ids.empty()) return std::make_shared<ContextGraph>();
    return std::make_shared<ContextGraph>(token_ids, context_score, 0.0f, scores, phrases);
}

float score(const ContextGraph& graph, const TokenizeFn& tokenize, const std::string& text) {
    const ContextState* state = graph.Root();
    float total = 0.0f;
    for (int32_t id : tokenize(text)) {
        auto [s, next, matched] = graph.ForwardOneStep(state, id);
        total += s;
        state = next;
    }
    auto [final_score, next] = graph.Finalize(state);
    total += final_score;
    return total;
}

std::vector<Hypothesis> rescore(std::vector<Hypothesis> hyps, const HotwordContext& ctx,
                                const TokenizeFn& tokenize, float lambda) {
    auto graph = build_context_graph(ctx, tokenize);
    for (Hypothesis& h : hyps) h.score += lambda * score(*graph, tokenize, h.text);
    std::sort(hyps.begin(), hyps.end(),
              [](const Hypothesis& a, const Hypothesis& b) { return a.score > b.score; });
    return hyps;
}
} // namespace modeldeploy::audio::tool
