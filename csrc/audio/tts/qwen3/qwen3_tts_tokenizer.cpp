// csrc/audio/tts/qwen3/qwen3_tts_tokenizer.cpp
#include "qwen3_tts_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>
#include "core/md_log.h"

namespace modeldeploy::audio::tts {
namespace {

std::string LoadFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

void AppendUtf8(uint32_t cp, std::string* out) {
    if (cp <= 0x7Fu) {
        out->push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out->push_back(static_cast<char>(0xC0u | ((cp >> 6) & 0x1Fu)));
        out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out->push_back(static_cast<char>(0xE0u | ((cp >> 12) & 0x0Fu)));
        out->push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        out->push_back(static_cast<char>(0xF0u | ((cp >> 18) & 0x07u)));
        out->push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out->push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

bool Utf8Next(const std::string& s, size_t* i, uint32_t* cp, size_t* nbytes) {
    if (*i >= s.size()) return false;
    const unsigned char c = static_cast<unsigned char>(s[*i]);
    if (c < 0x80) {
        *cp = c;
        *nbytes = 1;
        return true;
    }
    if ((c >> 5) == 0x6) {
        if (*i + 1 >= s.size()) return false;
        const unsigned char c1 = static_cast<unsigned char>(s[*i + 1]);
        if ((c1 >> 6) != 0x2) return false;
        *cp = ((c & 0x1Fu) << 6) | (c1 & 0x3Fu);
        *nbytes = 2;
        return true;
    }
    if ((c >> 4) == 0xE) {
        if (*i + 2 >= s.size()) return false;
        const unsigned char c1 = static_cast<unsigned char>(s[*i + 1]);
        const unsigned char c2 = static_cast<unsigned char>(s[*i + 2]);
        if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2) return false;
        *cp = ((c & 0x0Fu) << 12) | ((c1 & 0x3Fu) << 6) | (c2 & 0x3Fu);
        *nbytes = 3;
        return true;
    }
    if ((c >> 3) == 0x1E) {
        if (*i + 3 >= s.size()) return false;
        const unsigned char c1 = static_cast<unsigned char>(s[*i + 1]);
        const unsigned char c2 = static_cast<unsigned char>(s[*i + 2]);
        const unsigned char c3 = static_cast<unsigned char>(s[*i + 3]);
        if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2 || (c3 >> 6) != 0x2)
            return false;
        *cp = ((c & 0x07u) << 18) | ((c1 & 0x3Fu) << 12) | ((c2 & 0x3Fu) << 6) |
              (c3 & 0x3Fu);
        *nbytes = 4;
        return true;
    }
    return false;
}

bool IsNewline(uint32_t cp) { return cp == '\n' || cp == '\r'; }

bool IsAsciiSpace(uint32_t cp) { return cp == ' '; }

bool IsWhitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' ||
           cp == '\f';
}

bool IsAsciiAlpha(uint32_t cp) {
    return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
}

bool IsAsciiDigit(uint32_t cp) { return cp >= '0' && cp <= '9'; }

bool IsLetter(uint32_t cp) {
    if (IsAsciiAlpha(cp)) return true;
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    if (cp >= 0x3040 && cp <= 0x30FF) return true;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    if (cp >= 0x1100 && cp <= 0x11FF) return true;
    if (cp >= 0x00C0 && cp <= 0x02AF) return true;
    return false;
}

bool IsNumber(uint32_t cp) {
    if (IsAsciiDigit(cp)) return true;
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;
    return false;
}

bool IsWordChar(uint32_t cp) {
    return IsLetter(cp) || IsNumber(cp) || cp == '_';
}

// Qwen2 预分词 Split 正则（近似实现）：
//   '(?i:sd|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
std::vector<std::string> SplitByQwen3Pattern(const std::string& text) {
    std::vector<std::string> out;
    out.reserve(text.size() / 2 + 1);

    size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '\'') {
            auto lower = [](char c) -> char {
                return static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c)));
            };
            if (i + 1 < text.size()) {
                const char c1 = lower(text[i + 1]);
                if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                    out.push_back(text.substr(i, 2));
                    i += 2;
                    continue;
                }
                if (i + 2 < text.size()) {
                    const char c2 = lower(text[i + 2]);
                    if (c1 == 'r' && c2 == 'e') {
                        out.push_back(text.substr(i, 3));
                        i += 3;
                        continue;
                    }
                    if (c1 == 'v' && c2 == 'e') {
                        out.push_back(text.substr(i, 3));
                        i += 3;
                        continue;
                    }
                    if (c1 == 'l' && c2 == 'l') {
                        out.push_back(text.substr(i, 3));
                        i += 3;
                        continue;
                    }
                }
            }
        }

        size_t cur = i;
        uint32_t cp = 0;
        size_t n = 0;
        if (!Utf8Next(text, &cur, &cp, &n) || n == 0) {
            out.push_back(text.substr(i, 1));
            i += 1;
            continue;
        }

        auto peek_next_cp = [&](size_t pos, uint32_t* cp2, size_t* n2) -> bool {
            size_t t = pos;
            uint32_t x = 0;
            size_t nn = 0;
            if (!Utf8Next(text, &t, &x, &nn)) return false;
            if (cp2) *cp2 = x;
            if (n2) *n2 = nn;
            return true;
        };

        {
            uint32_t next_cp = 0;
            size_t next_n = 0;
            const bool has_next = peek_next_cp(i + n, &next_cp, &next_n);

            const bool cur_ok_prefix =
                (!IsNewline(cp) && !IsLetter(cp) && !IsNumber(cp));
            const bool cur_is_letter = IsLetter(cp);

            if (cur_is_letter || (cur_ok_prefix && has_next &&
                                  IsLetter(next_cp))) {
                const size_t start = i;
                size_t j = i;
                if (!cur_is_letter) {
                    j += n;
                    while (j < text.size()) {
                        size_t t = j;
                        uint32_t cpl = 0;
                        size_t nl = 0;
                        if (!Utf8Next(text, &t, &cpl, &nl)) break;
                        if (!IsLetter(cpl)) break;
                        j += nl;
                    }
                } else {
                    j = i;
                    while (j < text.size()) {
                        size_t t = j;
                        uint32_t cpl = 0;
                        size_t nl = 0;
                        if (!Utf8Next(text, &t, &cpl, &nl)) break;
                        if (!IsLetter(cpl)) break;
                        j += nl;
                    }
                }
                out.push_back(text.substr(start, j - start));
                i = j;
                continue;
            }
        }

        if (IsNumber(cp)) {
            out.push_back(text.substr(i, n));
            i += n;
            continue;
        }

        {
            const bool starts_with_space_prefix = IsAsciiSpace(cp);
            const size_t start = i;
            size_t j = i;

            auto is_punct_like = [&](uint32_t x) -> bool {
                return (!IsWhitespace(x) && !IsLetter(x) && !IsNumber(x));
            };

            if (starts_with_space_prefix) {
                uint32_t next_cp = 0;
                size_t next_n = 0;
                if (peek_next_cp(i + n, &next_cp, &next_n) &&
                    is_punct_like(next_cp)) {
                    j += n;
                    while (j < text.size()) {
                        size_t t = j;
                        uint32_t cx = 0;
                        size_t nx = 0;
                        if (!Utf8Next(text, &t, &cx, &nx)) break;
                        if (!is_punct_like(cx)) break;
                        j += nx;
                    }
                    while (j < text.size()) {
                        size_t t = j;
                        uint32_t cx = 0;
                        size_t nx = 0;
                        if (!Utf8Next(text, &t, &cx, &nx)) break;
                        if (!IsNewline(cx)) break;
                        j += nx;
                    }
                    out.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
            } else if (is_punct_like(cp)) {
                while (j < text.size()) {
                    size_t t = j;
                    uint32_t cx = 0;
                    size_t nx = 0;
                    if (!Utf8Next(text, &t, &cx, &nx)) break;
                    if (!is_punct_like(cx)) break;
                    j += nx;
                }
                while (j < text.size()) {
                    size_t t = j;
                    uint32_t cx = 0;
                    size_t nx = 0;
                    if (!Utf8Next(text, &t, &cx, &nx)) break;
                    if (!IsNewline(cx)) break;
                    j += nx;
                }
                out.push_back(text.substr(start, j - start));
                i = j;
                continue;
            }
        }

        {
            if (IsWhitespace(cp)) {
                const size_t start = i;
                size_t j = i;

                bool saw_newline = false;
                while (j < text.size()) {
                    size_t t = j;
                    uint32_t cx = 0;
                    size_t nx = 0;
                    if (!Utf8Next(text, &t, &cx, &nx)) break;
                    if (IsNewline(cx)) {
                        saw_newline = true;
                        break;
                    }
                    if (!IsWhitespace(cx)) break;
                    j += nx;
                }

                if (saw_newline) {
                    while (j < text.size()) {
                        size_t t = j;
                        uint32_t cx = 0;
                        size_t nx = 0;
                        if (!Utf8Next(text, &t, &cx, &nx)) break;
                        if (!IsNewline(cx)) break;
                        j += nx;
                    }
                    out.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
            }
        }

        if (IsWhitespace(cp)) {
            bool only_ws_to_end = true;
            size_t j = i;
            while (j < text.size()) {
                size_t t = j;
                uint32_t cx = 0;
                size_t nx = 0;
                if (!Utf8Next(text, &t, &cx, &nx)) break;
                if (!IsWhitespace(cx)) {
                    only_ws_to_end = false;
                    break;
                }
                j += nx;
            }
            if (only_ws_to_end) {
                out.push_back(text.substr(i));
                break;
            }
        }

        if (IsWhitespace(cp)) {
            const size_t start = i;
            size_t j = i;
            while (j < text.size()) {
                size_t t = j;
                uint32_t cx = 0;
                size_t nx = 0;
                if (!Utf8Next(text, &t, &cx, &nx)) break;
                if (!IsWhitespace(cx)) break;
                j += nx;
            }
            out.push_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        out.push_back(text.substr(i, n));
        i += n;
    }

    return out;
}

std::vector<std::string> SplitUtf8ToChars(const std::string& s) {
    std::vector<std::string> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        size_t t = i;
        uint32_t cp = 0;
        size_t n = 0;
        if (!Utf8Next(s, &t, &cp, &n) || n == 0) {
            out.push_back(s.substr(i, 1));
            i += 1;
            continue;
        }
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

std::string MakeMergeKey(const std::string& a, const std::string& b) {
    std::string k = a;
    k.push_back('\t');
    k.append(b);
    return k;
}

}  // namespace

Qwen3TtsTokenizer::Qwen3TtsTokenizer(const std::string& tokenizer_dir) {
    const std::string vocab_path = tokenizer_dir + "/vocab.json";
    const std::string merges_path = tokenizer_dir + "/merges.txt";
    const std::string config_path = tokenizer_dir + "/tokenizer_config.json";

    const std::string vocab_blob = LoadFile(vocab_path);
    const std::string merges_blob = LoadFile(merges_path);
    const std::string config_blob = LoadFile(config_path);

    if (vocab_blob.empty() || merges_blob.empty()) {
        MD_LOG_ERROR << "Failed to load tokenizer files from " << tokenizer_dir
                     << std::endl;
        return;
    }

    using nlohmann::json;
    const json vocab = json::parse(vocab_blob, nullptr, false);
    if (vocab.is_discarded() || !vocab.is_object()) {
        MD_LOG_ERROR << "Failed to parse " << vocab_path << std::endl;
        return;
    }
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        if (!it.value().is_number_integer()) continue;
        const int64_t id = it.value().get<int64_t>();
        if (id >= 0) token2id_[it.key()] = static_cast<int32_t>(id);
    }

    std::istringstream is(merges_blob);
    std::string line;
    int32_t rank = 0;
    while (std::getline(is, line)) {
        if (line.empty()) continue;
        if (line.rfind("#version", 0) == 0) continue;
        std::string left, right;
        {
            std::istringstream ls(line);
            if (!(ls >> left >> right)) continue;
        }
        merges_rank_[MakeMergeKey(left, right)] = rank++;
    }

    BuildBytesToUnicode();
    if (!config_blob.empty()) LoadTokenizerConfig(config_blob);
    Finalize();

    if (loaded()) {
        MD_LOG_INFO << "Qwen3TtsTokenizer loaded from " << tokenizer_dir
                    << ", vocab size = " << vocab_size() << std::endl;
    }
}

void Qwen3TtsTokenizer::BuildBytesToUnicode() {
    std::vector<uint32_t> bs;
    bs.reserve(256);
    for (uint32_t c = 33; c <= 126; ++c) bs.push_back(c);
    for (uint32_t c = 161; c <= 172; ++c) bs.push_back(c);
    for (uint32_t c = 174; c <= 255; ++c) bs.push_back(c);

    std::vector<uint32_t> cs = bs;
    cs.reserve(256);
    uint32_t n = 0;
    auto contains = [&](uint32_t b) -> bool {
        return std::find(bs.begin(), bs.end(), b) != bs.end();
    };
    for (uint32_t b = 0; b <= 255; ++b) {
        if (!contains(b)) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    unicode_to_byte_.clear();
    for (size_t i = 0; i < bs.size(); ++i) {
        const uint32_t b = bs[i];
        const uint32_t c = cs[i];
        std::string u;
        AppendUtf8(c, &u);
        byte_to_unicode_[b] = u;
        unicode_to_byte_[u] = static_cast<uint8_t>(b);
    }
}

void Qwen3TtsTokenizer::LoadTokenizerConfig(const std::string& config_blob) {
    using nlohmann::json;
    const json cfg = json::parse(config_blob, nullptr, false);
    if (cfg.is_discarded() || !cfg.is_object()) {
        MD_LOG_WARN << "Failed to parse tokenizer_config.json" << std::endl;
        return;
    }

    // added_tokens_decoder: {"151643": {"content": "...", "special": bool}, ...}
    if (cfg.contains("added_tokens_decoder") &&
        cfg["added_tokens_decoder"].is_object()) {
        const json& decoder = cfg["added_tokens_decoder"];
        for (auto it = decoder.begin(); it != decoder.end(); ++it) {
            const std::string content =
                it.value().value("content", std::string{});
            const bool special = it.value().value("special", false);
            if (content.empty()) continue;
            // 键即 id
            const int64_t id = std::strtoll(it.key().c_str(), nullptr, 10);
            if (id >= 0) {
                AddedToken t;
                t.content = content;
                t.id = static_cast<int32_t>(id);
                t.special = special;
                added_tokens_.push_back(std::move(t));
                token2id_[content] = t.id;
            }
        }
    }

    // additional_special_tokens（兜底，避免漏掉）
    if (cfg.contains("additional_special_tokens") &&
        cfg["additional_special_tokens"].is_array()) {
        for (const auto& v : cfg["additional_special_tokens"]) {
            if (!v.is_string()) continue;
            const std::string content = v.get<std::string>();
            if (content.empty()) continue;
            auto it = token2id_.find(content);
            if (it != token2id_.end()) continue;  // 已在 added_tokens_decoder
            // 不在 vocab 中则新增：id 延续 vocab 之后
            int32_t new_id = -1;
            int32_t max_id = -1;
            for (const auto& kv : token2id_) max_id = std::max(max_id, kv.second);
            new_id = max_id + 1;
            AddedToken t;
            t.content = content;
            t.id = new_id;
            t.special = true;
            added_tokens_.push_back(std::move(t));
            token2id_[content] = new_id;
        }
    }
}

void Qwen3TtsTokenizer::Finalize() {
    // 建特殊 id 集合
    special_ids_.clear();
    for (const auto& t : added_tokens_) {
        if (t.special && t.id >= 0) special_ids_.insert(t.id);
    }

    // 建 added tokens 最长匹配 trie（字节级）
    trie_.clear();
    trie_.push_back(TrieNode{});
    for (int32_t i = 0; i < static_cast<int32_t>(added_tokens_.size()); ++i) {
        int32_t node = 0;
        for (const unsigned char b : added_tokens_[i].content) {
            auto it = trie_[node].next.find(b);
            if (it == trie_[node].next.end()) {
                const int32_t new_node = static_cast<int32_t>(trie_.size());
                trie_.push_back(TrieNode{});
                trie_[node].next.emplace(b, new_node);
                node = new_node;
            } else {
                node = it->second;
            }
        }
        trie_[node].token_index = i;
    }

    // 建 id->token（供诊断/解码用）
    id2token_.assign(token2id_.size() + added_tokens_.size() + 8, std::string{});
    for (const auto& kv : token2id_) {
        if (kv.second < 0) continue;
        if (static_cast<size_t>(kv.second) >= id2token_.size())
            id2token_.resize(static_cast<size_t>(kv.second) + 1);
        id2token_[static_cast<size_t>(kv.second)] = kv.first;
    }
}

std::string ByteLevelEncode(const std::string& token,
                            const std::string byte_to_unicode[256]) {
    std::string out;
    out.reserve(token.size() * 2);
    for (const unsigned char b : token) out.append(byte_to_unicode[b]);
    return out;
}

std::vector<std::string> BpeEncodeWithCache(
    const std::string& word,
    const std::unordered_map<std::string, int32_t>& merges_rank,
    std::unordered_map<std::string, std::vector<std::string>>* cache) {
    auto it = cache->find(word);
    if (it != cache->end()) return it->second;

    std::vector<std::string> symbols = SplitUtf8ToChars(word);
    if (symbols.empty()) {
        (*cache)[word] = {};
        return {};
    }
    if (symbols.size() == 1) {
        (*cache)[word] = symbols;
        return symbols;
    }

    while (symbols.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        int32_t best_pos = -1;
        for (int32_t i = 0; i + 1 < static_cast<int32_t>(symbols.size()); ++i) {
            auto it2 = merges_rank.find(MakeMergeKey(symbols[i], symbols[i + 1]));
            if (it2 != merges_rank.end() && it2->second < best_rank) {
                best_rank = it2->second;
                best_pos = i;
            }
        }
        if (best_pos < 0) break;
        symbols[best_pos].append(symbols[best_pos + 1]);
        symbols.erase(symbols.begin() + best_pos + 1);
    }

    (*cache)[word] = symbols;
    return symbols;
}

std::vector<int64_t> Qwen3TtsTokenizer::Encode(const std::string& text) {
    std::vector<int64_t> out;
    if (token2id_.empty() || text.empty()) return out;

    auto match_added = [&](size_t pos, size_t* matched_len,
                           int32_t* token_index) {
        if (trie_.empty()) return;
        int32_t node = 0;
        int32_t best_idx = -1;
        int32_t best_len = 0;
        size_t i = pos;
        while (i < text.size()) {
            const uint8_t b = static_cast<uint8_t>(text[i]);
            auto it = trie_[node].next.find(b);
            if (it == trie_[node].next.end()) break;
            node = it->second;
            ++i;
            if (trie_[node].token_index >= 0) {
                best_idx = trie_[node].token_index;
                best_len = static_cast<int32_t>(i - pos);
            }
        }
        *matched_len = best_len > 0 ? static_cast<size_t>(best_len) : 0;
        *token_index = best_idx;
    };

    size_t pos = 0;
    size_t last = 0;
    while (pos < text.size()) {
        size_t mlen = 0;
        int32_t tidx = -1;
        match_added(pos, &mlen, &tidx);

        if (mlen > 0 && tidx >= 0) {
            if (pos > last) {
                const std::string seg = text.substr(last, pos - last);
                const auto pieces = SplitByQwen3Pattern(seg);
                for (const auto& p : pieces) {
                    const std::string bl = ByteLevelEncode(p, byte_to_unicode_);
                    const auto bpe_toks =
                        BpeEncodeWithCache(bl, merges_rank_, &bpe_cache_);
                    for (const auto& bt : bpe_toks) {
                        auto it = token2id_.find(bt);
                        if (it == token2id_.end()) continue;
                        out.push_back(it->second);
                    }
                }
            }

            const auto& atok = added_tokens_[static_cast<size_t>(tidx)];
            out.push_back(atok.id);

            pos += mlen;
            last = pos;
            continue;
        }

        ++pos;
    }

    if (last < text.size()) {
        const std::string seg = text.substr(last);
        const auto pieces = SplitByQwen3Pattern(seg);
        for (const auto& p : pieces) {
            const std::string bl = ByteLevelEncode(p, byte_to_unicode_);
            const auto bpe_toks = BpeEncodeWithCache(bl, merges_rank_, &bpe_cache_);
            for (const auto& bt : bpe_toks) {
                auto it = token2id_.find(bt);
                if (it == token2id_.end()) continue;
                out.push_back(it->second);
            }
        }
    }

    return out;
}

}  // namespace modeldeploy::audio::tts
