// csrc/audio/tts/audio8/audio8.cpp
#include "audio/tts/audio8/audio8.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

#include "audio/tts/common/sampling.h"
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
        if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2 || (c3 >> 6) != 0x2) return false;
        *cp = ((c & 0x07u) << 18) | ((c1 & 0x3Fu) << 12) | ((c2 & 0x3Fu) << 6) |
              (c3 & 0x3Fu);
        *nbytes = 4;
        return true;
    }
    return false;
}

// 取 s 中下标 pos 处的码点；false 表示越界/非法
bool CodePointAt(const std::string& s, size_t pos, uint32_t* cp) {
    if (pos >= s.size()) return false;
    size_t t = pos;
    uint32_t c = 0;
    size_t n = 0;
    if (!Utf8Next(s, &t, &c, &n) || n == 0) {
        *cp = static_cast<unsigned char>(s[pos]);
        return true;
    }
    *cp = c;
    return true;
}

bool IsNewline(uint32_t cp) { return cp == '\n' || cp == '\r'; }

// Python str 白格字符（含 Unicode 白格）
bool IsPyspace(uint32_t cp) {
    if (cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C ||
        cp == 0x0D)
        return true;
    if (cp >= 0x1C && cp <= 0x1F) return true;
    if (cp == 0x85 || cp == 0xA0 || cp == 0x1680) return true;
    if (cp >= 0x2000 && cp <= 0x200A) return true;
    if (cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
        cp == 0x3000)
        return true;
    return false;
}

bool IsCjkChar(uint32_t cp) {
    return (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x2E80 && cp <= 0x2FDF) ||
           (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0x3040 && cp <= 0x30FF) ||
           (cp >= 0x3100 && cp <= 0x31FF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0xA960 && cp <= 0xA97F) ||
           (cp >= 0xAC00 && cp <= 0xD7A3) || (cp >= 0xD7B0 && cp <= 0xD7FF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0xFE30 && cp <= 0xFE4F) ||
           (cp >= 0xFF01 && cp <= 0xFF9F) || (cp >= 0x20000 && cp <= 0x2FA1F);
}

// Unicode 其它类（C）近似：控制/格式/代理/私用/未分配（常见 Cc/Cf 范围）
bool IsUnicodeOther(uint32_t cp) {
    if (cp <= 0x1F) return !IsPyspace(cp);
    if (cp >= 0x7F && cp <= 0x9F) return true;
    if (cp >= 0xD800 && cp <= 0xDFFF) return true;
    if (cp >= 0xE000 && cp <= 0xF8FF) return true;
    if (cp == 0xAD || cp == 0x61C || cp == 0x6DD || cp == 0x070F || cp == 0xFEFF)
        return true;
    if (cp >= 0x200B && cp <= 0x200F) return true;
    if (cp >= 0x202A && cp <= 0x202E) return true;
    if (cp >= 0x2060 && cp <= 0x2064) return true;
    if (cp >= 0xFFF9 && cp <= 0xFFFB) return true;
    return false;
}

bool IsLinebreakCp(uint32_t cp) {
    return IsNewline(cp) || cp == '\v' || cp == '\f' || (cp >= 0x1C && cp <= 0x1E) ||
           cp == 0x85 || cp == 0x2028 || cp == 0x2029;
}

// pos 位于字符边界时，返回该位置前一个字符的起始字节下标；否则返回 npos
size_t PrevCharStart(const std::string& s, size_t pos) {
    if (pos == 0 || pos > s.size()) return std::string::npos;
    size_t p = 0;
    size_t prev = std::string::npos;
    while (p < pos) {
        size_t t = p;
        uint32_t cc = 0;
        size_t nn = 0;
        prev = p;
        if (!Utf8Next(s, &t, &cc, &nn) || nn == 0) {
            p += 1;
        } else {
            p += nn;
        }
    }
    return prev;
}

// prompt.py clean_text：保留白格、丢弃 C 类字符，随后规整白格并 strip
std::string CleanText(const std::string& text) {
    std::string kept;
    kept.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        uint32_t cp = 0;
        size_t n = 0;
        size_t t = i;
        if (!Utf8Next(text, &t, &cp, &n) || n == 0) {
            n = 1;
            cp = static_cast<unsigned char>(text[i]);
        }
        if (IsPyspace(cp) || !IsUnicodeOther(cp)) kept.append(text, i, n);
        i += n;
    }
    // _normalize_whitespace：re.sub(r"\s+", replace, text)
    std::string pre;
    pre.reserve(kept.size());
    size_t j = 0;
    while (j < kept.size()) {
        uint32_t cp = 0;
        size_t n = 0;
        size_t t = j;
        if (!Utf8Next(kept, &t, &cp, &n) || n == 0) n = 1;
        if (!IsPyspace(cp)) {
            pre.append(kept, j, n);
            j += n;
            continue;
        }
        // 收集白格 run
        size_t run_end = j + n;
        bool has_linebreak = IsLinebreakCp(cp);
        while (run_end < kept.size()) {
            uint32_t cp2 = 0;
            if (!CodePointAt(kept, run_end, &cp2)) break;
            if (!IsPyspace(cp2)) break;
            if (IsLinebreakCp(cp2)) has_linebreak = true;
            size_t t2 = run_end;
            size_t nn = 0;
            uint32_t tmp2 = 0;
            Utf8Next(kept, &t2, &tmp2, &nn);
            run_end += nn == 0 ? 1 : nn;
        }
        // 左右字符（CJK 间换行删除）
        uint32_t left_cp = 0, right_cp = 0;
        const size_t left_start = PrevCharStart(kept, j);
        const bool has_left = left_start != std::string::npos && CodePointAt(kept, left_start, &left_cp);
        const bool has_right = run_end < kept.size() && CodePointAt(kept, run_end, &right_cp);
        if (!(has_linebreak && has_left && has_right && IsCjkChar(left_cp) &&
              IsCjkChar(right_cp))) {
            pre.push_back(' ');
        }
        j = run_end;
    }
    // strip
    size_t b = 0, e = pre.size();
    while (b < e && pre[b] == ' ') ++b;
    while (e > b && pre[e - 1] == ' ') --e;
    return pre.substr(b, e - b);
}

// prompt.py format_reference_text：无 <|speaker:N|> 时前缀 <|speaker:0|>
std::string FormatReferenceText(std::string text) {
    text = CleanText(text);
    auto has_speaker = [](const std::string& s) {
        size_t p = s.find("<|speaker:");
        while (p != std::string::npos) {
            size_t q = p + std::string("<|speaker:").size();
            size_t d = q;
            while (d < s.size() && std::isdigit(static_cast<unsigned char>(s[d]))) ++d;
            if (d > q && d + 1 < s.size() && s[d] == '|' && s[d + 1] == '>')
                return true;
            p = s.find("<|speaker:", p + 1);
        }
        return false;
    };
    if (!has_speaker(text)) text = "<|speaker:0|>" + text;
    return text;
}

// ---------- 预分词（与 qwen3 tokenizer 的 Split 正则近似一致） ----------

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
bool IsWhitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' ||
           cp == '\f';
}

std::vector<std::string> SplitByQwenPattern(const std::string& text) {
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

        const bool cur_is_letter = IsLetter(cp);
        if (cur_is_letter ||
            (!IsNewline(cp) && !IsLetter(cp) && !IsNumber(cp) &&
             i + n < text.size())) {
            const bool prefix_letter_follows =
                cur_is_letter || [&]() {
                    uint32_t nx = 0;
                    return CodePointAt(text, i + n, &nx) && IsLetter(nx);
                }();
            if (prefix_letter_follows) {
                size_t j = i;
                if (!cur_is_letter) j += n;
                while (j < text.size()) {
                    uint32_t cpl = 0;
                    if (!CodePointAt(text, j, &cpl)) break;
                    if (!IsLetter(cpl)) break;
                    size_t t = j;
                    size_t nl = 0;
                    if (!Utf8Next(text, &t, &cpl, &nl)) break;
                    j += nl == 0 ? 1 : nl;
                }
                out.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
        }

        if (IsNumber(cp)) {
            out.push_back(text.substr(i, n));
            i += n;
            continue;
        }

        // 标点/特殊： ?[^\s\p{L}\p{N}]+[\r\n]*
        if (!IsWhitespace(cp) && !IsLetter(cp) && !IsNumber(cp)) {
            size_t start = i;
            size_t k = i;
            while (k < text.size()) {
                uint32_t cx = 0;
                if (!CodePointAt(text, k, &cx)) break;
                if (IsWhitespace(cx) || IsLetter(cx) || IsNumber(cx)) break;
                size_t t = k;
                size_t nx = 0;
                if (!Utf8Next(text, &t, &cx, &nx)) break;
                k += nx == 0 ? 1 : nx;
            }
            size_t k2 = k;
            while (k2 < text.size()) {
                uint32_t cx = 0;
                if (!CodePointAt(text, k2, &cx)) break;
                if (!IsNewline(cx)) break;
                size_t t = k2;
                size_t nx = 0;
                if (!Utf8Next(text, &t, &cx, &nx)) break;
                k2 += nx == 0 ? 1 : nx;
            }
            out.push_back(text.substr(i, k2 - i));
            i = k2;
            continue;
        }

        if (IsWhitespace(cp)) {
            size_t k = i;
            bool saw_nl = false;
            while (k < text.size()) {
                uint32_t cx = 0;
                if (!CodePointAt(text, k, &cx)) break;
                if (!IsWhitespace(cx)) break;
                if (IsNewline(cx)) saw_nl = true;
                size_t t = k;
                size_t nx = 0;
                if (!Utf8Next(text, &t, &cx, &nx)) break;
                k += nx == 0 ? 1 : nx;
            }
            if (saw_nl) {
                size_t k2 = k;
                while (k2 < text.size()) {
                    uint32_t cx = 0;
                    if (!CodePointAt(text, k2, &cx)) break;
                    if (!IsNewline(cx)) break;
                    size_t t = k2;
                    size_t nx = 0;
                    if (!Utf8Next(text, &t, &cx, &nx)) break;
                    k2 += nx == 0 ? 1 : nx;
                }
                out.push_back(text.substr(i, k2 - i));
                i = k2;
                continue;
            }
            out.push_back(text.substr(i, k - i));
            i = k;
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
        uint32_t cp = 0;
        size_t n = 0;
        size_t t = i;
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

// ---------- npy 读取（voice codes 持久化格式） ----------

std::string NpyFindQuoted(const std::string& header, const std::string& key) {
    const size_t p = header.find(key);
    if (p == std::string::npos) return {};
    const size_t colon = header.find(':', p);
    if (colon == std::string::npos) return {};
    const size_t q1 = header.find('\'', colon);
    const size_t qd = header.find('"', colon);
    size_t q = std::string::npos;
    if (q1 == std::string::npos) q = qd;
    else if (qd == std::string::npos) q = q1;
    else q = std::min(q1, qd);
    if (q == std::string::npos) return {};
    const char quote = header[q];
    const size_t end = header.find(quote, q + 1);
    if (end == std::string::npos) return {};
    return header.substr(q + 1, end - q - 1);
}

bool LoadNpyCodes(const std::string& path, int64_t expected_rows,
                  std::vector<int64_t>* codes, int64_t* frames) {
    const std::string blob = LoadFile(path);
    if (blob.size() < 11 || static_cast<unsigned char>(blob[0]) != 0x93 ||
        blob.compare(1, 5, "NUMPY") != 0) {
        MD_LOG_ERROR << "audio8: invalid npy magic: " << path << std::endl;
        return false;
    }
    const uint16_t hlen = static_cast<uint16_t>(
        (static_cast<unsigned char>(blob[9]) << 8) |
        static_cast<unsigned char>(blob[8]));
    if (10 + hlen > blob.size()) {
        MD_LOG_ERROR << "audio8: invalid npy header length: " << path << std::endl;
        return false;
    }
    const std::string header = blob.substr(10, hlen);
    std::vector<int64_t> shape;
    {
        const size_t p = header.find("shape");
        if (p == std::string::npos) return false;
        const size_t lp = header.find('(', p);
        const size_t rp = header.find(')', p);
        if (lp == std::string::npos || rp == std::string::npos || rp <= lp)
            return false;
        std::string dims = header.substr(lp + 1, rp - lp - 1);
        std::istringstream ss(dims);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            while (!tok.empty() &&
                   (tok.front() == ' ' || tok.front() == '\t' || tok.front() == '\n'))
                tok.erase(tok.begin());
            while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t' ||
                                    tok.back() == '\n'))
                tok.pop_back();
            if (tok.empty()) continue;
            shape.push_back(std::strtoll(tok.c_str(), nullptr, 10));
        }
    }
    size_t total = 1;
    for (const auto d : shape) total *= static_cast<size_t>(d);
    const std::string descr = NpyFindQuoted(header, "descr");
    const size_t data_off = 10 + hlen;
    codes->assign(total, 0);
    if (descr == "<u2") {
        const uint16_t* src =
            reinterpret_cast<const uint16_t*>(blob.data() + data_off);
        for (size_t k = 0; k < total; ++k) (*codes)[k] = src[k];
    } else if (descr == "<i8") {
        std::memcpy(codes->data(), blob.data() + data_off, total * sizeof(int64_t));
    } else if (descr == "<i2") {
        const int16_t* src =
            reinterpret_cast<const int16_t*>(blob.data() + data_off);
        for (size_t k = 0; k < total; ++k) (*codes)[k] = src[k];
    } else if (descr == "<f4") {
        const float* src = reinterpret_cast<const float*>(blob.data() + data_off);
        for (size_t k = 0; k < total; ++k) (*codes)[k] = static_cast<int64_t>(src[k]);
    } else {
        MD_LOG_ERROR << "audio8: unsupported npy descr '" << descr << "' in " << path
                     << std::endl;
        return false;
    }
    if (static_cast<int64_t>(shape.size()) != 2 || shape[0] != expected_rows) {
        MD_LOG_ERROR << "audio8: unexpected voice codes shape in " << path
                     << " (rows=" << (shape.empty() ? -1 : shape[0])
                     << " expected=" << expected_rows << ")" << std::endl;
        return false;
    }
    *frames = shape[1];
    if (*frames <= 0) return false;
    return true;
}

}  // namespace

// ================= Audio8::Impl =================

class Audio8::Impl {
public:
    audio8::Audio8Manifest manifest;
    audio8::Audio8Runtime runtime;

    bool loaded_ = false;

    // ---------- 设备(GPU-only) ----------
    modeldeploy::Device device_{modeldeploy::Device::CPU};
    int32_t device_id_ = 0;

    // ---------- tokenizer ----------
    std::unordered_map<std::string, int64_t> token2id_;
    std::unordered_map<std::string, int32_t> merges_rank_;
    std::string byte_to_unicode_[256];

    struct AddedToken {
        std::string content;
        int64_t id = 0;
    };
    struct TrieNode {
        std::unordered_map<uint8_t, int32_t> next;
        int32_t token_index = -1;
    };
    std::vector<AddedToken> added_tokens_;
    std::vector<TrieNode> trie_;
    std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

    // ---------- 生成参数（与 runtime.py 迭代器默认一致） ----------
    static constexpr int64_t kDefaultMaxNewTokens = 1024;
    static constexpr uint64_t kDefaultSeed = 42;
    static constexpr float kTemperature = 0.7f;
    static constexpr float kTopP = 0.9f;
    static constexpr int32_t kTopK = 50;

    bool Init(const std::string& model_dir, const RuntimeOption& opt) {
        if (!audio8::Audio8Manifest::FromJson(model_dir + "/runtime_manifest.json",
                                              model_dir, &manifest)) {
            MD_LOG_ERROR << "audio8: failed to parse manifest" << std::endl;
            return false;
        }
        const int32_t threads = opt.cpu_thread_num > 0 ? opt.cpu_thread_num : 0;
        device_ = opt.device;
        device_id_ = opt.device_id;
        if (!runtime.Load(manifest, threads, opt.device, opt.device_id)) {
            MD_LOG_ERROR << "audio8: failed to load onnx runtime" << std::endl;
            return false;
        }
        if (!InitTokenizer(manifest.tokenizer_path)) {
            MD_LOG_ERROR << "audio8: failed to load tokenizer" << std::endl;
            return false;
        }
        loaded_ = true;
        return true;
    }

    // ---------- tokenizer ----------
    void BuildBytesToUnicode() {
        std::vector<uint32_t> bs;
        bs.reserve(256);
        for (uint32_t c = 33; c <= 126; ++c) bs.push_back(c);
        for (uint32_t c = 161; c <= 172; ++c) bs.push_back(c);
        for (uint32_t c = 174; c <= 255; ++c) bs.push_back(c);
        std::vector<uint32_t> cs = bs;
        cs.reserve(256);
        uint32_t missing = 0;
        auto in_bs = [&](uint32_t b) {
            return std::find(bs.begin(), bs.end(), b) != bs.end();
        };
        for (uint32_t b = 0; b <= 255; ++b) {
            if (!in_bs(b)) {
                bs.push_back(b);
                cs.push_back(256 + missing);
                ++missing;
            }
        }
        for (size_t id = 0; id < bs.size(); ++id) {
            std::string u;
            AppendUtf8(cs[id], &u);
            byte_to_unicode_[bs[id]] = u;
        }
    }

    bool InitTokenizer(const std::string& path) {
        const std::string blob = LoadFile(path);
        if (blob.empty()) {
            MD_LOG_ERROR << "audio8: cannot open tokenizer " << path << std::endl;
            return false;
        }
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(blob);
        } catch (const std::exception& e) {
            MD_LOG_ERROR << "audio8: tokenizer parse failed: " << e.what() << std::endl;
            return false;
        }
        const auto& model = root["model"];
        if (model["vocab"].is_object()) {
            for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
                if (it.value().is_number_integer())
                    token2id_[it.key()] = it.value().get<int64_t>();
            }
        }
        int32_t rank = 0;
        if (model["merges"].is_array()) {
            for (const auto& m : model["merges"]) {
                std::string a, b;
                if (m.is_string()) {
                    const std::string line = m.get<std::string>();
                    const size_t sp = line.find(' ');
                    if (sp == std::string::npos || sp == 0) continue;
                    a = line.substr(0, sp);
                    b = line.substr(sp + 1);
                } else if (m.is_array() && m.size() == 2 && m[0].is_string() &&
                           m[1].is_string()) {
                    a = m[0].get<std::string>();
                    b = m[1].get<std::string>();
                } else {
                    continue;
                }
                merges_rank_[a + "\t" + b] = rank++;
            }
        }
        if (root["added_tokens"].is_array()) {
            for (const auto& at : root["added_tokens"]) {
                if (!at.is_object() || !at.contains("content") || !at.contains("id"))
                    continue;
                AddedToken t;
                t.content = at["content"].get<std::string>();
                t.id = at["id"].get<int64_t>();
                added_tokens_.push_back(std::move(t));
            }
        }
        BuildBytesToUnicode();
        trie_.clear();
        trie_.push_back(TrieNode{});
        for (int32_t ti = 0; ti < static_cast<int32_t>(added_tokens_.size()); ++ti) {
            int32_t node = 0;
            for (const unsigned char b : added_tokens_[ti].content) {
                const auto it = trie_[node].next.find(b);
                if (it == trie_[node].next.end()) {
                    const int32_t node_id = static_cast<int32_t>(trie_.size());
                    trie_.push_back(TrieNode{});
                    trie_[node].next.emplace(b, node_id);
                    node = node_id;
                } else {
                    node = it->second;
                }
            }
            trie_[node].token_index = ti;
        }
        if (token2id_.empty()) {
            MD_LOG_ERROR << "audio8: tokenizer vocab empty" << std::endl;
            return false;
        }
        MD_LOG_INFO << "audio8: tokenizer loaded vocab=" << token2id_.size()
                    << " merges=" << merges_rank_.size()
                    << " added=" << added_tokens_.size() << std::endl;
        return true;
    }

    std::string ByteLevelEncode(const std::string& token) const {
        std::string out;
        out.reserve(token.size() * 2);
        for (const unsigned char b : token) out.append(byte_to_unicode_[b]);
        return out;
    }

    std::string MergeKey(const std::string& a, const std::string& b) const {
        std::string k = a;
        k.push_back('\t');
        k.append(b);
        return k;
    }

    const std::vector<std::string>& BpeEncode(const std::string& word) {
        const auto cached = bpe_cache_.find(word);
        if (cached != bpe_cache_.end()) return cached->second;
        std::vector<std::string> symbols = SplitUtf8ToChars(word);
        if (symbols.size() > 1) {
            while (symbols.size() > 1) {
                int32_t best_rank = std::numeric_limits<int32_t>::max();
                int32_t best_pos = -1;
                for (int32_t k = 0; k + 1 < static_cast<int32_t>(symbols.size()); ++k) {
                    const auto fit = merges_rank_.find(MergeKey(symbols[k], symbols[k + 1]));
                    if (fit != merges_rank_.end() && fit->second < best_rank) {
                        best_rank = fit->second;
                        best_pos = k;
                    }
                }
                if (best_pos < 0) break;
                symbols[best_pos].append(symbols[best_pos + 1]);
                symbols.erase(symbols.begin() + best_pos + 1);
            }
        }
        return bpe_cache_.emplace(word, std::move(symbols)).first->second;
    }

    // 对齐 runtime.py 的 tokenizer.encode(text, add_special_tokens=False)
    std::vector<int64_t> Encode(const std::string& text) {
        std::vector<int64_t> out;
        if (token2id_.empty() || text.empty()) return out;
        auto match_added = [&](size_t pos, size_t* matched_len, int32_t* token_index) {
            if (trie_.empty()) return;
            int32_t node = 0;
            int32_t best_idx = -1;
            int32_t best_len = 0;
            size_t i = pos;
            while (i < text.size()) {
                const uint8_t b = static_cast<uint8_t>(text[i]);
                const auto it = trie_[node].next.find(b);
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
        auto flush_segment = [&](const std::string& seg) {
            const auto pieces = SplitByQwenPattern(seg);
            for (const auto& p : pieces) {
                const std::string bl = ByteLevelEncode(p);
                const auto& bpe_toks = BpeEncode(bl);
                for (const auto& bt : bpe_toks) {
                    const auto it = token2id_.find(bt);
                    if (it == token2id_.end()) continue;
                    out.push_back(it->second);
                }
            }
        };
        size_t pos = 0;
        size_t last = 0;
        while (pos < text.size()) {
            size_t mlen = 0;
            int32_t tidx = -1;
            match_added(pos, &mlen, &tidx);
            if (mlen > 0 && tidx >= 0) {
                if (pos > last) flush_segment(text.substr(last, pos - last));
                out.push_back(added_tokens_[static_cast<size_t>(tidx)].id);
                pos += mlen;
                last = pos;
                continue;
            }
            ++pos;
        }
        if (last < text.size()) flush_segment(text.substr(last));
        return out;
    }

    // ---------- voice ----------
    struct Voice {
        std::vector<int64_t> codes;  // [num_codebooks, frames] 扁平
        int64_t frames = 0;
        std::string reference_text;
    };

    bool LoadVoice(const std::string& name, Voice* out) {
        if (name.empty() || name == "." || name == ".." ||
            name.find('/') != std::string::npos ||
            name.find('\\') != std::string::npos ||
            name.find("..") != std::string::npos) {
            MD_LOG_ERROR << "audio8: invalid voice name '" << name << "'" << std::endl;
            return false;
        }
        const std::string dir = manifest.voices_dir + "/" + name;
        const std::string meta_blob = LoadFile(dir + "/meta.json");
        if (meta_blob.empty()) {
            MD_LOG_ERROR << "audio8: voice not found: " << name << " (no meta.json)"
                         << std::endl;
            return false;
        }
        nlohmann::json meta;
        try {
            meta = nlohmann::json::parse(meta_blob);
        } catch (const std::exception& e) {
            MD_LOG_ERROR << "audio8: invalid meta.json for voice " << name << ": "
                         << e.what() << std::endl;
            return false;
        }
        const std::string ref = meta.value("reference_text", "");
        if (ref.empty()) {
            MD_LOG_ERROR << "audio8: voice " << name << " has empty reference_text"
                         << std::endl;
            return false;
        }
        std::vector<int64_t> codes;
        int64_t frames = 0;
        if (!LoadNpyCodes(dir + "/codes.npy", manifest.num_codebooks, &codes, &frames))
            return false;
        out->codes = std::move(codes);
        out->frames = frames;
        out->reference_text = ref;
        return true;
    }

    // ---------- prompt 构造（prompt.py PromptBuilder.build） ----------
    bool BuildPrompt(const std::string& target_text, const std::string& reference_text,
                     const std::vector<int64_t>& reference_codes, int64_t ref_frames,
                     std::vector<int64_t>* prompt, int64_t* prompt_len) {
        const int64_t num_cb = manifest.num_codebooks;
        if (ref_frames <= 0 ||
            static_cast<int64_t>(reference_codes.size()) != num_cb * ref_frames) {
            MD_LOG_ERROR << "audio8: invalid reference codes" << std::endl;
            return false;
        }
        const std::vector<std::string> prefix_parts = {
            "<|im_start|>system\n",
            "convert the provided text to speech reference to the following:\n\nText:\n",
            FormatReferenceText(reference_text), "\n\nSpeech:\n"};
        const std::string suffix_parts[] = {
            "<|im_end|>\n", "<|im_start|>user\n", CleanText(target_text),
            "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"};
        std::vector<int64_t> prefix, suffix;
        for (const auto& part : prefix_parts) {
            const auto ids = Encode(part);
            prefix.insert(prefix.end(), ids.begin(), ids.end());
        }
        for (const auto& part : suffix_parts) {
            const auto ids = Encode(part);
            suffix.insert(suffix.end(), ids.begin(), ids.end());
        }
        std::vector<int64_t> row0;
        row0.reserve(prefix.size() + static_cast<size_t>(ref_frames) + suffix.size());
        row0.insert(row0.end(), prefix.begin(), prefix.end());
        for (int64_t j = 0; j < ref_frames; ++j)
            row0.push_back(reference_codes[j] + manifest.semantic_begin_id);
        row0.insert(row0.end(), suffix.begin(), suffix.end());
        const int64_t rows = num_cb + 1;
        const int64_t T = static_cast<int64_t>(row0.size());
        const int64_t begin = static_cast<int64_t>(prefix.size());
        prompt->assign(static_cast<size_t>(rows * T), 0);
        for (int64_t j = 0; j < T; ++j) (*prompt)[j] = row0[j];
        for (int64_t i = 1; i < rows; ++i) {
            for (int64_t j = 0; j < ref_frames; ++j)
                (*prompt)[i * T + begin + j] =
                    reference_codes[(i - 1) * ref_frames + j];
        }
        *prompt_len = T;
        return true;
    }

    // ---------- 语义采样（runtime.py _sample_semantic） ----------
    int64_t SampleSemantic(const std::vector<float>& logits,
                           const std::vector<int64_t>& previous, float temperature,
                           float top_p, int32_t top_k, std::mt19937_64& rng) {
        const int64_t begin = manifest.semantic_begin_id;
        const int64_t end = manifest.semantic_end_id;
        const int64_t stop = manifest.im_end_id;
        const int64_t count = (end - begin + 1) + 1;
        std::vector<int64_t> allowed_ids(count);
        for (int64_t i = 0; i < count - 1; ++i) allowed_ids[i] = begin + i;
        allowed_ids[count - 1] = stop;
        std::vector<float> allowed_logits(count);
        if (manifest.slow_logits_layout == "semantic_then_eos") {
            if (static_cast<int64_t>(logits.size()) != count) {
                MD_LOG_ERROR << "audio8: unexpected slow logits size " << logits.size()
                             << " expected " << count << std::endl;
                return stop;
            }
            allowed_logits = logits;
        } else {
            for (int64_t i = 0; i < count; ++i)
                allowed_logits[i] = logits[allowed_ids[i]];
        }
        std::vector<float> buf = allowed_logits;
        const int64_t normal_index =
            sampling::GumbelSample(buf.data(), count, temperature, top_p, top_k, rng);
        const int64_t normal = allowed_ids[normal_index];
        buf = allowed_logits;
        const int64_t high_index =
            sampling::GumbelSample(buf.data(), count, 1.0f, 0.9f, top_k, rng);
        const int64_t high = allowed_ids[high_index];
        if (normal >= begin && normal <= end &&
            std::find(previous.begin(), previous.end(), normal) != previous.end())
            return high;
        return normal;
    }

    // ---------- 生成循环（对齐 iter_codes） ----------
    // sink 每帧回调（10 个 codebook id）；返回 false 立即中止。
    bool RunGeneration(const std::string& text, const std::string& voice,
                       const std::function<bool(const int64_t* frame)>& sink) {
        Voice v;
        if (!LoadVoice(voice, &v)) return false;
        std::vector<int64_t> prompt;
        int64_t prompt_len = 0;
        if (!BuildPrompt(text, v.reference_text, v.codes, v.frames, &prompt,
                         &prompt_len))
            return false;
        const int64_t max_seq_len = manifest.max_seq_len;
        if (prompt_len >= max_seq_len) {
            MD_LOG_ERROR << "audio8: prompt length " << prompt_len
                         << " exceeds max sequence length " << max_seq_len
                         << ", refusing to truncate" << std::endl;
            return false;
        }
        const int64_t max_new =
            std::min<int64_t>(kDefaultMaxNewTokens, max_seq_len - prompt_len);

        const int64_t num_layers = manifest.num_layers;
        const int64_t seg = manifest.n_local_heads * max_seq_len * manifest.head_dim;
        std::vector<uint16_t> slow_cache(static_cast<size_t>(2 * num_layers * seg), 0);
        std::vector<int64_t> positions(prompt_len);
        for (int64_t j = 0; j < prompt_len; ++j) positions[j] = j;

        std::vector<float> logits;
        std::vector<uint16_t> hidden;
        // 硬性指标:audio8 仅 GPU;KV 常驻 GPU,故 prefill 前即建 GPU 状态并接管 KV。
        if (device_ != modeldeploy::Device::GPU) {
            MD_LOG_ERROR << "audio8: TTS is GPU-only; device is not GPU" << std::endl;
            return false;
        }
        audio8::Audio8Runtime::Audio8GpuStatePtr gstate =
            runtime.MakeGpuState(device_, device_id_);
        if (!gstate) {
            MD_LOG_ERROR << "audio8: GPU state unavailable; refusing CPU fallback"
                         << std::endl;
            return false;
        }
        const bool use_gpu = true;
        const bool prefill_ok = use_gpu
            ? runtime.SlowStepGpu(gstate.get(), prompt, positions, &logits, nullptr)
            : runtime.SlowStep(prompt, positions, &slow_cache, &logits, &hidden);
        if (!prefill_ok) {
            MD_LOG_ERROR << "audio8: slow prefill failed" << std::endl;
            return false;
        }

        std::mt19937_64 rng(kDefaultSeed);
        const int64_t begin = manifest.semantic_begin_id;
        const int64_t stop = manifest.im_end_id;
        const int64_t codebook_size = manifest.codebook_size;
        const int64_t num_codebooks = manifest.num_codebooks;

        std::vector<int64_t> previous;
        std::vector<int64_t> column(num_codebooks + 1);
        std::vector<int64_t> frame(num_codebooks);
        std::vector<uint16_t> fast_cache;
        const int64_t fseg =
            manifest.fast_n_local_heads * num_codebooks * manifest.fast_head_dim;
        fast_cache.assign(static_cast<size_t>(2 * manifest.num_fast_layers * fseg), 0);
        std::vector<float> fast_logits;

        for (int64_t step = 0; step < max_new; ++step) {
            const int64_t semantic =
                SampleSemantic(logits, previous, kTemperature, kTopP, kTopK, rng);
            if (semantic == stop) break;
            previous.push_back(semantic);
            if (static_cast<int64_t>(previous.size()) > 10)
                previous.erase(previous.begin(), previous.begin() + 1);
            std::fill(fast_cache.begin(), fast_cache.end(), 0);
            const bool boot_ok = use_gpu
                ? runtime.FastStepGpu(gstate.get(), 0, true, 0, hidden, &fast_logits)
                : runtime.FastStep(0, true, 0, hidden, &fast_cache, &fast_logits);
            if (!boot_ok) {
                MD_LOG_ERROR << "audio8: fast bootstrap failed" << std::endl;
                return false;
            }
            int64_t token = std::min(std::max(semantic - begin, int64_t(0)),
                                     codebook_size - 1);
            frame[0] = token;
            for (int64_t fast_pos = 1; fast_pos < num_codebooks; ++fast_pos) {
                const bool fok = use_gpu
                    ? runtime.FastStepGpu(gstate.get(), token, false, fast_pos, hidden,
                                          &fast_logits)
                    : runtime.FastStep(token, false, fast_pos, hidden, &fast_cache,
                                       &fast_logits);
                if (!fok) {
                    MD_LOG_ERROR << "audio8: fast step failed" << std::endl;
                    return false;
                }
                std::vector<float> buf = fast_logits;
                token = sampling::GumbelSample(buf.data(), codebook_size, kTemperature,
                                               kTopP, kTopK, rng);
                frame[fast_pos] = token;
            }
            if (sink && !sink(frame.data())) return false;
            if (step + 1 >= max_new) break;
            column[0] = semantic;
            for (int64_t k = 0; k < num_codebooks; ++k) column[k + 1] = frame[k];
            const std::vector<int64_t> one_pos{prompt_len + step};
            const bool s_ok = use_gpu
                ? runtime.SlowStepGpu(gstate.get(), column, one_pos, &logits, nullptr)
                : runtime.SlowStep(column, one_pos, &slow_cache, &logits, &hidden);
            if (!s_ok) {
                MD_LOG_ERROR << "audio8: slow step failed" << std::endl;
                return false;
            }
        }
        return true;
    }
};

// ================= Audio8 =================

Audio8::Audio8() = default;
Audio8::~Audio8() = default;

bool Audio8::Load(const std::string& model_dir, const RuntimeOption& opt) {
    if (model_dir.empty()) {
        MD_LOG_ERROR << "audio8: empty model_dir" << std::endl;
        return false;
    }
    auto impl = std::make_unique<Impl>();
    if (!impl->Init(model_dir, opt)) return false;
    impl_ = std::shared_ptr<Impl>(impl.release());
    return true;
}

std::unique_ptr<Audio8> Audio8::clone() const {
    auto c = std::make_unique<Audio8>();
    c->impl_ = impl_;  // 共享 session 的实现浅共享
    return c;
}

int32_t Audio8::get_sample_rate() const {
    return impl_ ? static_cast<int32_t>(impl_->manifest.sample_rate) : 44100;
}

bool Audio8::predict(const std::string& text, const std::string& voice, float,
                     std::vector<float>* out) {
    if (!impl_ || !impl_->loaded_) {
        MD_LOG_ERROR << "audio8: not initialized" << std::endl;
        return false;
    }
    if (out == nullptr || text.empty()) {
        MD_LOG_ERROR << "audio8: predict invalid args" << std::endl;
        return false;
    }
    const int64_t num_cb = impl_->manifest.num_codebooks;
    std::vector<std::vector<int64_t>> frames;
    const bool ok = impl_->RunGeneration(text, voice, [&frames, num_cb](const int64_t* f) {
        frames.emplace_back(f, f + num_cb);
        return true;
    });
    if (!ok) return false;
    if (frames.empty()) {
        MD_LOG_ERROR << "audio8: model produced no codec frames" << std::endl;
        return false;
    }
    const int64_t n = static_cast<int64_t>(frames.size());
    std::vector<int64_t> codes(static_cast<size_t>(num_cb * n));
    for (int64_t c = 0; c < n; ++c)
        for (int64_t r = 0; r < num_cb; ++r) codes[r * n + c] = frames[c][r];
    if (!impl_->runtime.DecodeCodes(codes, n, out)) {
        MD_LOG_ERROR << "audio8: decode failed" << std::endl;
        return false;
    }
    return true;
}

bool Audio8::predict_stream(const std::string& text, const std::string& voice, float,
                            int chunk_frames,
                            const std::function<bool(const float*, int, float)>& cb) {
    if (!impl_ || !impl_->loaded_ || !cb) {
        MD_LOG_ERROR << "audio8: predict_stream invalid state" << std::endl;
        return false;
    }
    const int64_t num_cb = impl_->manifest.num_codebooks;
    const int64_t hop = impl_->manifest.codec_hop_length;
    const int64_t context = impl_->manifest.stream_context_frames;
    const int64_t guard = impl_->manifest.stream_guard_frames * hop;

    std::vector<std::vector<int64_t>> all_frames;

    auto decode_window = [&](int64_t start_frame, std::vector<float>* audio) -> bool {
        const int64_t nw = static_cast<int64_t>(all_frames.size()) - start_frame;
        if (nw <= 0) return false;
        std::vector<int64_t> codes(static_cast<size_t>(num_cb * nw));
        for (int64_t r = 0; r < num_cb; ++r)
            for (int64_t c = 0; c < nw; ++c)
                codes[r * nw + c] = all_frames[start_frame + c][r];
        return impl_->runtime.DecodeCodes(codes, nw, audio);
    };

    if (chunk_frames <= 0) {
        // batch 语义：一次性合成，单次回调
        const bool gen_ok =
            impl_->RunGeneration(text, voice, [&all_frames, num_cb](const int64_t* f) {
                all_frames.emplace_back(f, f + num_cb);
                return true;
            });
        if (!gen_ok) return false;
        if (all_frames.empty()) {
            MD_LOG_ERROR << "audio8: model produced no codec frames" << std::endl;
            return false;
        }
        std::vector<float> audio;
        if (!decode_window(0, &audio)) return false;
        return cb(audio.data(), static_cast<int>(audio.size()), 1.0f);
    }

    // 官方 stream()：滑动窗口 + guard 去叠逐块回调
    const int64_t cf = std::max<int64_t>(1, chunk_frames);
    int64_t emitted = 0;
    const bool gen_ok =
        impl_->RunGeneration(text, voice, [&](const int64_t* f) {
            all_frames.emplace_back(f, f + num_cb);
            if (static_cast<int64_t>(all_frames.size()) % cf != 0) return true;
            const int64_t len = static_cast<int64_t>(all_frames.size());
            const int64_t start_frame = std::max<int64_t>(0, len - context - cf);
            std::vector<float> audio;
            if (!decode_window(start_frame, &audio)) return false;
            const int64_t absolute_start = start_frame * hop;
            const int64_t stable_end =
                absolute_start +
                std::max<int64_t>(0, static_cast<int64_t>(audio.size()) - guard);
            const int64_t beg = std::max<int64_t>(0, emitted - absolute_start);
            const int64_t end = std::max<int64_t>(beg, stable_end - absolute_start);
            if (end <= beg) return true;
            const float progress =
                static_cast<float>(len) / static_cast<float>(Impl::kDefaultMaxNewTokens);
            if (!cb(audio.data() + beg, static_cast<int>(end - beg), progress))
                return false;
            emitted += (end - beg);
            return true;
        });
    if (!gen_ok) return false;
    if (all_frames.empty()) {
        MD_LOG_ERROR << "audio8: model produced no codec frames" << std::endl;
        return false;
    }
    const int64_t len = static_cast<int64_t>(all_frames.size());
    const int64_t start_frame = std::max<int64_t>(0, len - context - cf);
    std::vector<float> audio;
    if (!decode_window(start_frame, &audio)) return false;
    const int64_t absolute_start = start_frame * hop;
    const int64_t beg = std::max<int64_t>(0, emitted - absolute_start);
    if (beg < static_cast<int64_t>(audio.size())) {
        if (!cb(audio.data() + beg, static_cast<int>(audio.size() - beg), 1.0f))
            return false;
    }
    return true;
}

}  // namespace modeldeploy::audio::tts
