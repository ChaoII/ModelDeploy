#include "nlp/tools/splitter.h"
namespace modeldeploy::nlp::tool {
namespace {
bool term_at(const std::string& t, size_t i) {
    const unsigned char c = (unsigned char)t[i];
    if (c == '\n') return true;
    if (i + 2 >= t.size()) return false;
    const unsigned char d1 = (unsigned char)t[i+1], d2 = (unsigned char)t[i+2];
    if (c == 0xE3 && d1 == 0x80 && d2 == 0x82) return true;   // 。(U+3002)
    if (c == 0xEF && d1 == 0xBC) return (d2 == 0x81 || d2 == 0x9F || d2 == 0x9B); // ！？；(U+FF01/FF1F/FF1B)
    return false;
}
} // namespace
std::vector<std::string> Splitter::split_sentences(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < text.size(); ++i) {
        if (term_at(text, i)) {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            if (text[i] == '\n') continue;
            i += 2; // skip 3-byte punctuation
        } else {
            cur.push_back(text[i]);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}
} // namespace modeldeploy::nlp::tool
