#include "nlp/tools/normalizer.h"
namespace modeldeploy::nlp::tool {
// 精简适配层：全角 ASCII/数字/标点 → 半角（确定性、无字库依赖）。
std::string Normalizer::normalize(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = (unsigned char)text[i];
        if (c >= 0xEF && i + 2 < text.size()) {
            unsigned int cp = (((unsigned int)(unsigned char)text[i] & 0x0F) << 12)
                            | (((unsigned int)(unsigned char)text[i+1] & 0x3F) << 6)
                            | ((unsigned int)(unsigned char)text[i+2] & 0x3F);
            if (cp >= 0xFF01 && cp <= 0xFF5E) {
                char half = (char)(cp - 0xFF01 + 0x21);
                out.push_back(half);
            } else if (cp == 0x3000) { out.append(" "); }
            else { out.append(text, i, 3); }
            i += 2;
            continue;
        }
        if (c == ' ' || c == '\t') continue;
        out.push_back((char)c);
    }
    return out;
}
} // namespace modeldeploy::nlp::tool
