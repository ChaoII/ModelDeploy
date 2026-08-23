#include "nlp/tools/stats.h"
#include <sstream>
namespace modeldeploy::nlp::tool {
size_t Stats::char_count(const std::string& text) {
    size_t n = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = (unsigned char)text[i];
        if ((c & 0xC0) != 0x80) ++n;
    }
    return n;
}
size_t Stats::word_count(const std::string& text) {
    std::istringstream iss(text); size_t n = 0; std::string w;
    while (iss >> w) ++n;
    return n;
}
size_t Stats::sentence_count(const std::string& text) {
    size_t n = 0;
    bool in = false;
    for (size_t i = 0; i < text.size(); ++i) {
        const unsigned char c = (unsigned char)text[i];
        bool term = false;
        if (c == '\n') term = true;
        else if (i + 2 < text.size()) {
            const unsigned char d1 = (unsigned char)text[i+1], d2 = (unsigned char)text[i+2];
            if (c == 0xE3 && d1 == 0x80 && d2 == 0x82) term = true;          // 。
            else if (c == 0xEF && d1 == 0xBC && (d2 == 0x81 || d2 == 0x9F || d2 == 0x9B)) term = true; // ！？；
        }
        if (term) {
            if (in) { ++n; in = false; }
            if (text[i] != '\n') i += 2;
        } else {
            in = true;
        }
    }
    if (in) ++n;
    return n;
}
} // namespace modeldeploy::nlp::tool
