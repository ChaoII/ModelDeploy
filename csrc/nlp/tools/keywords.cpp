#include "nlp/tools/keywords.h"
#include <algorithm>
#include <map>
#include <sstream>
namespace modeldeploy::nlp::tool {
std::vector<std::pair<std::string,int>> Keywords::top(const std::string& text, int k) {
    std::map<std::string,int> freq;
    std::istringstream iss(text);
    std::string w;
    while (iss >> w) freq[w]++;
    std::vector<std::pair<std::string,int>> out(freq.begin(), freq.end());
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){ return a.second > b.second; });
    if ((int)out.size() > k) out.resize(k);
    return out;
}
} // namespace modeldeploy::nlp::tool
