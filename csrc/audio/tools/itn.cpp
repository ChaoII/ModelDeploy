#include "audio/tools/itn.h"
#include <cwchar>
#include <regex>
#include <functional>
#include <unordered_map>
#include <csrc/utils/utils.h>
#include <string>

namespace modeldeploy::audio::tool {
using modeldeploy::utf8_to_wstring;
using modeldeploy::wstring_to_string;

namespace {
int digit_of(wchar_t c) {
    switch (c) {
        case L'零': case L'〇': return 0;
        case L'一': return 1;
        case L'二': case L'两': return 2;
        case L'三': return 3;
        case L'四': return 4;
        case L'五': return 5;
        case L'六': return 6;
        case L'七': return 7;
        case L'八': return 8;
        case L'九': return 9;
        default: return -1;
    }
}
int year_digit(wchar_t c) { return digit_of(c); }

// 走一遍 wregex，对每个命中调用 fn(match) 替换，返回新串（避免 regex_replace 模板推导）。
std::wstring transform(const std::wstring& in, const std::wregex& re,
                       const std::function<std::wstring(const std::wsmatch&)>& fn) {
    std::wstring out;
    auto begin = std::wsregex_iterator(in.begin(), in.end(), re);
    const auto end = std::wsregex_iterator{};
    size_t last = 0;
    for (auto it = begin; it != end; ++it) {
        out += in.substr(last, (size_t)it->position() - last);
        out += fn(*it);
        last = (size_t)it->position() + (size_t)it->length();
    }
    out += in.substr(last);
    return out;
}
} // namespace

long long InverseTextNormalizer::parse_low(const std::wstring& s) {
    long long result = 0, cur = 0;
    for (wchar_t c : s) {
        switch (c) {
            case L'零': cur = 0; break;
            case L'十': result += (cur == 0 ? 1 : cur) * 10; cur = 0; break;
            case L'百': result += (cur == 0 ? 1 : cur) * 100; cur = 0; break;
            case L'千': result += (cur == 0 ? 1 : cur) * 1000; cur = 0; break;
            default: cur = cur * 10 + digit_of(c); break;
        }
    }
    result += cur;
    return result;
}

long long InverseTextNormalizer::parse_cn_number(const std::wstring& s) {
    if (s.empty()) return 0;
    long long total = 0;
    std::wstring rest = s;
    auto pos = rest.find(L'亿');
    if (pos != std::wstring::npos) {
        total += parse_low(rest.substr(0, pos)) * 100000000LL;
        rest = rest.substr(pos + 1);
    }
    pos = rest.find(L'万');
    if (pos != std::wstring::npos) {
        total += parse_low(rest.substr(0, pos)) * 10000LL;
        rest = rest.substr(pos + 1);
    }
    total += parse_low(rest);
    return total;
}

bool InverseTextNormalizer::convertible_run(const std::wstring& run) {
    return run.size() >= 2;
}

std::wstring InverseTextNormalizer::render_cn_number(const std::wstring& s) {
    if (s.find(L'万') == std::wstring::npos && s.find(L'亿') == std::wstring::npos) {
        return std::to_wstring(parse_cn_number(s));
    }
    std::wstring rest = s;
    long long yi = 0, wan = 0, low = 0;
    auto pos = rest.find(L'亿');
    if (pos != std::wstring::npos) { yi = parse_low(rest.substr(0, pos)); rest = rest.substr(pos + 1); }
    pos = rest.find(L'万');
    if (pos != std::wstring::npos) { wan = parse_low(rest.substr(0, pos)); rest = rest.substr(pos + 1); }
    low = parse_low(rest);

    std::wstring out;
    if (yi > 0) {
        out += std::to_wstring(yi) + L"亿";
        if (wan == 0 && low > 0) out += L"零";
    }
    if (wan > 0) out += std::to_wstring(wan) + L"万";
    if (low > 0) out += std::to_wstring(low);
    if (out.empty()) out = L"0";
    return out;
}

std::wstring InverseTextNormalizer::normalize(const std::wstring& in) const {
    using std::wregex;
    using std::wsmatch;
    std::wstring text = in;

    // 1) 百分之X -> X%
    text = transform(text, wregex(L"百分之([零一二两三四五六七八九十百千万亿]+)"),
        [](const wsmatch& m) -> std::wstring {
            return std::to_wstring(parse_cn_number(m[1].str())) + L"%";
        });

    // 2) 时间（带 分/半/刻）在“小数”之前处理
    text = transform(text, wregex(L"([零一二两三四五六七八九十]+)点(半|一刻|三刻)"),
        [](const wsmatch& m) -> std::wstring {
            long long h = parse_cn_number(m[1].str());
            const std::wstring q = m[2].str();
            if (q == L"半") return std::to_wstring(h) + L":30";
            if (q == L"一刻") return std::to_wstring(h) + L":15";
            return std::to_wstring(h) + L":45";
        });
    text = transform(text, wregex(L"([零一二两三四五六七八九十]+)点([零一二三四五六七八九十]+)分"),
        [](const wsmatch& m) -> std::wstring {
            long long h = parse_cn_number(m[1].str());
            long long mn = parse_cn_number(m[2].str());
            std::wstring mns = std::to_wstring(mn);
            if (mn < 10) mns = L"0" + mns;
            return std::to_wstring(h) + L":" + mns;
        });

    // 3) 小数：整数+点+个位数字（点后不能是 分/半/刻）
    text = transform(text, wregex(L"([零一二两三四五六七八九十百千万亿]+)点(?![刻分])([零一二三四五六七八九]+)"),
        [](const wsmatch& m) -> std::wstring {
            long long n = parse_cn_number(m[1].str());
            std::wstring d;
            for (wchar_t c : m[2].str()) d += wchar_t(L'0' + digit_of(c));
            return std::to_wstring(n) + L"." + d;
        });

    // 4) 整点：X点（点后不能是 分/半/刻/数字）
    text = transform(text, wregex(L"([零一二两三四五六七八九十]+)点(?![半一刻分零一二三四五六七八九])"),
        [](const wsmatch& m) -> std::wstring {
            return std::to_wstring(parse_cn_number(m[1].str())) + L"点";
        });

    // 5) 四位年份
    text = transform(text, wregex(L"([〇零一二两三四五六七八九]{4})年"),
        [](const wsmatch& m) -> std::wstring {
            std::wstring d;
            for (wchar_t c : m[1].str()) d += wchar_t(L'0' + year_digit(c));
            return d + L"年";
        });

    // 6) 年月日
    text = transform(text, wregex(L"([零一二两三四五六七八九十百千万亿]+)年([零一二两三四五六七八九十百千万亿]+)月([零一二两三四五六七八九十百千万亿]+)日"),
        [](const wsmatch& m) -> std::wstring {
            return std::to_wstring(parse_cn_number(m[1].str())) + L"年" +
                   std::to_wstring(parse_cn_number(m[2].str())) + L"月" +
                   std::to_wstring(parse_cn_number(m[3].str())) + L"日";
        });
    text = transform(text, wregex(L"([零一二两三四五六七八九十百千万亿]+)年([零一二两三四五六七八九十百千万亿]+)月"),
        [](const wsmatch& m) -> std::wstring {
            return std::to_wstring(parse_cn_number(m[1].str())) + L"年" +
                   std::to_wstring(parse_cn_number(m[2].str())) + L"月";
        });
    text = transform(text, wregex(L"([零一二两三四五六七八九十]+)月([零一二两三四五六七八九十]+)日"),
        [](const wsmatch& m) -> std::wstring {
            return std::to_wstring(parse_cn_number(m[1].str())) + L"月" +
                   std::to_wstring(parse_cn_number(m[2].str())) + L"日";
        });

    // 7) 序数
    text = transform(text, wregex(L"第([零一二两三四五六七八九十百千万亿]+)"),
        [](const wsmatch& m) -> std::wstring {
            return L"第" + std::to_wstring(parse_cn_number(m[1].str()));
        });

    // 8) 一般数字：仅转换长度>=2 的数词串，保留单字模糊词
    text = transform(text, wregex(L"[零一二两三四五六七八九十百千万亿]{2,}"),
        [](const wsmatch& m) -> std::wstring {
            if (!InverseTextNormalizer::convertible_run(m.str())) return m.str();
            return render_cn_number(m.str());
        });

    return text;
}

std::string InverseTextNormalizer::normalize(const std::string& text) const {
    return wstring_to_string(normalize(utf8_to_wstring(text)));
}
} // namespace modeldeploy::audio::tool
