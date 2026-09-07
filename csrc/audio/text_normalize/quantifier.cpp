//
// Created by aichao on 2025/5/21.
//

#include <algorithm>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "audio/text_normalize/number.h"

namespace modeldeploy::audio {
    std::unordered_map<std::wstring, std::wstring> measure_dict = {
        {L"cm2", L"平方厘米"},
        {L"cm²", L"平方厘米"},
        {L"cm3", L"立方厘米"},
        {L"cm³", L"立方厘米"},
        {L"m2", L"平方米"},
        {L"m²", L"平方米"},
        {L"m³", L"立方米"},
        {L"m3", L"立方米"},
        // 常用度量符号 -> 中文量词（长度降序替换，避免 cm/m、mm/m 等子串冲突）
        {L"km", L"千米"},
        {L"kg", L"千克"},
        {L"cm", L"厘米"},
        {L"mm", L"毫米"},
        {L"ml", L"毫升"},
        {L"mg", L"毫克"},
        {L"min", L"分钟"},
        {L"db", L"分贝"},
        {L"ds", L"毫秒"},
        {L"m", L"米"},
        {L"s", L"秒"},
        {L"h", L"小时"},
    };

    // 使用宽字符版本的正则表达式
    std::wregex re_temperature(LR"((-?)(\d+(\.\d+)?)(°C|℃|度|摄氏度))");

    std::wstring replace_temperature(const std::wsmatch& match) {
        std::wstring sign = match.str(1);
        std::wstring temperature = match.str(2);
        std::wstring unit = match.str(4);
        sign = sign.empty() ? L"" : L"零下";
        temperature = num2str(temperature); // 假设 num2str 返回宽字符串
        unit = unit == L"摄氏度" ? L"摄氏度" : L"度";
        return match.prefix().str() + sign + temperature + unit + match.suffix().str();
    }

    std::wstring replace_measure(std::wstring sentence) {
        // 按单位符号长度降序替换，确保 "cm"/"mm"/"min" 先于 "m"/"s" 处理，
        // 避免长单位被短单位子串抢先或破坏（如 "10cm" 不能被 "m" 拆成 "10c米"）。
        std::vector<std::pair<std::wstring, std::wstring>> items(measure_dict.begin(), measure_dict.end());
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
        for (const auto& q_notation : items) {
            size_t pos = 0;
            while ((pos = sentence.find(q_notation.first, pos)) != std::wstring::npos) {
                sentence.replace(pos, q_notation.first.length(), q_notation.second);
                pos += q_notation.second.length();
            }
        }
        return sentence;
    }
}
