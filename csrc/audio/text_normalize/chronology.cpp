//
// Created by aichao on 2025/5/21.
//

#include <regex>
#include <string>
#include <unordered_map>
#include "csrc/audio/text_normalize/number.h"
#include "csrc/audio/text_normalize/chronology.h"

namespace modeldeploy::audio {
    // 时刻表达式 (使用宽字符 wregex)
    std::wregex RE_TIME(LR"(([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?)");
    // 时间范围，如8:30-12:30
    std::wregex RE_TIME_RANGE(
        LR"(([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?(~|-)([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?)");
    // 日期表达式
    std::wregex RE_DATE(LR"((\d{4}|\d{2})年((0?[1-9]|1[0-2])月)?(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?)");
    // 用 / 或者 - 分隔的 YY/MM/DD 或者
    std::wregex RE_DATE2(LR"((\d{4})([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01]))");

    // 特殊时间数字转换 (改为宽字符版本)
    std::wstring _time_num2str(const std::wstring& num_string) {
        std::wstring result = num2str(num_string.substr(num_string.find_first_not_of(L'0')));
        if (num_string[0] == L'0') {
            result = DIGITS[L'0'] + result;
        }
        return result;
    }

    // 替换时间 (改为宽字符版本)
    std::wstring replace_time(const std::wsmatch& match) {
        const bool is_range = match.size() > 5;

        const std::wstring hour = match.str(1);
        const std::wstring minute = match.str(2);
        const std::wstring second = match.str(4);

        std::wstring result = num2str(hour) + L"点";
        if (!minute.empty() && minute != L"00") {
            if (std::stoi(minute) == 30) {
                result += L"半";
            }
            else {
                result += _time_num2str(minute) + L"分";
            }
        }
        if (!second.empty() && second != L"00") {
            result += _time_num2str(second) + L"秒";
        }

        if (is_range) {
            const std::wstring hour_2 = match.str(6);
            const std::wstring minute_2 = match.str(7);
            const std::wstring second_2 = match.str(9);

            result += L"至" + num2str(hour_2) + L"点";
            if (!minute_2.empty() && minute_2 != L"00") {
                if (std::stoi(minute_2) == 30) {
                    result += L"半";
                }
                else {
                    result += _time_num2str(minute_2) + L"分";
                }
            }
            if (!second_2.empty() && second_2 != L"00") {
                result += _time_num2str(second_2) + L"秒";
            }
        }

        return match.prefix().str() + result + match.suffix().str();
    }

    // 替换日期 (改为宽字符版本)
    std::wstring replace_date(const std::wsmatch& match) {
        const std::wstring year = match.str(1);
        const std::wstring month = match.str(3);
        const std::wstring day = match.str(5);
        std::wstring result;
        if (!year.empty()) {
            result += verbalize_digit(year) + L"年";
        }
        if (!month.empty()) {
            result += verbalize_cardinal(month) + L"月";
        }
        if (!day.empty()) {
            result += verbalize_cardinal(day) + L"日";
        }
        return match.prefix().str() + result + match.suffix().str();
    }

    // 替换日期2 (改为宽字符版本)
    std::wstring replace_date2(const std::wsmatch& match) {
        const std::wstring year = match.str(1);
        const std::wstring month = match.str(3);
        const std::wstring day = match.str(4);
        std::wstring result;
        if (!year.empty()) {
            result += verbalize_digit(year) + L"年";
        }
        if (!month.empty()) {
            result += verbalize_cardinal(month) + L"月";
        }
        if (!day.empty()) {
            result += verbalize_cardinal(day) + L"日";
        }
        return match.prefix().str() + result + match.suffix().str();
    }
}
