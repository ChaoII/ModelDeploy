//
// Created by aichao on 2025/5/21.
//

#pragma once
#include <regex>

namespace modeldeploy::audio {
    extern std::wregex RE_TIME;
    // 时间范围，如8:30-12:30
    extern std::wregex RE_TIME_RANGE;
    // 日期表达式
    extern std::wregex RE_DATE;
    // 用 / 或者 - 分隔的 YY/MM/DD 或者
    extern std::wregex RE_DATE2;
    std::wstring _time_num2str(const std::wstring& num_string);
    std::wstring replace_time(const std::wsmatch& match);
    std::wstring replace_date(const std::wsmatch& match);
    std::wstring replace_date2(const std::wsmatch& match);
}
