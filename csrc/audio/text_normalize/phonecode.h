//
// Created by aichao on 2025/5/21.
//

#pragma once

#include <regex>
#include <string>

namespace modeldeploy::audio {
    extern std::wregex re_mobile_phone;
    extern std::wregex re_telephone;
    extern std::wregex re_national_uniform_number;
    extern std::wregex re_country_code;

    std::wstring phone2str(const std::wstring& phone_string, bool mobile = true);
    std::wstring replace_phone(const std::wsmatch& match);
    std::wstring replace_mobile(const std::wsmatch& match);
    std::wstring process_mobile_number(const std::wstring& phone);
    std::wstring process_landline_number(const std::wstring& phone);
    std::wstring process_uniform_number(const std::wstring& phone);
    bool is_valid_phone_number(const std::wstring& text, const std::wsmatch& match);
}
