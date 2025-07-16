//
// Created by aichao on 2025/5/21.
//

#include <cwctype>  // 包含 iswdigit 所需的头文件
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <locale>
#define NOGDI
#define NOCRYPT
#endif

#include "audio/text_normalize/number.h"

namespace modeldeploy::audio {
    // std::wregex re_mobile_phone(LR"((?<!\d)((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})(?!\d))");
    // std::wregex re_telephone(LR"((?<!\d)((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})(?!\d))");
    // std::wregex re_national_uniform_number(LR"((400)(-)?\d{3}(-)?\d{4}))");
    std::wregex re_mobile_phone(LR"((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})");
    std::wregex re_telephone(LR"((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})");
    std::wregex re_national_uniform_number(LR"(400-?\d{3}-?\d{4})");

    // 手动检查是否有前后数字
    bool is_valid_phone_number(const std::wstring& text, const std::wsmatch& match) {
        // 检查手机号前面和后面的字符是否为数字
        if (match.position() > 0 && std::iswdigit(text[match.position() - 1])) {
            return false; // 前面有数字，不符合要求
        }
        if (match.position() + match.length() < text.size() && std::iswdigit(text[match.position() + match.length()])) {
            return false; // 后面有数字，不符合要求
        }
        return true;
    }

    std::wstring phone2str(const std::wstring& phone_string, bool mobile = true) {
        std::wstring result;
        if (mobile) {
            std::wstringstream ss(phone_string);
            std::wstring part;
            std::vector<std::wstring> parts;
            while (std::getline(ss, part, L' ')) {
                parts.push_back(verbalize_digit(part, true));
            }
            for (size_t i = 0; i < parts.size(); ++i) {
                result += parts[i];
                if (i != parts.size() - 1) {
                    result += L"，";
                }
            }
        }
        else {
            std::wstringstream ss(phone_string);
            std::wstring part;
            std::vector<std::wstring> parts;
            while (getline(ss, part, L'-')) {
                parts.push_back(verbalize_digit(part, true));
            }
            for (size_t i = 0; i < parts.size(); ++i) {
                result += parts[i];
                if (i != parts.size() - 1) {
                    result += L"，";
                }
            }
        }
        return result;
    }

    std::wstring replace_phone(const std::wsmatch& match) {
        return phone2str(match.str(0), false);
    }

    std::wstring replace_mobile(const std::wsmatch& match) {
        return phone2str(match.str(0));
    }

    std::wstring process_mobile_number(const std::wstring& phone) {
        // 匹配并处理国家代码部分
        const std::wregex re_country_code(LR"(\+?86 ?)");
        std::wstring result = regex_replace(phone, re_country_code, L"中国，");

        // 剩下的手机号部分
        const std::wregex re_mobile_body(LR"(\d{11})");
        std::wsmatch match;
        if (regex_search(result, match, re_mobile_body)) {
            const std::wstring mobile_number = match.str(0);
            // 使用 verbalize_digit 处理数字
            result = regex_replace(result, re_mobile_body, phone2str(mobile_number, true));
        }

        return result;
    }

    std::wstring process_landline_number(const std::wstring& phone) {
        // 匹配区号部分
        const std::wregex re_area_code(LR"(0\d{2,3})");
        std::wsmatch match;
        std::wstring result = phone;

        if (regex_search(phone, match, re_area_code)) {
            const std::wstring area_code = match.str(0);
            result = regex_replace(result, re_area_code, verbalize_digit(area_code) + L"，");
        }

        // 匹配剩余的电话号码部分
        const std::wregex re_phone_body(LR"(\d{6,8})");
        if (regex_search(result, match, re_phone_body)) {
            const std::wstring phone_body = match.str(0);
            result = regex_replace(result, re_phone_body, verbalize_digit(phone_body, true));
        }

        return result;
    }

    std::wstring process_uniform_number(const std::wstring& phone) {
        // 匹配400号码
        const std::wregex re_400(LR"(400)");
        std::wsmatch match;
        std::wstring result = phone;

        if (regex_search(phone, match, re_400)) {
            result = regex_replace(result, re_400, L"四，零，零");
        }

        // 匹配剩余的号码部分
        const std::wregex re_phone_body(LR"(\d{3}-\d{4})");
        if (regex_search(result, match, re_phone_body)) {
            const std::wstring phone_body = match.str(0);
            result = regex_replace(result, re_phone_body, verbalize_digit(phone_body, true));
        }

        return result;
    }
}
