//
// Created by aichao on 2025/5/21.
//

#include <regex>
#include <string>
#include <unordered_map>
#ifdef _WIN32
#include <locale>
#define NOGDI
#define NOCRYPT
#endif

#include "audio/text_normalize/constant.h"

namespace modeldeploy::audio {
    std::unordered_map<wchar_t, wchar_t> F2H_ASCII_LETTERS;
    std::unordered_map<wchar_t, wchar_t> H2F_ASCII_LETTERS;
    std::unordered_map<wchar_t, wchar_t> F2H_DIGITS;
    std::unordered_map<wchar_t, wchar_t> H2F_DIGITS;
    std::unordered_map<wchar_t, wchar_t> F2H_PUNCTUATIONS;
    std::unordered_map<wchar_t, wchar_t> H2F_PUNCTUATIONS;
    std::unordered_map<wchar_t, wchar_t> F2H_SPACE;
    std::unordered_map<wchar_t, wchar_t> H2F_SPACE;
    // 初始化字符映射
    void initialize_constant_maps() {
        // ASCII 字母 全角 -> 半角
        for (wchar_t ch = L'a'; ch <= L'z'; ++ch) {
            F2H_ASCII_LETTERS[ch + 65248] = ch;
            H2F_ASCII_LETTERS[ch] = ch + 65248;
        }
        for (wchar_t ch = L'A'; ch <= L'Z'; ++ch) {
            F2H_ASCII_LETTERS[ch + 65248] = ch;
            H2F_ASCII_LETTERS[ch] = ch + 65248;
        }

        // 数字字符 全角 -> 半角
        for (wchar_t ch = L'0'; ch <= L'9'; ++ch) {
            F2H_DIGITS[ch + 65248] = ch;
            H2F_DIGITS[ch] = ch + 65248;
        }

        // 标点符号 全角 -> 半角
        const std::wstring punctuations = L"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
        for (wchar_t ch : punctuations) {
            F2H_PUNCTUATIONS[ch + 65248] = ch;
            H2F_PUNCTUATIONS[ch] = ch + 65248;
        }
        // 中文句号。 全角 -> 半角
        F2H_PUNCTUATIONS[L'\u3002'] = L'.';
        H2F_PUNCTUATIONS[L'.'] = L'\u3002';

        // 空格 全角 -> 半角
        F2H_SPACE[L'\u3000'] = L' ';
        H2F_SPACE[L' '] = L'\u3000';
    }

    // 将全角字符转换为半角
    std::wstring full_width_to_half_width(const std::wstring& input) {
        std::wstring result;
        for (wchar_t ch : input) {
            if (F2H_ASCII_LETTERS.find(ch) != F2H_ASCII_LETTERS.end()) {
                result += F2H_ASCII_LETTERS[ch];
            }
            else if (F2H_DIGITS.find(ch) != F2H_DIGITS.end()) {
                result += F2H_DIGITS[ch];
            }
            else if (F2H_PUNCTUATIONS.find(ch) != F2H_PUNCTUATIONS.end()) {
                result += F2H_PUNCTUATIONS[ch];
            }
            else if (F2H_SPACE.find(ch) != F2H_SPACE.end()) {
                result += F2H_SPACE[ch];
            }
            else {
                result += ch; // 如果没有匹配，保持原字符
            }
        }
        return result;
    }

    // 将半角字符转换为全角
    std::wstring half_width_to_full_width(const std::wstring& input) {
        std::wstring result;
        for (wchar_t ch : input) {
            if (H2F_ASCII_LETTERS.find(ch) != H2F_ASCII_LETTERS.end()) {
                result += H2F_ASCII_LETTERS[ch];
            }
            else if (H2F_DIGITS.find(ch) != H2F_DIGITS.end()) {
                result += H2F_DIGITS[ch];
            }
            else if (H2F_PUNCTUATIONS.find(ch) != H2F_PUNCTUATIONS.end()) {
                result += H2F_PUNCTUATIONS[ch];
            }
            else if (H2F_SPACE.find(ch) != H2F_SPACE.end()) {
                result += H2F_SPACE[ch];
            }
            else {
                result += ch; // 如果没有匹配，保持原字符
            }
        }
        return result;
    }

    // 正则表达式匹配非拼音的汉字字符串（根据支持 UCS4 的不同情况）
    std::wregex RE_NSW(L"[^\\u3007\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff]+");
}
