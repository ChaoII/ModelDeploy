//
// Created by aichao on 2025/5/21.
//

#pragma once

#include <unordered_map>

namespace modeldeploy::audio {
    // 初始化全角 -> 半角 映射表
    extern std::unordered_map<wchar_t, wchar_t> F2H_ASCII_LETTERS;
    extern std::unordered_map<wchar_t, wchar_t> H2F_ASCII_LETTERS;
    extern std::unordered_map<wchar_t, wchar_t> F2H_DIGITS;
    extern std::unordered_map<wchar_t, wchar_t> H2F_DIGITS;
    extern std::unordered_map<wchar_t, wchar_t> F2H_PUNCTUATIONS;
    extern std::unordered_map<wchar_t, wchar_t> H2F_PUNCTUATIONS;
    extern std::unordered_map<wchar_t, wchar_t> F2H_SPACE;
    extern std::unordered_map<wchar_t, wchar_t> H2F_SPACE;

    void initialize_constant_maps();
    std::wstring full_width_to_half_width(const std::wstring& input);
    std::wstring half_width_to_full_width(const std::wstring& input);
}
