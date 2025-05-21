//
// Created by aichao on 2025/5/21.
//

#pragma once

#include <filesystem>
#include <unordered_map>

namespace modeldeploy::audio {
    extern std::unordered_map<wchar_t, wchar_t> s2t_dict;
    extern std::unordered_map<wchar_t, wchar_t> t2s_dict;
    void initialize_char_maps(const std::filesystem::path& char_map_folder);
    // 从文件中读取字符串
    std::wstring readFile(const std::string& filename);
    // 将繁体转换为简体
    std::wstring traditional_to_simplified(const std::wstring& text);
    // 将简体转换为繁体
    std::wstring simplified_to_traditional(const std::wstring& text);
    // 保存映射到二进制文件
    void save_map_to_binary_file(const std::unordered_map<wchar_t, wchar_t>& map, const std::string& filename);
    // 从二进制文件加载映射
    std::unordered_map<wchar_t, wchar_t> load_map_from_binary_file(const std::string& filename);
    std::string wstring_to_string(const std::wstring& wstr);
    std::wstring string_to_wstring(const std::string& str);
}
