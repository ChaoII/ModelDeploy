//
// Created by aichao on 2025/5/21.
//

#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#define NOGDI
#define NOCRYPT
#include <windows.h>
#endif
#include "csrc/audio/text_normalize/char_convert.h"
#include "csrc/core/md_log.h"

namespace modeldeploy::audio {
    std::unordered_map<wchar_t, wchar_t> s2t_dict;
    std::unordered_map<wchar_t, wchar_t> t2s_dict;
    // 从文件中读取字符串
    std::wstring readFile(const std::string& filename) {
        std::wifstream file(filename);
        if (!file.is_open()) {
            MD_LOG_ERROR << "text_normalization::readFile:: Cannot openfile:  " << filename << std::endl;
            return L"";
        }
        // 设置 locale 以处理 UTF-8 编码
        file.imbue(std::locale(std::locale(), new std::codecvt_utf8<wchar_t>));

        std::wstringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    // 保存映射到二进制文件
    void save_map_to_binary_file(const std::unordered_map<wchar_t, wchar_t>& map, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            MD_LOG_ERROR << "save_map_to_binary_file:: Cannot openfile: " << filename << std::endl;
            return;
        }

        const size_t size = map.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));

        for (const auto& pair : map) {
            file.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
            file.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
        }
        file.close();
    }

    // 从二进制文件加载映射
    std::unordered_map<wchar_t, wchar_t> load_map_from_binary_file(const std::string& filename) {
        std::unordered_map<wchar_t, wchar_t> map;
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            MD_LOG_ERROR << "load_map_from_binary_file:: Cannot openfile: " << filename << std::endl;
            return map;
        }
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        wchar_t key, value;
        for (size_t i = 0; i < size; ++i) {
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            map[key] = value;
        }
        file.close();
        return map;
    }

    std::string wstring_to_string(const std::wstring& wstr) {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        return converter.to_bytes(wstr);
    }

    std::wstring string_to_wstring(const std::string& str) {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        return converter.from_bytes(str);
    }

    // 将繁体转换为简体
    std::wstring traditional_to_simplified(const std::wstring& text) {
        std::wstring result;
        for (wchar_t ch : text) {
            if (t2s_dict.contains(ch)) {
                result += t2s_dict[ch];
            }
            else {
                result += ch; // 保持原字符
            }
        }
        return result;
    }

    // 将简体转换为繁体
    std::wstring simplified_to_traditional(const std::wstring& text) {
        std::wstring result;
        for (wchar_t ch : text) {
            if (s2t_dict.contains(ch)) {
                result += s2t_dict[ch];
            }
            else {
                result += ch; // 保持原字符
            }
        }
        return result;
    }

    void initialize_char_maps(const std::filesystem::path& char_map_folder) {
        const std::filesystem::path s2t_path = char_map_folder / "s2t_map.bin";
        const std::filesystem::path t2s_map = char_map_folder / "t2s_map.bin";
        // 从二进制文件加载映射
        s2t_dict = load_map_from_binary_file(s2t_path.string());
        t2s_dict = load_map_from_binary_file(t2s_map.string());
    }
}
