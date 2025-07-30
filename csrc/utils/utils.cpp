//
// Created by aichao on 2025/2/20.
//

#include "utils/utils.h"

#include <codecvt>
#include <fstream>
#include <regex>

namespace modeldeploy {
    std::vector<int64_t> get_stride(const std::vector<int64_t>& dims) {
        const auto dims_size = dims.size();
        std::vector<int64_t> result(dims_size, 1);
        for (int i = static_cast<int>(dims_size) - 2; i >= 0; --i) {
            result[i] = result[i + 1] * dims[i + 1];
        }
        return result;
    }

#ifdef _WIN32
#include <Windows.h>
    using os_string = std::wstring;
#else
    using os_string = std::string;
#endif
    os_string to_os_string(std::string_view utf8_str) {
#ifdef _WIN32
        const int len = MultiByteToWideChar(CP_UTF8, 0, utf8_str.data(),
                                            static_cast<int>(utf8_str.size()), nullptr, 0);
        os_string result(len, 0);
        MultiByteToWideChar(CP_UTF8, 0, utf8_str.data(),
                            static_cast<int>(utf8_str.size()), result.data(), len);
        return result;
#else
        return std::string(utf8_str);
#endif
    }

    bool read_binary_from_file(const std::string& path, std::string* contents) {
        if (!contents) {
            return false;
        }
        auto& result = *contents;
        result.clear();

        std::ifstream file(to_os_string(path), std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return false;
        }
        if (const auto file_size = file.tellg(); file_size != -1) {
            result.resize(file_size);
            file.seekg(0, std::ios::beg);
            file.read(result.data(), file_size);
        }
        else {
            // no size available, read to EOF
            constexpr auto chunk_size = 4096;
            std::string chunk(chunk_size, 0);
            while (!file.fail()) {
                file.read(chunk.data(), chunk_size);
                result.insert(result.end(), chunk.data(), chunk.data() + file.gcount());
            }
        }
        return true;
    }
#ifdef _WIN32
    std::wstring to_wstring(const std::string& str) {
        const unsigned len = str.size() + 1; // +1 for the null terminator
        std::wstring w_str(len, 0);
        size_t converted_chars = 0;
        if (const errno_t err = mbstowcs_s(&converted_chars, w_str.data(), len, str.c_str(), _TRUNCATE); err != 0) {
            // Handle error
            return L"";
        }
        w_str.resize(converted_chars - 1); // Remove the null terminator
        return w_str;
    }
#endif


    std::wstring utf8_to_wstring(const std::string& str) {
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
        return conv.from_bytes(str);
    }

    std::string wstring_to_string(const std::wstring& ws) {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
        return conv.to_bytes(ws);
    }


    std::unordered_map<int, std::string> parse_label_map(const std::string& label_string) {
        std::unordered_map<int, std::string> result;
        // 匹配形如 0: 'head' 的片段
        std::regex pattern(R"((\d+):\s*'([^']+)')");
        auto begin = std::sregex_iterator(label_string.begin(), label_string.end(), pattern);
        auto end = std::sregex_iterator();
        for (std::sregex_iterator i = begin; i != end; ++i) {
            int key = std::stoi((*i)[1]);
            std::string value = (*i)[2];
            result[key] = value;
        }
        return result;
    }

    std::vector<std::string> string_split(const std::string& s, const std::string& delimiter) {
        std::vector<std::string> tokens;
        size_t start = 0, end = s.find(delimiter);
        while (end != std::string::npos) {
            tokens.push_back(s.substr(start, end - start));
            start = end + delimiter.length();
            end = s.find(delimiter, start);
        }
        tokens.push_back(s.substr(start)); // 添加最后一个子串
        return tokens;
    }


    std::vector<unsigned char> base64_decode(const std::string& base64_str) {
        const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::vector<unsigned char> ret;
        const int in_len = static_cast<int>(base64_str.size());
        unsigned char char_array_4[4], char_array_3[3];
        int i = 0;
        for (int in_ = 0; in_ < in_len; ++in_) {
            if (base64_str[in_] == '=' || base64_chars.find(base64_str[in_]) == std::string::npos) {
                break;
            }
            char_array_4[i++] = base64_chars.find(base64_str[in_]);
            if (i == 4) {
                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
                ret.insert(ret.end(), char_array_3, char_array_3 + 3);
                i = 0;
            }
        }

        if (i) {
            for (int j = i; j < 4; ++j) {
                char_array_4[j] = 0;
            }
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
            ret.insert(ret.end(), char_array_3, char_array_3 + i - 1);
        }
        return ret;
    }

    // Base64实现
    std::string base64_encode(const std::vector<unsigned char>& data) {
        const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string ret;
        int in_len = static_cast<int>(data.size());
        int i = 0;
        int j = 0;
        unsigned char char_array_3[3];
        unsigned char char_array_4[4];
        while (in_len--) {
            char_array_3[i++] = data[j++];
            if (i == 3) {
                char_array_4[0] = (char_array_3[0] << 2) + ((char_array_3[1] & 0x30) >> 4);
                char_array_4[1] = ((char_array_3[1] & 0xf) << 4) + ((char_array_3[2] & 0x3c) >> 2);
                char_array_4[2] = ((char_array_3[2] & 0x3) << 6) + char_array_4[3];

                for (i = 0; i < 4; i++) {
                    ret += base64_chars[char_array_4[i]];
                }
                i = 0;
            }
        }

        if (i) {
            for (j = i; j < 3; j++) {
                char_array_3[j] = 0;
            }
            char_array_4[0] = (char_array_3[0] << 2) + ((char_array_3[1] & 0x30) >> 4);
            char_array_4[1] = ((char_array_3[1] & 0xf) << 4) + ((char_array_3[2] & 0x3c) >> 2);
            char_array_4[2] = ((char_array_3[2] & 0x3) << 6) + char_array_4[3];

            for (j = 0; j < i + 1; j++) {
                ret += base64_chars[char_array_4[j]];
            }

            while ((i++ < 3)) {
                ret += '=';
            }
        }

        return ret;
    }

    std::string base64_encode(const std::string& data) {
        std::vector<unsigned char> vec(data.begin(), data.end());
        return base64_encode(vec);
    }


    int argmax(const std::vector<float>& vec) {
        if (vec.empty()) {
            MD_LOG_ERROR << "Vector is empty." << std::endl;
            return -1; // 或者抛出异常
        }

        int max_index = 0;
        float max_value = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < vec.size(); ++i) {
            if (vec[i] > max_value) {
                max_value = vec[i];
                max_index = i;
            }
        }
        return max_index;
    }
}
