//
// Created by aichao on 2025/2/20.
//

#include "utils.h"

#include <fstream>

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
        const unsigned len = str.size() * 2;
        setlocale(LC_CTYPE, "");
        auto* p = new wchar_t[len];
        mbstowcs(p, str.c_str(), len);
        std::wstring w_str(p);
        delete[] p;
        return w_str;
    }
#endif

    std::vector<unsigned char> base64_decode(const std::string& base64_str) {
        const std::string base64_chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";

        int in_len = base64_str.size();
        int i = 0;
        int j = 0;
        int in_ = 0;
        unsigned char char_array_4[4], char_array_3[3];
        std::vector<unsigned char> ret;

        while (in_len-- && (base64_str[in_] != '=') && base64_chars.find(base64_str[in_]) != std::string::npos) {
            char_array_4[i++] = base64_str[in_];
            in_++;
            if (i == 4) {
                for (i = 0; i < 4; i++)
                    char_array_4[i] = base64_chars.find(char_array_4[i]);
                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
                for (i = 0; (i < 3); i++)
                    ret.push_back(char_array_3[i]);
                i = 0;
            }
        }
        if (i) {
            for (j = i; j < 4; j++)
                char_array_4[j] = 0;
            for (j = 0; j < 4; j++)
                char_array_4[j] = base64_chars.find(char_array_4[j]);
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
            for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
        }
        return ret;
    }
}
