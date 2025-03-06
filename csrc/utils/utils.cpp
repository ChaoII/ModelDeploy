//
// Created by aichao on 2025/2/20.
//

#include "utils.h"

#include <fstream>

namespace modeldeploy {
    std::vector<int64_t> get_stride(const std::vector<int64_t>& dims) {
        auto dims_size = dims.size();
        std::vector<int64_t> result(dims_size, 1);
        for (int i = dims_size - 2; i >= 0; --i) {
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
    os_string to_osstring(std::string_view utf8_str) {
#ifdef _WIN32
        int len = MultiByteToWideChar(CP_UTF8, 0, utf8_str.data(), static_cast<int>(utf8_str.size()), nullptr, 0);
        os_string result(len, 0);
        MultiByteToWideChar(CP_UTF8, 0, utf8_str.data(), static_cast<int>(utf8_str.size()), result.data(), len);
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

        std::ifstream file(to_osstring(path), std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return false;
        }
        auto file_size = file.tellg();
        if (file_size != -1) {
            result.resize(file_size);
            file.seekg(0, std::ios::beg);
            file.read(const_cast<char*>(result.data()), file_size);
        }
        else {
            // no size available, read to EOF
            constexpr auto chunk_size = 4096;
            std::string chunk(chunk_size, 0);
            while (!file.fail()) {
                file.read(const_cast<char*>(chunk.data()), chunk_size);
                result.insert(result.end(), chunk.data(), chunk.data() + file.gcount());
            }
        }
        return true;
    }
#ifdef _WIN32
    std::wstring to_wstring(const std::string& str) {
        unsigned len = str.size() * 2;
        setlocale(LC_CTYPE, "");
        auto* p = new wchar_t[len];
        mbstowcs(p, str.c_str(), len);
        std::wstring wstr(p);
        delete[] p;
        return wstr;
    }
#endif
}
