//
// Created by aichao on 2025/2/20.
//

#pragma once
#include <filesystem>
#include <vector>
#include <iterator>
#include "csrc/core/md_log.h"

namespace modeldeploy {
    ///
    /// 从文件中读取二进制内容
    /// 用于将文件中的二进制数据读取到字符串中，通常用于数据加载
    ///
    /// @param path 文件路径，指定要读取的文件
    /// @param contents 指向字符串的指针，用于存储读取的文件内容
    /// @return bool 表示是否成功读取文件内容
    bool read_binary_from_file(const std::string& path, std::string* contents);


    std::vector<std::string> string_split(const std::string& s, const std::string& delimiter);


#ifdef _WIN32
    std::wstring to_wstring(const std::string& str);
#endif

    std::wstring utf8_to_wstring(const std::string& str);

    std::string wstring_to_string(const std::wstring& ws);

    template <typename T>
    std::vector<T> remove_consecutive_duplicates(const std::vector<T>& input) {
        std::vector<T> result;
        if (input.empty()) return result;
        result.push_back(input[0]);
        for (size_t i = 1; i < input.size(); ++i) {
            if (input[i] != result.back()) {
                result.push_back(input[i]);
            }
        }
        return result;
    }


    std::vector<int64_t> get_stride(const std::vector<int64_t>& dims);

    static int canonical_axis(const int axis, const int rank) {
        if (axis < 0) {
            return axis + rank;
        }
        return axis;
    }

    static int size_to_axis(const int axis, const std::vector<int64_t>& dims) {
        int size = 1;
        for (int i = 0; i < axis; i++) {
            size *= static_cast<int>(dims[i]);
        }
        return size;
    }

    static int size_from_axis(const int axis,
                              const std::vector<int64_t>& dims) {
        int size = 1;
        for (int i = axis; i < dims.size(); i++) {
            size *= static_cast<int>(dims[i]);
        }
        return size;
    }

    static int size_out_axis(const int axis,
                             const std::vector<int64_t>& dims) {
        int size = 1;
        for (int i = axis + 1; i < dims.size(); i++) {
            size *= static_cast<int>(dims[i]);
        }
        return size;
    }

    template <typename T>
    static std::string vector_to_string(const std::vector<T>& v) {
        std::ostringstream oss;
        if (v.empty()) {
            return "[]";
        }
        oss << "[";
        std::copy(v.begin(), v.end() - 1, std::ostream_iterator<T>(oss, ", "));
        oss << v.back() << "]";
        return oss.str();
    }

    std::vector<unsigned char> base64_decode(const std::string& base64_str);

    int argmax(const std::vector<float>& vec);

    template <typename T>
    void calculate_statis_info(const void* src_ptr, int size, double* mean,
                               double* max, double* min) {
        const T* ptr = static_cast<const T*>(src_ptr);
        *mean = static_cast<double>(0);
        *max = static_cast<double>(-99999999);
        *min = static_cast<double>(99999999);
        for (int i = 0; i < size; ++i) {
            if (*(ptr + i) > *max) {
                *max = *(ptr + i);
            }
            if (*(ptr + i) < *min) {
                *min = *(ptr + i);
            }
            *mean += *(ptr + i);
        }
        *mean = *mean / size;
    }
}
