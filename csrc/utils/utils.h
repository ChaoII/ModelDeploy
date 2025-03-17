//
// Created by aichao on 2025/2/20.
//

#pragma once
#include <vector>
#include "csrc/core/md_log.h"

namespace modeldeploy {
    bool read_binary_from_file(const std::string& path, std::string* contents);

#ifdef _WIN32
    std::wstring to_wstring(const std::string& str);
#endif


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
    static std::string print_vector(std::vector<T> v) {
        std::string res;
        for (auto i : v) {
            res += std::to_string(i) + " ";
        }
        return res;
    }

#define MD_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...)                   \
    case enum_type: {                                                                       \
        using HINT = type;                                                                  \
        __VA_ARGS__();                                                                      \
        break;                                                                              \
       }

#define MD_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...)                                    \
    MD_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, data_t, __VA_ARGS__)

#define MD_VISIT_ALL_TYPES(TYPE, NAME, ...)                                                 \
    [&] {                                                                                   \
        const auto& __dtype__ = TYPE;                                                       \
        switch (__dtype__) {                                                                \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::Type::UINT8, uint8_t,                    \
                                    __VA_ARGS__)                                            \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::Type::BOOL, bool,                        \
                                    __VA_ARGS__)                                            \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::Type::INT32, int32_t,                    \
                                    __VA_ARGS__)                                            \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::Type::INT64, int64_t,                    \
                                    __VA_ARGS__)                                            \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::Type::FP32, float,                       \
                                    __VA_ARGS__)                                            \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::Type::FP64, double,                      \
                                    __VA_ARGS__)                                            \
            default:                                                                        \
                std::cerr<<"Invalid enum data type. Expect to accept data "                 \
                "type BOOL, INT32, INT64, FP32, FP64,"                                      \
                "but receive type "<<MDDataType::str(__dtype__)<<std::endl;                 \
        }                                                                                   \
    }()

#define MD_VISIT_FLOAT_TYPES(TYPE, NAME, ...)                                               \
    [&] {                                                                                   \
        const auto& __dtype__ = TYPE;                                                       \
        switch (__dtype__) {                                                                \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::Type::FP32, float, __VA_ARGS__)          \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::Type::FP64, double, __VA_ARGS__)         \
            default:                                                                        \
                std::cerr<<"Invalid enum data type. Expect to accept data type FP32, "      \
                "FP64, but receive type "<<MDDataType::str(__dtype__)<<std::endl;           \
        }                                                                                   \
    }()
}
