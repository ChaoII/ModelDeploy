//
// Created by aichao on 2025/2/20.
//

#pragma once
#include <iostream>
#include <vector>

namespace modeldeploy {
    bool read_binary_from_file(const std::string& path, std::string* contents);

#ifdef _WIN32
    std::wstring to_wstring(const std::string& str);
#endif


    std::vector<int64_t> get_stride(const std::vector<int64_t>& dims);

    static inline int canonical_axis(const int axis, const int rank) {
        if (axis < 0) {
            return axis + rank;
        }
        return axis;
    }

    static inline int size_to_axis(const int axis, const std::vector<int64_t>& dims) {
        int size = 1;
        for (int i = 0; i < axis; i++) {
            size *= dims[i];
        }
        return size;
    }

    static inline int size_from_axis(const int axis,
                                     const std::vector<int64_t>& dims) {
        int size = 1;
        for (int i = axis; i < dims.size(); i++) {
            size *= dims[i];
        }
        return size;
    }

    static inline int size_out_axis(const int axis,
                                    const std::vector<int64_t>& dims) {
        int size = 1;
        for (int i = axis + 1; i < dims.size(); i++) {
            size *= dims[i];
        }
        return size;
    }

#define MD_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...)      \
    case enum_type: {                                                            \
        using HINT = type;                                                         \
        __VA_ARGS__();                                                             \
        break;                                                                     \
       }

#define MD_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...)                       \
    MD_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, data_t, __VA_ARGS__)

#define MD_VISIT_ALL_TYPES(TYPE, NAME, ...)                                    \
    [&] {                                                                        \
        const auto& __dtype__ = TYPE;                                              \
        switch (__dtype__) {                                                       \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::UINT8, uint8_t,     \
                                    __VA_ARGS__)                                        \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::BOOL, bool,         \
                                    __VA_ARGS__)                                        \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::INT32, int32_t,     \
                                    __VA_ARGS__)                                        \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::INT64, int64_t,     \
                                    __VA_ARGS__)                                        \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::FP32, float,        \
                                    __VA_ARGS__)                                        \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::FP64, double,       \
                                    __VA_ARGS__)                                        \
            default:                                                                   \
                std::cerr<<"Invalid enum data type. Expect to accept data "                \
                "type BOOL, INT32, INT64, FP32, FP64, but receive type "<<std::endl;               \
        }                                                                          \
    }()

#define MD_VISIT_FLOAT_TYPES(TYPE, NAME, ...)                                  \
    [&] {                                                                        \
        const auto& __dtype__ = TYPE;                                              \
        switch (__dtype__) {                                                       \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::FP32, float,        \
                                    __VA_ARGS__)                                        \
            MD_PRIVATE_CASE_TYPE(NAME, MDDataType::FP64, double,       \
                                    __VA_ARGS__)                                        \
            default:                                                                   \
                std::cerr<< "Invalid enum data type. Expect to accept data type FP32, "     \
                "FP64, but receive type" <<std::endl; \
            }                                                                          \
    }()
}
