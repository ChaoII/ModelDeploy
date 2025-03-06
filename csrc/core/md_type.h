//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <iostream>
#include <string>
#include <vector>
namespace modeldeploy {
    enum MDDataType {
        BOOL,
        INT16,
        INT32,
        INT64,
        FP32,
        FP64,
        UNKNOWN1,
        UNKNOWN2,
        UNKNOWN3,
        UNKNOWN4,
        UNKNOWN5,
        UNKNOWN6,
        UNKNOWN7,
        UNKNOWN8,
        UNKNOWN9,
        UNKNOWN10,
        UNKNOWN11,
        UNKNOWN12,
        UNKNOWN13,
        UINT8,
        INT8
      };

    template <typename T>
    std::string print_vector(std::vector<T> v){
        std::string res;
        for (auto i : v) {
            res += std::to_string(i) + " ";
        }
        return res;
    }

    std::string str(const MDDataType& fdt);

    int32_t md_dtype_size(const MDDataType& data_dtype);

    template <typename PlainType>
    struct  TypeToDataType {
        static const MDDataType dtype;
    };
}