//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <iostream>
#include <string>
#include "csrc/core/float16.h"

namespace modeldeploy {
    class MDDataType {
    public:
        enum Type {
            BOOL,
            INT16,
            INT32,
            INT64,
            FP16,
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


        static std::string str(const Type& data_type);

        static int32_t size(const Type& data_type);
    };

    std::ostream& operator<<(std::ostream& out, const MDDataType::Type& md_dtype);

    template <typename PlainType>
    struct TypeToDataType {
        static const MDDataType::Type dtype;
    };
}
