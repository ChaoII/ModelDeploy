//
// Created by aichao on 2025/2/20.
//
#include <cstdint>
#include "csrc/core/md_type.h"

#include "md_log.h"

namespace modeldeploy {
    int32_t MDDataType::size(const Type& data_type) {
        if (data_type == Type::BOOL) {
            return sizeof(bool);
        }
        if (data_type == MDDataType::INT16) {
            return sizeof(int16_t);
        }
        if (data_type == MDDataType::INT32) {
            return sizeof(int32_t);
        }
        if (data_type == MDDataType::INT64) {
            return sizeof(int64_t);
        }
        if (data_type == MDDataType::FP32) {
            return sizeof(float);
        }
        if (data_type == MDDataType::FP64) {
            return sizeof(double);
        }
        if (data_type == MDDataType::UINT8) {
            return sizeof(uint8_t);
        }
        if (data_type == MDDataType::INT8) {
            return sizeof(int8_t);
        }
        MD_LOG_ERROR("Unexpected data type: {}", str(data_type));
        return -1;
    }

    std::string MDDataType::str(const Type& data_type) {
        std::string out;
        switch (data_type) {
        case MDDataType::BOOL:
            out = "MDDataType::BOOL";
            break;
        case MDDataType::INT16:
            out = "MDDataType::INT16";
            break;
        case MDDataType::INT32:
            out = "MDDataType::INT32";
            break;
        case MDDataType::INT64:
            out = "MDDataType::INT64";
            break;
        case MDDataType::FP32:
            out = "MDDataType::FP32";
            break;
        case MDDataType::FP64:
            out = "MDDataType::FP64";
            break;
        case MDDataType::UINT8:
            out = "MDDataType::UINT8";
            break;
        case MDDataType::INT8:
            out = "MDDataType::INT8";
            break;
        default:
            out = "MDDataType::UNKNOWN";
        }
        return out;
    }

    std::ostream& operator<<(std::ostream& out, const MDDataType::Type& md_dtype) {
        switch (md_dtype) {
        case MDDataType::BOOL:
            out << "MDDataType::BOOL";
            break;
        case MDDataType::INT16:
            out << "MDDataType::INT16";
            break;
        case MDDataType::INT32:
            out << "MDDataType::INT32";
            break;
        case MDDataType::INT64:
            out << "MDDataType::INT64";
            break;
        case MDDataType::FP32:
            out << "MDDataType::FP32";
            break;
        case MDDataType::FP64:
            out << "MDDataType::FP64";
            break;
        case MDDataType::UINT8:
            out << "MDDataType::UINT8";
            break;
        case MDDataType::INT8:
            out << "MDDataType::INT8";
            break;
        default:
            out << "MDDataType::UNKNOWN";
        }
        return out;
    }

    template <typename PlainType>
    const MDDataType::Type TypeToDataType<PlainType>::dtype = MDDataType::Type::UNKNOWN1;

    template <>
    const MDDataType::Type TypeToDataType<bool>::dtype = MDDataType::Type::BOOL;

    template <>
    const MDDataType::Type TypeToDataType<int16_t>::dtype = MDDataType::Type::INT16;

    template <>
    const MDDataType::Type TypeToDataType<int32_t>::dtype = MDDataType::Type::INT32;

    template <>
    const MDDataType::Type TypeToDataType<int64_t>::dtype = MDDataType::Type::INT64;

    template <>
    const MDDataType::Type TypeToDataType<float>::dtype = MDDataType::Type::FP32;

    template <>
    const MDDataType::Type TypeToDataType<double>::dtype = MDDataType::Type::FP64;

    template <>
    const MDDataType::Type TypeToDataType<uint8_t>::dtype = MDDataType::Type::UINT8;

    template <>
    const MDDataType::Type TypeToDataType<int8_t>::dtype = MDDataType::Type::INT8;
}
