//
// Created by aichao on 2025/2/20.
//
#pragma once

#include <cstdint>
#include "csrc/core/md_type.h"

namespace modeldeploy {
    class Scalar {
    public:
        // Constructor support implicit
        Scalar() : Scalar(0) {
        }

        explicit Scalar(const double val) : dtype_(MDDataType::Type::FP64) {
            data_.f64 = val;
        }

        explicit Scalar(const float val) : dtype_(MDDataType::Type::FP32) {
            data_.f32 = val;
        }


        explicit Scalar(const int64_t val) : dtype_(MDDataType::Type::INT64) {
            data_.i64 = val;
        }

        explicit Scalar(const int32_t val) : dtype_(MDDataType::Type::INT32) {
            data_.i32 = val;
        }

        explicit Scalar(const int16_t val) : dtype_(MDDataType::Type::INT16) {
            data_.i16 = val;
        }

        explicit Scalar(const int8_t val) : dtype_(MDDataType::Type::INT8) {
            data_.i8 = val;
        }

        explicit Scalar(const uint8_t val) : dtype_(MDDataType::Type::UINT8) {
            data_.ui8 = val;
        }

        explicit Scalar(const bool val) : dtype_(MDDataType::Type::BOOL) {
            data_.b = val;
        }


        template <typename RT>
        RT to() const {
            switch (dtype_) {
            case MDDataType::FP32:
                return static_cast<RT>(data_.f32);
            case MDDataType::FP64:
                return static_cast<RT>(data_.f64);
            case MDDataType::INT32:
                return static_cast<RT>(data_.i32);
            case MDDataType::INT64:
                return static_cast<RT>(data_.i64);
            case MDDataType::INT16:
                return static_cast<RT>(data_.i16);
            case MDDataType::INT8:
                return static_cast<RT>(data_.i8);
            case MDDataType::UINT8:
                return static_cast<RT>(data_.ui8);
            case MDDataType::BOOL:
                return static_cast<RT>(data_.b);
            default:
                return static_cast<RT>(data_.f32);
            }
        }

        [[nodiscard]] MDDataType::Type dtype() const { return dtype_; }

    private:
        MDDataType::Type dtype_;

        union data {
            bool b;
            int8_t i8;
            int16_t i16;
            int32_t i32;
            int64_t i64;
            uint8_t ui8;
            float f32;
            double f64;
        } data_{};
    };
}
