//
// Created by aichao on 2025/2/20.
//
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include "md_type.h"

namespace modeldeploy {
    class Scalar {
    public:
        // Constructor support implicit
        Scalar() : Scalar(0) {
        }

        Scalar(double val) : dtype_(MDDataType::FP64) {
            // NOLINT
            data_.f64 = val;
        }

        Scalar(float val) : dtype_(MDDataType::FP32) {
            // NOLINT
            data_.f32 = val;
        }


        Scalar(int64_t val) : dtype_(MDDataType::INT64) {
            // NOLINT
            data_.i64 = val;
        }

        Scalar(int32_t val) : dtype_(MDDataType::INT32) {
            // NOLINT
            data_.i32 = val;
        }

        Scalar(int16_t val) : dtype_(MDDataType::INT16) {
            // NOLINT
            data_.i16 = val;
        }

        Scalar(int8_t val) : dtype_(MDDataType::INT8) {
            // NOLINT
            data_.i8 = val;
        }

        Scalar(uint8_t val) : dtype_(MDDataType::UINT8) {
            // NOLINT
            data_.ui8 = val;
        }

        Scalar(bool val) : dtype_(MDDataType::BOOL) {
            // NOLINT
            data_.b = val;
        }


        template <typename RT>
        inline RT to() const {
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

        MDDataType dtype() const { return dtype_; }

    private:
        MDDataType dtype_;

        union data {
            bool b;
            int8_t i8;
            int16_t i16;
            int32_t i32;
            int64_t i64;
            uint8_t ui8;
            float f32;
            double f64;
        } data_;
    };
}
