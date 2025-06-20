//
// Created by aichao on 2025/2/19.
//

#pragma once
#include <string>
#include <MNN/Interpreter.hpp>
#include "csrc/core/tensor.h"

namespace modeldeploy {
    halide_type_t md_dtype_to_mnn_dtype(const DataType& dtype);

    // Convert OrtDataType to FDDataType
    DataType mnn_dtype_to_md_dtype(const halide_type_t& mnn_dtype);

    std::string mnn_type_to_string(const halide_type_t& type);

    template <class SrcType, class DstType>
    std::vector<DstType> convert_shape(std::vector<SrcType> shape) {
        std::vector<DstType> out_shape(shape.size());
        std::transform(
            shape.begin(), shape.end(), out_shape.begin(),
            [](const SrcType& value) { return static_cast<DstType>(value); });
        return out_shape;
    }
}
