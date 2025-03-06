
//
// Created by aichao on 2025/2/20.
//
#pragma once

#include "../core/md_tensor.h"

namespace modeldeploy::function {
    /** Excute the transpose operation for input FDTensor along given dims.
        @param x The input tensor.
        @param out The output tensor which stores the result.
        @param dims The vector of axis which the input tensor will transpose.
    */
    void Transpose(const MDTensor& x, MDTensor* out,const std::vector<int64_t>& dims);
}
