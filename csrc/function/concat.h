//
// Created by aichao on 2025/2/20.
//
#pragma once

#include "csrc/core/md_tensor.h"

namespace modeldeploy::function  {


/** Excute the concatenate operation for input FDTensor along given axis.
    @param x The input tensor.
    @param out The output tensor which stores the result.
    @param axis Axis which will be concatenated.
*/
    void Concat(const std::vector<MDTensor>& x, MDTensor* out,
                            int axis = 0);

}

