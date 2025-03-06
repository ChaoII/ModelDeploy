//
// Created by aichao on 2025/2/24.
//
#pragma once

#include "csrc/core/md_tensor.h"

namespace modeldeploy {
namespace function {
/** Excute the softmax operation for input FDTensor along given dims.
    @param x The input tensor.
    @param out The output tensor which stores the result.
    @param axis The axis to be computed softmax value.
*/
void softmax(const MDTensor& x, MDTensor* out, int axis = -1);

}  // namespace function
}  // namespace fastdeploy
