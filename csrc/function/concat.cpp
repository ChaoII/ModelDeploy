//
// Created by aichao on 2025/2/20.
//
#include "concat.h"

#include "../utils/utils.h"
#include <cstring>
#include <limits>
#include <set>
#include <sstream>

namespace modeldeploy::function  {

std::vector<int64_t>
ComputeAndCheckConcatOutputShape(const std::vector<MDTensor>& input, int axis) {
  const size_t n = input.size();
  auto out_dims = input[0].shape;
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; ++i) {
    if(input[i].shape.size() != out_dims.size()){
             std::cerr<<"The shape of input[0] and input["<<i<<"] is expected to be equal. But "
             "received input[0]'s shape = "<<print_vector(out_dims)<<
          ", input["<<i<<"]'s shape = "<<print_vector(input[i].shape)<<"."<<std::endl;
             }
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis) {
        out_dims[axis] += input[i].shape[axis];
      } else {
        if(input[0].shape[j] != input[i].shape[j]){
            std::cerr<<"The "<<j<<"-th dimension of input[0] and input["<<i<<"] is expected to be "
            "equal."
            "But received input[0]'s shape = "<<print_vector(input[0].shape)<<", input["<<i<<"]'s shape = "<<print_vector(input[1].shape)<<std::endl;
            }
      }
    }
  }
  return out_dims;
}

template <typename T> struct ConcatFunctor {
  void operator()(const std::vector<MDTensor>& input, int axis,
                  MDTensor* output) {
    size_t num = input.size();

    int64_t rows = 1;
    auto dim_0 = input[0].shape;
    for (int i = 0; i < axis; ++i) {
      rows *= dim_0[i];
    }
    int64_t out_rows = rows, out_cols = 0;

    std::vector<int64_t> input_cols(num);
    for (size_t i = 0; i < num; ++i) {
      int64_t t_cols = input[i].total() / rows;
      out_cols += t_cols;
      input_cols[i] = t_cols;
    }

    // computation
    T* output_data = reinterpret_cast<T*>(output->data());
    int64_t col_idx = 0;
    for (size_t j = 0; j < num; ++j) {
      int64_t col_len = input_cols[j];
      const T* input_data = reinterpret_cast<const T*>(input[j].data());
      for (int64_t k = 0; k < out_rows; ++k) {
        MDTensor::copy_buffer(output_data + k * out_cols + col_idx,
                             input_data + k * col_len, sizeof(T) * col_len);
      }
      col_idx += col_len;
    }
  }
};

template <typename T>
void ConcatKernel(const std::vector<MDTensor>& input, MDTensor* output,
                  int axis) {
  auto output_shape = ComputeAndCheckConcatOutputShape(input, axis);
  MDTensor output_tmp;
  output_tmp.resize(output_shape, TypeToDataType<T>::dtype, output->name);

  ConcatFunctor<T> functor;
  functor(input, axis, &output_tmp);
  *output = std::move(output_tmp);
}

void Concat(const std::vector<MDTensor>& x, MDTensor* out, int axis) {
  if(x.size() < 0){
           std::cerr<<"The number of FDTensor array should be larger than 0, but the size "
           "of input is " << x.size()<<std::endl;
           }
  int64_t rank = x[0].shape.size();
  if(!(axis >= -rank && axis < rank)){
           std::cerr<<"The axis is expected to be in range of ["<<rank <<", "<<axis<<"), but got "<<-rank<<std::endl;
           }
  if (axis < 0) {
    axis += rank;
  }

  MD_VISIT_ALL_TYPES(x[0].dtype, "Concat",
                     ([&] { ConcatKernel<data_t>(x, out, axis); }));
}

}

