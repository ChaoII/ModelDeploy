//
// Created by aichao on 2025/2/20.
//

#include "csrc/utils/utils.h"
#include "csrc/function/concat.h"

namespace modeldeploy::function {
    std::vector<int64_t>
    ComputeAndCheckConcatOutputShape(const std::vector<MDTensor>& input, const int axis) {
        const size_t n = input.size();
        auto out_dims = input[0].shape;
        const size_t in_zero_dims_size = out_dims.size();
        for (size_t i = 1; i < n; ++i) {
            if (input[i].shape.size() != out_dims.size()) {
                MD_LOG_ERROR(
                    "The shape of input[0] and input[{}] is expected to be equal. But "
                    "received input[0]'s shape = {}, input[{}]'s shape = {} .",
                    i, print_vector(out_dims), i, print_vector(input[i].shape));
            }
            for (size_t j = 0; j < in_zero_dims_size; j++) {
                if (j == axis) {
                    out_dims[axis] += input[i].shape[axis];
                }
                else {
                    if (input[0].shape[j] != input[i].shape[j]) {
                        MD_LOG_ERROR(
                            "The {}-th dimension of input[0] and input[{}] is expected to be "
                            "equal.But received input[0]'s shape = {}, input[{}]'s shape = { }",
                            j, i, print_vector(input[0].shape), print_vector(input[1].shape));
                    }
                }
            }
        }
        return out_dims;
    }

    template <typename T>
    struct ConcatFunctor {
        void operator()(const std::vector<MDTensor>& input, const int axis,
                        MDTensor* output) const {
            const size_t num = input.size();

            int64_t rows = 1;
            const auto dim_0 = input[0].shape;
            for (int i = 0; i < axis; ++i) {
                rows *= dim_0[i];
            }
            const int64_t out_rows = rows;
            int64_t out_cols = 0;

            std::vector<int64_t> input_cols(num);
            for (size_t i = 0; i < num; ++i) {
                const int64_t t_cols = input[i].total() / rows;
                out_cols += t_cols;
                input_cols[i] = t_cols;
            }

            // computation
            T* output_data = static_cast<T*>(output->data());
            int64_t col_idx = 0;
            for (size_t j = 0; j < num; ++j) {
                const int64_t col_len = input_cols[j];
                const T* input_data = static_cast<const T*>(input[j].data());
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
        const auto output_shape = ComputeAndCheckConcatOutputShape(input, axis);
        MDTensor output_tmp;
        output_tmp.resize(output_shape, TypeToDataType<T>::dtype, output->name);
        ConcatFunctor<T> functor;
        functor(input, axis, &output_tmp);
        *output = std::move(output_tmp);
    }

    void Concat(const std::vector<MDTensor>& x, MDTensor* out, int axis) {
        if (x.size() < 0) {
            MD_LOG_ERROR(
                "The number of MDTensor array should be larger than 0, but the size of input is {}", x.size());
        }
        const auto rank = static_cast<int>(x[0].shape.size());
        if (!(axis >= -rank && axis < rank)) {
            MD_LOG_ERROR("The axis is expected to be in range of [ {} , {} ], but got {}", rank, axis, -rank);
        }
        if (axis < 0) {
            axis += rank;
        }
        MD_VISIT_ALL_TYPES(x[0].dtype, "Concat",
                           ([&] { ConcatKernel<data_t>(x, out, axis); }));
    }
}
