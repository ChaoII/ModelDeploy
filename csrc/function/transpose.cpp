//
// Created by aichao on 2025/2/20.
//


#include "csrc/utils/utils.h"
#include "csrc/function/eigen.h"
#include "csrc/function/transpose.h"

#include <csrc/core/md_log.h>

namespace modeldeploy::function {
    template <typename T>
    struct TransposeNormalKernel {
        void operator()(const MDTensor& in, MDTensor* out,
                        const std::vector<int64_t>& axis) const {
            const int rank = static_cast<int>(axis.size());
            const auto in_stride = get_stride(in.shape);
            const auto out_stride = get_stride(out->shape);
            auto transpose_helper = [&](const int64_t beg, const int64_t end) {
                const T* in_ptr = static_cast<const T*>(in.data());
                T* out_ptr = static_cast<T*>(out->data());
                for (int64_t out_idx = beg; out_idx < end; ++out_idx) {
                    int64_t in_idx = 0;
                    int64_t tmp_idx = out_idx;
                    // calculate the input index
                    for (int i = 0; i < rank; ++i) {
                        const int64_t coordinate = tmp_idx / out_stride[i];
                        tmp_idx -= coordinate * out_stride[i];
                        in_idx += coordinate * in_stride[axis[i]];
                    }
                    out_ptr[out_idx] = in_ptr[in_idx];
                }
            };
            transpose_helper(0, out->total());
        }
    };

    template <typename T, int Rank>
    struct TransposeKernelImpl {
        void operator()(const MDTensor& in, MDTensor* out,
                        const std::vector<int64_t>& axis) {
            Eigen::array<int, Rank> permute;
            for (int i = 0; i < Rank; i++) {
                permute[i] = axis[i];
            }

            auto& place = *EigenDeviceWrapper::GetInstance()->GetDevice();
            auto eigen_in = EigenTensor<T, Rank>::From(in);
            auto eigen_out = EigenTensor<T, Rank>::From(*out);
            eigen_out.device(place) = eigen_in.shuffle(permute);
        }
    };

    template <typename T>
    void TransposeKernel(const MDTensor& x, MDTensor* out, const std::vector<int64_t>& axis) {
        switch (axis.size()) {
        case 1:
            TransposeKernelImpl<T, 1> trans1;
            trans1(x, out, axis);
            break;
        case 2:
            TransposeKernelImpl<T, 2> trans2;
            trans2(x, out, axis);
            break;
        case 3:
            TransposeKernelImpl<T, 3> trans3;
            trans3(x, out, axis);
            break;
        case 4:
            TransposeKernelImpl<T, 4> trans4;
            trans4(x, out, axis);
            break;
        default:
            // for rank >= 4 situation
            TransposeNormalKernel<T> trans_normal;
            trans_normal(x, out, axis);
        }
    }

    void Transpose(const MDTensor& x, MDTensor* out, const std::vector<int64_t>& dims) {
        const size_t dims_size = dims.size();
        if (dims_size != x.shape.size()) {
            MD_LOG_ERROR << "The input tensor's dimension should be equal to the dims size. "
                "Expect dims size is " << x.shape.size() << ", but receive " << dims_size << "." << std::endl;
        }
        std::vector count(dims_size, 0);
        for (size_t i = 0; i < dims_size; i++) {
            if (dims[i] < 0) {
                MD_LOG_ERROR << "The dims should be greater than or equal to 0, but receive "
                    << dims[i] << "." << std::endl;
            }
            if (dims[i] >= static_cast<int>(dims_size) && ++count[dims[i]] == 1) {
                MD_LOG_ERROR << "Each element of Attribute axis should be a unique value range "
                    "from 0 to (dims - 1), where the dims is the axis's size, unique "
                    "value means this axis value can appear only once." << std::endl;
            }
        }
        std::vector<int64_t> out_dims(dims_size);
        for (size_t i = 0; i < dims_size; i++) {
            out_dims[i] = x.shape[dims[i]];
        }

        // notice: The MDTensor out may equal to MDTensor x, so firstly we
        // use out_temp to get the transposed result, then we move the out_temp to
        // out.
        MDTensor out_temp;
        out_temp.allocate(out_dims, x.dtype);
        MD_VISIT_ALL_TYPES(x.dtype, "TransposeKernel",
                           ([&] { TransposeKernel<data_t>(x, &out_temp, dims); }));
        *out = std::move(out_temp);
    }
} // namespace function
