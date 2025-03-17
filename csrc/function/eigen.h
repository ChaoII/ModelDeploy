//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <algorithm>
#include <memory>
#include <vector>
#include "csrc/core/md_tensor.h"
#include "csrc/utils/utils.h"
#include "third_party/eigen/unsupported/Eigen/CXX11/Tensor"

namespace modeldeploy::function {
    // EigenDim converts shape into Eigen::DSizes.
    template <int D>
    struct EigenDim {
        using Type = Eigen::DSizes<Eigen::DenseIndex, D>;

        static Type From(const std::vector<int64_t>& dims) {
            Type ret;
            for (int64_t d = 0; d < dims.size(); d++) {
                ret[d] = dims[d];
            }
            return ret;
        }
    };

    // Interpret MDTensor as EigenTensor and EigenConstTensor.
    template <typename T, size_t D, int MajorType = Eigen::RowMajor,
              typename IndexType = Eigen::DenseIndex>
    struct EigenTensor {
        using Type = Eigen::TensorMap<Eigen::Tensor<T, D, MajorType, IndexType>>;
        using ConstType = Eigen::TensorMap<Eigen::Tensor<const T, D, MajorType, IndexType>>;

        static Type From(MDTensor& tensor, const std::vector<int64_t>& dims) {
            return Type(static_cast<T*>(tensor.data()), EigenDim<D>::From(dims));
        }

        static Type From(MDTensor& tensor) {
            return From(tensor, tensor.shape);
        }

        static ConstType From(const MDTensor& tensor, const std::vector<int64_t>& dims) {
            return ConstType(static_cast<const T*>(tensor.data()), EigenDim<D>::From(dims));
        }

        static ConstType From(const MDTensor& tensor) {
            return From(tensor, tensor.shape);
        }
    };

    template <typename T, int MajorType = Eigen::RowMajor,
              typename IndexType = Eigen::DenseIndex>
    struct EigenScalar {
        // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
        using Type = Eigen::TensorMap<
            Eigen::TensorFixedSize<T, Eigen::Sizes<>, MajorType, IndexType>>;
        using ConstType = Eigen::TensorMap<
            Eigen::TensorFixedSize<const T, Eigen::Sizes<>, MajorType, IndexType>>;

        static Type From(MDTensor& tensor) {
            return Type(static_cast<T*>(tensor.data()));
        }

        static ConstType From(const MDTensor& tensor) {
            return ConstType(static_cast<const T*>(tensor.data()));
        }
    };

    template <typename T, int MajorType = Eigen::RowMajor,
              typename IndexType = Eigen::DenseIndex>
    struct EigenVector : EigenTensor<T, 1, MajorType, IndexType> {
        // Flatten reshapes a Tensor into an EigenVector.
        static typename EigenVector::Type Flatten(MDTensor& tensor) {
            return EigenVector::From(tensor, {tensor.total()});
        }

        static typename EigenVector::ConstType Flatten(
            const MDTensor& tensor) {
            return EigenVector::From(tensor, {tensor.total()});
        }
    };

    template <typename T, int MajorType = Eigen::RowMajor,
              typename IndexType = Eigen::DenseIndex>
    struct EigenMatrix : EigenTensor<T, 2, MajorType, IndexType> {
        static typename EigenMatrix::Type Reshape(MDTensor& tensor, // NOLINT
                                                  const int num_col_dims) {
            if (const int rank = static_cast<int>(tensor.shape.size()); !(num_col_dims > 0 && num_col_dims < rank)) {
                MD_LOG_ERROR("Input dimension number(num_col_dims) must be between 0 and {}, "
                             "but received number is {}.", rank, num_col_dims);
            }
            const int n = size_to_axis(num_col_dims, tensor.shape);
            const int d = size_from_axis(num_col_dims, tensor.shape);
            return EigenMatrix::From(tensor, {n, d});
        }

        static typename EigenMatrix::ConstType Reshape(const MDTensor& tensor,
                                                       const int num_col_dims) {
            if (const int rank = static_cast<int>(tensor.shape.size()); !(num_col_dims > 0 && num_col_dims < rank)) {
                MD_LOG_ERROR("Input dimension number(num_col_dims) must be between 0 and {}, "
                             "but received number is {}.", rank, num_col_dims);
            }
            const int n = size_to_axis(num_col_dims, tensor.shape);
            const int d = size_from_axis(num_col_dims, tensor.shape);
            return EigenMatrix::From(tensor, {n, d});
        }
    };

    class EigenDeviceWrapper {
    public:
        static std::shared_ptr<EigenDeviceWrapper> GetInstance();
        [[nodiscard]] const Eigen::DefaultDevice* GetDevice() const;

    private:
        Eigen::DefaultDevice device_;
        static std::shared_ptr<EigenDeviceWrapper> instance_;
    };
}
