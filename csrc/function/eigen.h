//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <algorithm>
#include <memory>
#include <vector>
#include "../core/md_tensor.h"
#include "../utils/utils.h"
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

// Interpret FDTensor as EigenTensor and EigenConstTensor.
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenTensor {
  using Type = Eigen::TensorMap<Eigen::Tensor<T, D, MajorType, IndexType>>;

  using ConstType =
      Eigen::TensorMap<Eigen::Tensor<const T, D, MajorType, IndexType>>;

  static Type From(MDTensor& tensor,
                   const std::vector<int64_t>& dims) {  // NOLINT
    return Type(reinterpret_cast<T*>(tensor.data()), EigenDim<D>::From(dims));
  }

  static Type From(MDTensor& tensor) {  // NOLINT
    return From(tensor, tensor.shape);
  }  // NOLINT

  static ConstType From(const MDTensor& tensor,
                        const std::vector<int64_t>& dims) {
    return ConstType(reinterpret_cast<const T*>(tensor.data()),
                     EigenDim<D>::From(dims));
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
    return Type(reinterpret_cast<T*>(tensor.data()));
  }  // NOLINT

  static ConstType From(const MDTensor& tensor) {
    return ConstType(reinterpret_cast<const T*>(tensor.data()));
  }
};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenVector : public EigenTensor<T, 1, MajorType, IndexType> {
  // Flatten reshapes a Tensor into an EigenVector.
  static typename EigenVector::Type Flatten(MDTensor& tensor) {  // NOLINT
    return EigenVector::From(tensor, {tensor.total()});
  }

  static typename EigenVector::ConstType Flatten(
      const MDTensor& tensor) {  // NOLINT
    return EigenVector::From(tensor, {tensor.total()});
  }
};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenMatrix : public EigenTensor<T, 2, MajorType, IndexType> {
  static typename EigenMatrix::Type Reshape(MDTensor& tensor,  // NOLINT
                                            int num_col_dims) {
    int rank = tensor.shape.size();
    if(!(num_col_dims > 0 && num_col_dims < rank)){
             std::cerr<<"Input dimension number(num_col_dims) must be between 0 and "<<rank<<", "
             "but received number is "<<num_col_dims<<"."<<std::endl;
        }
    const int n = size_to_axis(num_col_dims, tensor.shape);
    const int d = size_from_axis(num_col_dims, tensor.shape);
    return EigenMatrix::From(tensor, {n, d});
  }

  static typename EigenMatrix::ConstType Reshape(const MDTensor& tensor,
                                                 int num_col_dims) {
    int rank = tensor.shape.size();
    if(!(num_col_dims > 0 && num_col_dims < rank)){
             std::cerr<<"Input dimension number(num_col_dims) must be between 0 and "<<rank<<", "
             "but received number is "<<num_col_dims<<"."<<std::endl;
             }
    const int n = size_to_axis(num_col_dims, tensor.shape);
    const int d = size_from_axis(num_col_dims, tensor.shape);
    return EigenMatrix::From(tensor, {n, d});
  }
};

class EigenDeviceWrapper {
 public:
  static std::shared_ptr<EigenDeviceWrapper> GetInstance();
  const Eigen::DefaultDevice* GetDevice() const;

 private:
  Eigen::DefaultDevice device_;
  static std::shared_ptr<EigenDeviceWrapper> instance_;
};

}

