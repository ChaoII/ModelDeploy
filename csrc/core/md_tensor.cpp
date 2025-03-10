//
// Created by aichao on 2025/2/20.
//

#include <algorithm>
#include <cstring>
#include "md_tensor.h"
#include <numeric>

#include "allocate.h"

namespace modeldeploy {
    void* MDTensor::mutable_data() {
        if (external_data_ptr_ != nullptr) {
            return external_data_ptr_;
        }
        return buffer_;
    }

    void* MDTensor::data() {
        if (external_data_ptr_ != nullptr) {
            return external_data_ptr_;
        }
        return buffer_;
    }

    const void* MDTensor::data() const {
        if (external_data_ptr_ != nullptr) {
            return external_data_ptr_;
        }
        return buffer_;
    }

    void MDTensor::stop_sharing() {
        if (is_shared()) {
            re_alloc_fn(total_bytes());
            copy_buffer(buffer_, external_data_ptr_, total_bytes());
            external_data_ptr_ = nullptr;
        }
    }


    void MDTensor::set_external_data(const std::vector<int64_t>& new_shape,
                                     const MDDataType& data_type, void* data_buffer) {
        dtype = data_type;
        shape.assign(new_shape.begin(), new_shape.end());
        external_data_ptr_ = data_buffer;
    }

    void MDTensor::expand_dim(int64_t axis) {
        size_t ndim = shape.size();
        if (!(axis >= 0 && axis <= ndim)) {
            std::cerr << "The allowed 'axis' must be in range of " << ndim << std::endl;
        }
        shape.insert(shape.begin() + axis, 1);
    }

    void MDTensor::squeeze(int64_t axis) {
        size_t ndim = shape.size();
        if (!(axis >= 0 && axis < ndim)) {
            std::cerr << "The allowed 'axis' must be in range of" << ndim << std::endl;
        };
        if (shape[axis] != 1) {
            std::cerr << "The No." << axis << "dimension of shape should be 1, but it is " << shape[axis] << "!";
        }
        shape.erase(shape.begin() + axis);
    }

    void MDTensor::allocate(const std::vector<int64_t>& new_shape,
                            const MDDataType& data_type,
                            const std::string& tensor_name) {
        dtype = data_type;
        name = tensor_name;
        shape.assign(new_shape.begin(), new_shape.end());
        size_t nbytes = total_bytes();
        if (!re_alloc_fn(nbytes)) {
            std::cerr << "The FastDeploy MDTensor allocate cpu memory error";
        }
    }

    int MDTensor::total_bytes() const { return total() * md_dtype_size(dtype); }

    int MDTensor::total() const {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
    }

    void MDTensor::resize(size_t new_total_bytes) { re_alloc_fn(new_total_bytes); }

    void MDTensor::resize(const std::vector<int64_t>& new_shape) {
        int numel = total();
        int new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies());
        if (new_numel > numel || external_data_ptr_ != nullptr) {
            size_t nbytes = new_numel * md_dtype_size(dtype);
            re_alloc_fn(nbytes);
        }
        shape.assign(new_shape.begin(), new_shape.end());
        external_data_ptr_ = nullptr;
    }

    void MDTensor::resize(const std::vector<int64_t>& new_shape,
                          const MDDataType& data_type,
                          const std::string& tensor_name) {
        free_fn();
        external_data_ptr_ = nullptr;
        name = tensor_name;
        dtype = data_type;
        int new_nbytes = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                         std::multiplies()) * md_dtype_size(data_type);
        re_alloc_fn(new_nbytes);
        shape.assign(new_shape.begin(), new_shape.end());
    }

    bool MDTensor::reshape(const std::vector<int64_t>& new_shape) {
        int numel = total();
        constexpr int64_t unk_dim_val = -1;
        constexpr int64_t copy_dim_val = 0;

        std::vector<int64_t> output_shape(new_shape.size(), 0);
        int64_t capacity = 1;
        int unk_dim_idx = -1;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            if (new_shape[i] == unk_dim_val) {
                if (unk_dim_idx != -1) {
                    std::cerr << "Only one dimension value of 'shape' in ReshapeOp can "
                        "be -1. But received shape shape" << i << " is also -1." << std::endl;
                }
                unk_dim_idx = i;
            }
            else if (new_shape[i] == copy_dim_val) {
                if (i > shape.size()) {
                    std::cerr << "The index of 0 in `shape` must be less than "
                        "the input tensor X's dimensions. "
                        "But received shape =" << print_vector(new_shape) << ", shape" << i << "= 0, X's shape =" <<
                        print_vector(shape) <<
                        "X's dimensions = " << shape.size() << std::endl;
                    return false;
                }
            }
            else {
                if (new_shape[i] < 0) {
                    std::cerr << "Each dimension value of 'shape' in ReshapeOp must not "
                        "be negative except one unknown dimension. "
                        "But received  shape = " << print_vector(new_shape) << ", shape[" << i << "] =" << new_shape[i]
                        <<
                        std::endl;
                    return false;
                };
            }
            capacity *= (new_shape[i] ? new_shape[i] : shape[i]);
            output_shape[i] = (new_shape[i] ? new_shape[i] : shape[i]);
        }
        if (unk_dim_idx != -1) {
            output_shape[unk_dim_idx] = -numel / capacity;
            if (output_shape[unk_dim_idx] * capacity != -numel) {
                std::cerr << "The 'shape' attribute in ReshapeOp is invalid. "
                    "The input tensor X'size must be divisible by known "
                    "capacity of 'shape'. "
                    "But received X's shape = [" << print_vector(shape) << "], X's size = " << numel << ", "
                    "'shape' is [" << print_vector(new_shape) << "], known capacity of 'shape' is " << capacity << "."
                    <<
                    std::endl;
                return false;
            }
            else {
                if (numel != capacity) {
                    std::cerr << "The 'shape' in ReshapeOp is invalid. "
                        "The input tensor X'size must be equal to the capacity of "
                        "'shape'. But received X's shape = [" << print_vector(shape) << "], X's size = " << numel <<
                        ", 'shape' is "
                        "[" << print_vector(shape) << "], the capacity of 'shape' is " << capacity << "." << std::endl;
                    return false;
                }
            }
            shape = output_shape;
            return true;
        }
        return true;
    }


    bool MDTensor::re_alloc_fn(size_t nbytes) {
        buffer_ = realloc(buffer_, nbytes);
        total_bytes_allocated_ = nbytes;
        return buffer_ != nullptr;
    }

    void MDTensor::free_fn() {
        if (external_data_ptr_ != nullptr) {
            external_data_ptr_ = nullptr;
        }
        if (buffer_ != nullptr) {
            MDHostFree()(buffer_);
            buffer_ = nullptr;
            total_bytes_allocated_ = 0;
        }
    }

    void MDTensor::copy_buffer(void* dst, const void* src, size_t nbytes) {
        std::memcpy(dst, src, nbytes);
    }

    MDTensor::MDTensor(const std::string& tensor_name) { name = tensor_name; }
    MDTensor::MDTensor(const char* tensor_name) { name = tensor_name; }

    MDTensor::MDTensor(const Scalar& scalar) {
        allocate({1}, scalar.dtype());
        switch (scalar.dtype()) {
        case MDDataType::BOOL:
            (static_cast<bool*>(data()))[0] = scalar.to<bool>();
            break;
        case MDDataType::UINT8:
            (static_cast<uint8_t*>(data()))[0] = scalar.to<uint8_t>();
            break;
        case MDDataType::INT8:
            (static_cast<int8_t*>(data()))[0] = scalar.to<int8_t>();
            break;
        case MDDataType::INT16:
            (static_cast<int16_t*>(data()))[0] = scalar.to<int16_t>();
            break;
        case MDDataType::INT32:
            (static_cast<int*>(data()))[0] = scalar.to<int>();
            break;
        case MDDataType::INT64:
            (static_cast<int64_t*>(data()))[0] = scalar.to<int64_t>();
            break;
        case MDDataType::FP32:
            (static_cast<float*>(data()))[0] = scalar.to<float>();
            break;
        case MDDataType::FP64:
            (static_cast<double*>(data()))[0] = scalar.to<double>();
            break;
        default:
            break;
        }
    }

    MDTensor::MDTensor(const MDTensor& other)
        : name(other.name),
          shape(other.shape),
          dtype(other.dtype) {
        // Copy buffer
        if (other.buffer_ == nullptr) {
            free_fn();
        }
        else {
            size_t nbytes = total_bytes();
            if (!re_alloc_fn(nbytes)) {
                std::cerr << "The ModelDeploy MDTensor allocate memory error" << std::endl;
            }
            copy_buffer(buffer_, other.buffer_, nbytes);
        }
        external_data_ptr_ = other.external_data_ptr_;
    }

    MDTensor::MDTensor(MDTensor&& other)
        : name(std::move(other.name)),
          buffer_(other.buffer_),
          shape(std::move(other.shape)),
          dtype(other.dtype),
          external_data_ptr_(other.external_data_ptr_),
          total_bytes_allocated_(other.total_bytes_allocated_) {
        other.name = "";
        other.buffer_ = nullptr;
        other.external_data_ptr_ = nullptr;
    }

    MDTensor& MDTensor::operator=(const MDTensor& other) {
        if (&other != this) {
            // Copy buffer
            if (other.buffer_ == nullptr) {
                free_fn();
                buffer_ = nullptr;
                shape = other.shape;
                name = other.name;
                dtype = other.dtype;
            }
            else {
                resize(other.shape, other.dtype, other.name);
                size_t nbytes = total_bytes();
                copy_buffer(buffer_, other.buffer_, nbytes);
            }
            external_data_ptr_ = other.external_data_ptr_;
        }
        return *this;
    }

    MDTensor& MDTensor::operator=(MDTensor&& other) {
        if (&other != this) {
            free_fn();
            buffer_ = other.buffer_;
            external_data_ptr_ = other.external_data_ptr_;
            shape = std::move(other.shape);
            name = std::move(other.name);
            dtype = other.dtype;
            total_bytes_allocated_ = other.total_bytes_allocated_;
            other.name = "";
            other.buffer_ = nullptr;
            other.external_data_ptr_ = nullptr;
        }
        return *this;
    }
}
