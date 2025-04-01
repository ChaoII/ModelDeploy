//
// Created by aichao on 2025/2/20.
//

#include <algorithm>
#include <cstring>
#include <numeric>
#include "csrc/core/md_tensor.h"
#include "csrc/utils/utils.h"
#include "csrc/core/allocate.h"

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


    void MDTensor::print_info(const std::string& prefix) const {
        double mean = 0;
        double max = -99999999;
        double min = 99999999;
        if (dtype == MDDataType::Type::FP16) {
            calculate_statis_info<float16>(data(), total(), &mean, &max, &min);
        }
        else if (dtype == MDDataType::Type::FP32) {
            calculate_statis_info<float>(data(), total(), &mean, &max, &min);
        }
        else if (dtype == MDDataType::Type::FP64) {
            calculate_statis_info<double>(data(), total(), &mean, &max, &min);
        }
        else if (dtype == MDDataType::Type::INT8) {
            calculate_statis_info<int8_t>(data(), total(), &mean, &max, &min);
        }
        else if (dtype == MDDataType::Type::UINT8) {
            calculate_statis_info<uint8_t>(data(), total(), &mean, &max, &min);
        }
        else if (dtype == MDDataType::Type::INT32) {
            calculate_statis_info<int32_t>(data(), total(), &mean, &max, &min);
        }
        else if (dtype == MDDataType::Type::INT64) {
            calculate_statis_info<int64_t>(data(), total(), &mean, &max, &min);
        }
        else {
            MD_LOG_ERROR << "PrintInfo function doesn't support current situation, maybe you "
                "need enhance this function now." << std::endl;
        }
        std::cout << termcolor::green << prefix << std::endl;
        std::cout << "name=" << name << "\n"
            << "shape = " << print_vector(shape) << "\n"
            << "buffer_=" << buffer_ << "\n"
            << "external_data_ptr=" << external_data_ptr_ << "\n"
            << "dtype=" << MDDataType::str(dtype)
            << "mean=" << mean
            << "max=" << max
            << "min=" << min
            << termcolor::reset << std::endl;
    }


    void MDTensor::stop_sharing() {
        if (is_shared()) {
            re_alloc_fn(total_bytes());
            copy_buffer(buffer_, external_data_ptr_, total_bytes());
            external_data_ptr_ = nullptr;
        }
    }


    void MDTensor::set_external_data(const std::vector<int64_t>& new_shape,
                                     const MDDataType::Type& data_type, void* data_buffer) {
        dtype = data_type;
        shape.assign(new_shape.begin(), new_shape.end());
        external_data_ptr_ = data_buffer;
    }

    void MDTensor::expand_dim(const int64_t axis) {
        const size_t ndim = shape.size();
        if (!(axis >= 0 && axis <= ndim)) {
            MD_LOG_ERROR << "The allowed 'axis' must be in range of " << ndim << "." << std::endl;
        }
        shape.insert(shape.begin() + axis, 1);
    }

    void MDTensor::squeeze(const int64_t axis) {
        const size_t ndim = shape.size();
        if (!(axis >= 0 && axis < ndim)) {
            MD_LOG_ERROR << "The allowed 'axis' must be in range of " << ndim << "." << std::endl;
        }
        if (shape[axis] != 1) {
            MD_LOG_ERROR << "The No." << axis << " dimension of shape should be 1, "
                "but it is " << shape[axis] << "!" << std::endl;
        }
        shape.erase(shape.begin() + axis);
    }

    void MDTensor::allocate(const std::vector<int64_t>& new_shape,
                            const MDDataType::Type& data_type,
                            const std::string& tensor_name) {
        dtype = data_type;
        name = tensor_name;
        shape.assign(new_shape.begin(), new_shape.end());
        const size_t num_bytes = total_bytes();
        if (!re_alloc_fn(num_bytes)) {
            MD_LOG_ERROR << "The FastDeploy MDTensor allocate cpu memory error!" << std::endl;
        }
    }

    int MDTensor::total_bytes() const { return total() * MDDataType::size(dtype); }

    int MDTensor::total() const {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
    }

    void MDTensor::resize(const size_t total_bytes) { re_alloc_fn(total_bytes); }

    void MDTensor::resize(const std::vector<int64_t>& new_shape) {
        const int num_el = total();
        const int new_num_el = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies());
        if (new_num_el > num_el || external_data_ptr_ != nullptr) {
            const size_t total_bytes = new_num_el * MDDataType::size(dtype);
            re_alloc_fn(total_bytes);
        }
        shape.assign(new_shape.begin(), new_shape.end());
        external_data_ptr_ = nullptr;
    }

    void MDTensor::resize(const std::vector<int64_t>& new_shape,
                          const MDDataType::Type& data_type,
                          const std::string& tensor_name) {
        free_fn();
        external_data_ptr_ = nullptr;
        name = tensor_name;
        dtype = data_type;
        const int new_total_bytes = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                                    std::multiplies()) * MDDataType::size(data_type);
        re_alloc_fn(new_total_bytes);
        shape.assign(new_shape.begin(), new_shape.end());
    }

    bool MDTensor::reshape(const std::vector<int64_t>& new_shape) {
        const int num_el = total();
        constexpr int64_t unk_dim_val = -1;
        constexpr int64_t copy_dim_val = 0;

        std::vector<int64_t> output_shape(new_shape.size(), 0);
        int64_t capacity = 1;
        int unk_dim_idx = -1;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            if (new_shape[i] == unk_dim_val) {
                if (unk_dim_idx != -1) {
                    MD_LOG_ERROR << "Only one dimension value of 'shape' in ReshapeOp can "
                        "be -1. But received shape shape" << i << "is also -1.";
                }
                unk_dim_idx = static_cast<int>(i);
            }
            else if (new_shape[i] == copy_dim_val) {
                if (i > shape.size()) {
                    MD_LOG_ERROR <<
                        "The index of 0 in `shape` must be less than "
                        "the input tensor X's dimensions. But received shape ="
                        << print_vector(new_shape) << ", shape " << i << "= 0, "
                        "X's shape = " << print_vector(shape) << " X's dimensions = "
                        << print_vector(shape) << "." << std::endl;
                    return false;
                }
            }
            else {
                if (new_shape[i] < 0) {
                    MD_LOG_ERROR <<
                        "Each dimension value of 'shape' in ReshapeOp must not "
                        "be negative except one unknown dimension. "
                        "But received  shape = " << print_vector(new_shape)
                        << ", shape[" << i << "] = " << new_shape[i] << "." << std::endl;
                    return false;
                }
            }
            capacity *= new_shape[i] ? new_shape[i] : shape[i];
            output_shape[i] = new_shape[i] ? new_shape[i] : shape[i];
        }
        if (unk_dim_idx != -1) {
            output_shape[unk_dim_idx] = -num_el / capacity;
            if (output_shape[unk_dim_idx] * capacity != -num_el) {
                MD_LOG_ERROR <<
                    "The 'shape' attribute in ReshapeOp is invalid. "
                    "The input tensor X's size must be divisible by known "
                    "capacity of 'shape'. But received X's shape = ["
                    << print_vector(shape) << "], X's size = " << num_el << ", "
                    "'shape' is [" << print_vector(new_shape) << "], "
                    "known capacity of 'shape' is " << capacity << "." << std::endl;
                return false;
            }
            if (num_el != capacity) {
                MD_LOG_ERROR
                    << "The 'shape' in ReshapeOp is invalid. "
                    "The input tensor X's size must be equal to the capacity of "
                    "'shape'. But received X's shape = [" << print_vector(shape) << "], "
                    "X's size = " << num_el << "'shape' is [" << print_vector(shape)
                    << "], the capacity of 'shape' is " << capacity << "." << std::endl;
                return false;
            }
            shape = output_shape;
            return true;
        }
        return true;
    }


    bool MDTensor::re_alloc_fn(size_t num_bytes) {
        buffer_ = realloc(buffer_, num_bytes);
        total_bytes_allocated_ = num_bytes;
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

    void MDTensor::copy_buffer(void* dst, const void* src, const size_t num_bytes) {
        std::memcpy(dst, src, num_bytes);
    }

    MDTensor::MDTensor(const std::string& tensor_name) { name = tensor_name; }
    MDTensor::MDTensor(const char* tensor_name) { name = tensor_name; }

    MDTensor::MDTensor(const Scalar& scalar) {
        allocate({1}, scalar.dtype());
        switch (scalar.dtype()) {
        case MDDataType::Type::BOOL:
            static_cast<bool*>(data())[0] = scalar.to<bool>();
            break;
        case MDDataType::Type::UINT8:
            static_cast<uint8_t*>(data())[0] = scalar.to<uint8_t>();
            break;
        case MDDataType::Type::INT8:
            static_cast<int8_t*>(data())[0] = scalar.to<int8_t>();
            break;
        case MDDataType::Type::INT16:
            static_cast<int16_t*>(data())[0] = scalar.to<int16_t>();
            break;
        case MDDataType::Type::INT32:
            static_cast<int*>(data())[0] = scalar.to<int>();
            break;
        case MDDataType::Type::INT64:
            static_cast<int64_t*>(data())[0] = scalar.to<int64_t>();
            break;
        case MDDataType::Type::FP32:
            static_cast<float*>(data())[0] = scalar.to<float>();
            break;
        case MDDataType::Type::FP64:
            static_cast<double*>(data())[0] = scalar.to<double>();
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
            const size_t num_bytes = total_bytes();
            if (!re_alloc_fn(num_bytes)) {
                MD_LOG_ERROR << "The ModelDeploy MDTensor allocate memory error!" << std::endl;
            }
            copy_buffer(buffer_, other.buffer_, num_bytes);
        }
        external_data_ptr_ = other.external_data_ptr_;
    }

    MDTensor::MDTensor(MDTensor&& other) noexcept:
        name(std::move(other.name)),
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
                const size_t num_bytes = total_bytes();
                copy_buffer(buffer_, other.buffer_, num_bytes);
            }
            external_data_ptr_ = other.external_data_ptr_;
        }
        return *this;
    }

    MDTensor& MDTensor::operator=(MDTensor&& other) noexcept {
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
