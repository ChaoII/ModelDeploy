#pragma once

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <functional>
#include <iostream>

#include "csrc/core/md_decl.h"

namespace modeldeploy {
    enum class DataType {
        FP32,
        FP64,
        INT32,
        INT64,
        UINT8,
        INT8,
        UNKNOWN
    };


    // 辅助函数实现
    inline std::string datatype_to_string(const DataType dtype) {
        switch (dtype) {
        case DataType::FP32: return "FP32";
        case DataType::FP64: return "FP64";
        case DataType::INT32: return "INT32";
        case DataType::INT64: return "INT64";
        case DataType::UINT8: return "UINT8";
        case DataType::INT8: return "INT8";
        case DataType::UNKNOWN: return "UNKNOWN";
        default: return "";
        }
    }

    inline std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
        return os << datatype_to_string(dtype);
    }


    // 内存池管理器
    class MemoryPool {
    public:
        static void* allocate(size_t size);
        static void deallocate(void* ptr, size_t size);
        static void* reallocate(void* ptr, size_t new_size);
    };

    // 内存块封装，支持引用计数
    class MemoryBlock {
    public:
        explicit MemoryBlock(size_t size);
        MemoryBlock(void* data, size_t size, std::function<void(void*)> deleter);
        ~MemoryBlock();

        void* data() { return data_; }
        [[nodiscard]] const void* data() const { return data_; }
        [[nodiscard]] size_t size() const { return size_; }

    private:
        void* data_;
        size_t size_;
        std::function<void(void*)> deleter_;
    };

    class TensorView;

    class MODELDEPLOY_CXX_EXPORT Tensor {
    public:
        // 构造函数
        Tensor() = default;
        Tensor(const std::vector<int64_t>& shape, DataType dtype, std::string name = "");
        Tensor(void* data, const std::vector<int64_t>& shape, DataType dtype,
               std::function<void(void*)> deleter = nullptr, std::string name = "");
        Tensor(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        ~Tensor() = default;

        // 运算符重载
        Tensor& operator=(const Tensor& other);
        Tensor& operator=(Tensor&& other) noexcept;

        // 基本属性
        void* data();
        [[nodiscard]] const void* data() const;
        [[nodiscard]] size_t size() const; // 返回元素总数
        [[nodiscard]] size_t byte_size() const; // 返回字节大小
        [[nodiscard]] const std::vector<int64_t>& shape() const;
        [[nodiscard]] DataType dtype() const;
        [[nodiscard]] const std::string& get_name() const;
        void set_name(const std::string& name);
        static size_t get_element_size(DataType dtype);
        [[nodiscard]] size_t outer_dim(int axis) const;
        [[nodiscard]] bool get_owns_data() const;
        void set_owns_data(bool owns_data);

        // 数据操作接口 - 优化版本
        template <typename T>
        void set_data(const T* data, size_t size);
        template <typename T>
        void set_data(std::vector<T>&& data); // 移动语义版本
        template <typename T>
        const T* data_ptr() const; // 返回指针而非复制
        template <typename T>
        std::vector<T> get_data() const; // 保留兼容性

        // 索引操作
        template <typename T>
        T& at(const std::vector<int64_t>& indices);

        template <typename T>
        [[nodiscard]] const T& at(const std::vector<int64_t>& indices) const;

        [[nodiscard]] float at(const std::vector<int64_t>& indices) const;

        // 惰性Tensor操作
        [[nodiscard]] TensorView view() const;
        [[nodiscard]] TensorView reshape(const std::vector<int64_t>& new_shape) const;
        [[nodiscard]] TensorView transpose(const std::vector<int64_t>& axes) const;
        [[nodiscard]] TensorView slice(const std::vector<int64_t>& starts, const std::vector<int64_t>& ends) const;

        // 强制具体化操作
        [[nodiscard]] Tensor materialize() const;

        // 原地操作
        [[nodiscard]] Tensor clone() const;
        void resize(const std::vector<int64_t>& shape, const DataType& dtype, const std::string& name = "");
        void allocate(const std::vector<int64_t>& shape, const DataType& dtype, const std::string& name = "");
        void from_external_memory(void* data,
                                  const std::vector<int64_t>& shape, DataType dtype,
                                  std::function<void(void*)> deleter = [](void*) {
                                  }, std::string name = "");
        // 其他操作
        void set_display_max_ele_width(int width);
        void print(std::ostream& os = std::cout) const;
        [[nodiscard]] std::string to_string() const;
        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
        static Tensor concat(const std::vector<Tensor>& tensors, int axis);
        [[nodiscard]] Tensor softmax(int axis = -1) const;
        void expand_dim(int64_t axis);

        // 工具函数
        [[nodiscard]] bool is_same_shape(const Tensor& other) const;
        [[nodiscard]] size_t get_dim_size(size_t dim) const;
        [[nodiscard]] size_t get_rank() const;
        [[nodiscard]] bool is_empty() const;
        [[nodiscard]] size_t compute_index(const std::vector<size_t>& indices) const;

    private:
        int display_max_ele_width_ = 8;
        std::string name_{};
        std::vector<int64_t> shape_{0};
        std::vector<int64_t> strides_{};
        DataType dtype_{DataType::FP32};
        std::shared_ptr<MemoryBlock> memory_{}; // 替代原来的data_buffer_
        size_t element_size_{0};
        void* data_ptr_{nullptr}; // 指向实际数据的指针，可能是memory_内部的数据或外部数据
        bool owns_data_{true}; // 是否拥有数据所有权

        // 辅助函数
        static void validate_shape(const std::vector<int64_t>& shape);
        [[nodiscard]] size_t calculate_total_size() const;
        void calculate_strides();
        friend class TensorView;
    };

    // TensorView类 - 提供无复制的视图
    class MODELDEPLOY_CXX_EXPORT TensorView {
    public:
        explicit TensorView(const Tensor& tensor);
        TensorView(const Tensor& tensor,
                   const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& strides,
                   void* data_ptr);

        // TensorView接口
        [[nodiscard]] const std::vector<int64_t>& shape() const { return shape_; }
        [[nodiscard]] const std::vector<int64_t>& strides() const { return strides_; }
        void* data() { return data_ptr_; }
        [[nodiscard]] const void* data() const { return data_ptr_; }
        [[nodiscard]] DataType dtype() const { return base_tensor_->dtype(); }
        [[nodiscard]] size_t size() const;
        [[nodiscard]] size_t byte_size() const;
        [[nodiscard]] size_t get_element_size() const;
        [[nodiscard]] bool is_contiguous() const;
        // 转换为具体Tensor
        [[nodiscard]] Tensor to_tensor() const;

        // 视图操作
        [[nodiscard]] TensorView reshape(const std::vector<int64_t>& new_shape) const;
        [[nodiscard]] TensorView transpose(const std::vector<int64_t>& axes) const;
        [[nodiscard]] TensorView slice(const std::vector<int64_t>& starts, const std::vector<int64_t>& ends) const;

        // 索引操作
        template <typename T>
        T& at(const std::vector<int64_t>& indices);
        template <typename T>
        const T& at(const std::vector<int64_t>& indices) const;

    private:
        std::shared_ptr<const Tensor> base_tensor_; // 持有基础张量的引用
        std::vector<int64_t> shape_;
        std::vector<int64_t> strides_;
        void* data_ptr_;
    };
} // namespace modeldeploy
