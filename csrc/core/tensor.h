#pragma once

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <functional>
#include <iostream>
#include "core/md_decl.h"
#include "core/enum_variables.h"

namespace modeldeploy {
    // 内存块封装，支持引用计数
    class MemoryBlock {
    public:
        explicit MemoryBlock(size_t size, Device device);
        // 通过外部buffer拷贝buffer并构造一个MemoryBlock
        explicit MemoryBlock(const void* data, size_t size, Device device);
        // 外部数据共享构造一个MemoryBlock，如果不传deleter，则内存的释放由外部管理，比如OpenCV的mat
        explicit MemoryBlock(void* data, size_t size, Device device, std::function<void(void*)> deleter);
        ~MemoryBlock();
        void* data() { return data_; }
        [[nodiscard]] const void* data() const { return data_; }
        [[nodiscard]] size_t size() const { return size_; }
        bool copy_from_extern_buffer(void* data, size_t size, Device extern_device) const;
        bool shared_from_extern_buffer(void* data, size_t size, Device extern_device);

    private:
        void* data_;
        size_t size_;
        Device device_;
        std::function<void(void*)> deleter_;
    };

    class MODELDEPLOY_CXX_EXPORT Tensor {
    public:
        // 构造函数
        Tensor() = default;
        Tensor(const std::vector<int64_t>& shape, DataType dtype, Device device = Device::CPU, std::string name = "");
        Tensor(void* data, const std::vector<int64_t>& shape, DataType dtype, Device device,
               std::function<void(void*)> deleter = nullptr, std::string name = "");
        // 浅拷贝：共享 MemoryBlock（引用计数管理），深拷贝请用 clone()
        Tensor(const Tensor& other) = default;
        Tensor(Tensor&& other) noexcept = default;
        ~Tensor() = default;

        // 运算符重载（浅拷贝，共享底层内存）
        Tensor& operator=(const Tensor& other) = default;
        Tensor& operator=(Tensor&& other) noexcept = default;

        // 基本属性
        void* data();
        [[nodiscard]] const void* data() const;
        [[nodiscard]] size_t size() const; // 返回元素总数（缓存，O(1)）
        [[nodiscard]] size_t byte_size() const; // 返回字节大小（缓存，O(1)）
        [[nodiscard]] const std::vector<int64_t>& shape() const;
        [[nodiscard]] const std::vector<int64_t>& strides() const { return strides_; }
        [[nodiscard]] DataType dtype() const;
        [[nodiscard]] Device device() const;
        [[nodiscard]] const std::string& get_name() const;
        void set_name(const std::string& name);
        static size_t get_element_size(DataType dtype);
        [[nodiscard]] size_t outer_dim(int axis) const;
        [[nodiscard]] bool get_owns_data() const;
        void set_owns_data(bool owns_data);

        // DataLayout（NCHW/NHWC 等）
        void set_layout(DataLayout layout) { layout_ = layout; }
        [[nodiscard]] DataLayout layout() const { return layout_; }

        // 数据操作接口 - 优化版本
        template <typename T>
        void set_data(const T* data, size_t size, Device device, bool copy);
        template <typename T>
        const T* data_ptr() const; // 返回指针而非复制
        template <typename T>
        T* data_ptr(); // 返回指针而非复制

        // 索引操作（基于 strides，支持非连续视图）
        template <typename T>
        T& at(const std::vector<int64_t>& indices);

        template <typename T>
        [[nodiscard]] const T& at(const std::vector<int64_t>& indices) const;

        [[nodiscard]] float at(const std::vector<int64_t>& indices) const;

        // 视图操作（返回共享内存的 Tensor，非连续视图通过 strides 表达）
        // 这些方法返回 Tensor，共享底层 memory_，仅调整 shape/strides/data_ptr_
        [[nodiscard]] Tensor view() const;
        [[nodiscard]] Tensor reshape(const std::vector<int64_t>& new_shape) const;
        [[nodiscard]] Tensor transpose(const std::vector<int64_t>& axes) const;
        [[nodiscard]] Tensor slice(const std::vector<int64_t>& starts, const std::vector<int64_t>& ends) const;

        // 检查内存是否连续（行优先，按 strides 判定）
        [[nodiscard]] bool is_contiguous() const;
        // 物化：非连续时复制为连续内存并返回新 Tensor；连续时返回 *this
        [[nodiscard]] Tensor contiguous() const;

        // 原地操作
        [[nodiscard]] Tensor clone() const;
        void resize(const std::vector<int64_t>& shape, const DataType& dtype, const std::string& name = "");
        void allocate(const std::vector<int64_t>& shape,
                      const DataType& dtype,
                      Device device = Device::CPU,
                      const std::string& name = "");
        void from_external_memory(void* data,
                                  const std::vector<int64_t>& shape, DataType dtype,
                                  std::function<void(void*)> deleter = nullptr,
                                  Device device = Device::CPU,
                                  std::string name = "");
        bool copy_from_extern_memory(void* data, size_t byte_size, Device extern_device);
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
        DataLayout layout_{DataLayout::UNDEFINED};
        std::shared_ptr<MemoryBlock> memory_{}; // 引用计数内存块，可被多个 Tensor 共享
        Device device_{Device::CPU};
        size_t element_size_{0};
        size_t numel_{0}; // 元素总数缓存（O(1) 访问）
        size_t total_bytes_{0}; // 字节数缓存（O(1) 访问）
        void* data_ptr_{nullptr}; // 指向实际数据的指针，可能是memory_内部的数据或外部数据
        bool owns_data_{true}; // 是否拥有数据所有权（外部内存时 false，仅供语义标记）

        // 辅助函数
        static void validate_shape(const std::vector<int64_t>& shape);
        [[nodiscard]] size_t calculate_total_size() const;
        void calculate_strides();
        // 重算 numel_/total_bytes_ 缓存（shape_/dtype_ 变化后调用）
        void refresh_cache();
    };
} // namespace modeldeploy
