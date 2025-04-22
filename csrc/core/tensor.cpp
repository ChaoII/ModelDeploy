#include "tensor.h"
#include <stdexcept>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <utility>
#include <cmath>

#include "md_log.h"

namespace modeldeploy {
    // MemoryPool实现
    void* MemoryPool::allocate(const size_t size) {
        return malloc(size);
    }

    void MemoryPool::deallocate(void* ptr, size_t) {
        free(ptr);
    }

    void* MemoryPool::reallocate(void* ptr, const size_t new_size) {
        return realloc(ptr, new_size);
    }

    // MemoryBlock实现
    MemoryBlock::MemoryBlock(const size_t size)
        : size_(size), deleter_([](void* ptr) { MemoryPool::deallocate(ptr, 0); }) {
        data_ = MemoryPool::allocate(size);
        if (!data_) {
            throw std::bad_alloc();
        }
    }

    MemoryBlock::MemoryBlock(void* data, const size_t size, std::function<void(void*)> deleter)
        : data_(data), size_(size), deleter_(std::move(deleter)) {
    }

    MemoryBlock::~MemoryBlock() {
        if (data_ && deleter_) {
            deleter_(data_);
            data_ = nullptr;
        }
    }

    // Tensor实现
    size_t Tensor::get_element_size(const DataType dtype) {
        switch (dtype) {
        case DataType::FP32: return sizeof(float);
        case DataType::FP64: return sizeof(double);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::INT64: return sizeof(int64_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::INT8: return sizeof(int8_t);
        default: throw std::runtime_error("不支持的数据类型");
        }
    }

    Tensor::Tensor(const std::vector<int64_t>& shape, const DataType dtype, std::string name)
        : name_(std::move(name)), shape_(shape), dtype_(dtype), element_size_(get_element_size(dtype)) {
        validate_shape(shape);
        size_t total_size = calculate_total_size();
        memory_ = std::make_shared<MemoryBlock>(total_size);
        data_ptr_ = memory_->data();
        calculate_strides();
    }

    Tensor::Tensor(void* data, const std::vector<int64_t>& shape, const DataType dtype,
                   std::function<void(void*)> deleter, std::string name)
        : name_(std::move(name)), shape_(shape), dtype_(dtype), element_size_(get_element_size(dtype)) {
        validate_shape(shape);
        size_t total_size = calculate_total_size();

        if (deleter) {
            // 使用外部内存和自定义删除器
            memory_ = std::make_shared<MemoryBlock>(data, total_size, std::move(deleter));
            owns_data_ = false;
        }
        else {
            // 复制外部数据
            memory_ = std::make_shared<MemoryBlock>(total_size);
            std::memcpy(memory_->data(), data, total_size);
            owns_data_ = true;
        }
        data_ptr_ = memory_->data();
        calculate_strides();
    }

    Tensor::Tensor(const Tensor& other)
        : name_(other.name_), shape_(other.shape_), strides_(other.strides_),
          dtype_(other.dtype_), element_size_(other.element_size_) {
        if (other.owns_data_) {
            // 如果原始张量拥有数据，我们需要复制数据
            memory_ = std::make_shared<MemoryBlock>(other.byte_size());
            data_ptr_ = memory_->data();
            std::memcpy(data_ptr_, other.data_ptr_, other.byte_size());
            owns_data_ = true;
        }
        else {
            // 如果原始张量不拥有数据，我们共享同一个内存块
            memory_ = other.memory_;
            data_ptr_ = other.data_ptr_;
            owns_data_ = false;
        }
    }

    Tensor::Tensor(Tensor&& other) noexcept
        : name_(std::move(other.name_)), shape_(std::move(other.shape_)),
          strides_(std::move(other.strides_)), dtype_(other.dtype_),
          memory_(std::move(other.memory_)), element_size_(other.element_size_),
          data_ptr_(other.data_ptr_), owns_data_(other.owns_data_) {
        other.data_ptr_ = nullptr;
        other.owns_data_ = false;
    }

    Tensor& Tensor::operator=(const Tensor& other) {
        if (this != &other) {
            name_ = other.name_;
            shape_ = other.shape_;
            strides_ = other.strides_;
            dtype_ = other.dtype_;
            element_size_ = other.element_size_;

            if (other.owns_data_) {
                // 复制数据
                memory_ = std::make_shared<MemoryBlock>(other.byte_size());
                data_ptr_ = memory_->data();
                std::memcpy(data_ptr_, other.data_ptr_, other.byte_size());
                owns_data_ = true;
            }
            else {
                // 共享内存
                memory_ = other.memory_;
                data_ptr_ = other.data_ptr_;
                owns_data_ = false;
            }
        }
        return *this;
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            name_ = std::move(other.name_);
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            dtype_ = other.dtype_;
            element_size_ = other.element_size_;
            memory_ = std::move(other.memory_);
            data_ptr_ = other.data_ptr_;
            owns_data_ = other.owns_data_;

            other.data_ptr_ = nullptr;
            other.owns_data_ = false;
        }
        return *this;
    }

    void* Tensor::data() {
        return data_ptr_;
    }

    const void* Tensor::data() const {
        return data_ptr_;
    }

    size_t Tensor::size() const {
        return std::accumulate(shape_.begin(), shape_.end(),
                               1LL, std::multiplies<>());
    }

    size_t Tensor::byte_size() const {
        return size() * element_size_;
    }

    const std::vector<int64_t>& Tensor::shape() const {
        return shape_;
    }

    DataType Tensor::dtype() const {
        return dtype_;
    }

    const std::string& Tensor::name() const {
        return name_;
    }

    void Tensor::set_name(const std::string& name) {
        name_ = name;
    }

    template <typename T>
    void Tensor::set_data(const T* data, const size_t size) {
        if (size * sizeof(T) != byte_size()) {
            throw std::runtime_error("数据大小不匹配");
        }
        std::memcpy(data_ptr_, data, byte_size());
    }

    template <typename T>
    void Tensor::set_data(std::vector<T>&& data) {
        if (data.size() * sizeof(T) != byte_size()) {
            throw std::runtime_error("数据大小不匹配");
        }
        // 创建新的内存块直接使用移动后的数据
        auto raw_data = data.data();
        size_t data_size = data.size() * sizeof(T);
        // 使用自定义删除器释放内存
        data.clear(); // 清空但不释放内存
        memory_ = std::make_shared<MemoryBlock>(
            raw_data, data_size,
            [vec = std::move(data)](void*) mutable {
                // 当内存块被销毁时，让vector的析构函数释放内存
                std::vector<T>().swap(vec);
            }
        );

        data_ptr_ = memory_->data();
        owns_data_ = true;
    }

    template <typename T>
    const T* Tensor::data_ptr() const {
        if (sizeof(T) != element_size_) {
            throw std::runtime_error("类型大小不匹配");
        }
        return static_cast<const T*>(data_ptr_);
    }

    template <typename T>
    std::vector<T> Tensor::get_data() const {
        if (sizeof(T) != element_size_) {
            throw std::runtime_error("类型大小不匹配");
        }
        std::vector<T> result(size());
        std::memcpy(result.data(), data_ptr_, byte_size());
        return result;
    }


    // 索引操作
    template <typename T>
    T& Tensor::at(const std::vector<int64_t>& indices) {
        if (indices.size() != shape_.size()) {
            throw std::runtime_error("索引数量与张量维度不匹配");
        }
        // 验证索引范围
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || indices[i] >= shape_[i]) {
                throw std::runtime_error("");
            }
        }
        // 计算偏移量
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * strides_[i];
        }
        return *reinterpret_cast<T*>(static_cast<char*>(data_ptr_) + offset * element_size_);
    }

    template <typename T>
    const T& Tensor::at(const std::vector<int64_t>& indices) const {
        return const_cast<Tensor*>(this)->at<T>(indices);
    }

    [[nodiscard]] float Tensor::at(const std::vector<int64_t>& indices) const {
        return at<float>(indices);
    }

    void Tensor::calculate_strides() {
        strides_.resize(shape_.size());
        if (strides_.empty()) return;

        strides_.back() = 1;
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    TensorView Tensor::view() const {
        return TensorView(*this);
    }

    TensorView Tensor::reshape(const std::vector<int64_t>& new_shape) const {
        // 验证新形状是否与当前元素数量匹配
        const size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                                1LL, std::multiplies());
        if (new_size != size()) {
            throw std::runtime_error("无法重塑：元素数量不匹配");
        }
        std::vector<int64_t> new_strides(new_shape.size());
        if (!new_strides.empty()) {
            new_strides.back() = 1;
            for (int i = static_cast<int>(new_shape.size()) - 2; i >= 0; --i) {
                new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
            }
        }
        return TensorView{*this, new_shape, new_strides, data_ptr_};
    }

    TensorView Tensor::transpose(const std::vector<int64_t>& axes) const {
        if (axes.size() != shape_.size()) {
            throw std::runtime_error("转置的轴数与tensor维度不匹配");
        }
        // 检查轴是否有效
        std::vector used(shape_.size(), false);
        for (auto& axe : axes) {
            if (axe < 0 || axe >= static_cast<int64_t>(shape_.size())) {
                throw std::runtime_error("转置轴超出范围");
            }
            if (used[axe]) {
                throw std::runtime_error("转置轴重复");
            }
            used[axe] = true;
        }
        // 创建新的形状和步长
        std::vector<int64_t> new_shape(shape_.size());
        std::vector<int64_t> new_strides(shape_.size());
        for (size_t i = 0; i < axes.size(); ++i) {
            new_shape[i] = shape_[axes[i]];
            new_strides[i] = strides_[axes[i]];
        }
        return TensorView{*this, new_shape, new_strides, data_ptr_};
    }

    TensorView Tensor::slice(const std::vector<int64_t>& starts, const std::vector<int64_t>& ends) const {
        if (starts.size() != shape_.size() || ends.size() != shape_.size()) {
            throw std::runtime_error("切片的起始和结束索引必须与tensor维度匹配");
        }
        // 验证切片范围
        std::vector<int64_t> new_shape(shape_.size());
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (starts[i] < 0 || starts[i] >= shape_[i] || ends[i] > shape_[i] || starts[i] >= ends[i]) {
                throw std::runtime_error("无效的切片范围");
            }
            new_shape[i] = ends[i] - starts[i];
        }
        // 计算新的起始数据指针
        size_t offset = 0;
        for (size_t i = 0; i < shape_.size(); ++i) {
            offset += starts[i] * strides_[i];
        }
        char* new_data_ptr = static_cast<char*>(data_ptr_) + offset * element_size_;
        return TensorView{*this, new_shape, strides_, new_data_ptr};
    }

    Tensor Tensor::materialize() const {
        return *this;
    }

    Tensor Tensor::clone() const {
        Tensor result(shape_, dtype_);
        std::memcpy(result.data_ptr_, data_ptr_, calculate_total_size());
        return result;
    }

    void Tensor::resize(const std::vector<int64_t>& shape, const DataType& dtype, const std::string& name) {
        // 验证新形状
        validate_shape(shape);
        // 保存旧的数据信息
        const DataType old_dtype = dtype_;
        const size_t old_total_size = byte_size();
        const void* old_data = data_ptr_;

        // 更新元数据
        if (!name.empty()) {
            name_ = name;
        }
        shape_ = shape;
        dtype_ = dtype;
        element_size_ = get_element_size(dtype);
        // 重新计算步长
        calculate_strides();
        // 计算新的总大小
        size_t new_size = calculate_total_size();
        // 如果数据类型和大小都没变，不需要重新分配
        if (old_dtype == dtype && old_total_size == new_size) {
            return;
        }
        // 分配新内存
        auto new_memory = std::make_shared<MemoryBlock>(new_size);
        void* new_data = new_memory->data();
        // 如果数据类型相同，尝试复制旧数据
        if (old_dtype == dtype && old_data != nullptr) {
            const size_t copy_size = std::min(old_total_size, new_size);
            std::memcpy(new_data, old_data, copy_size);
        }
        // 更新内存和数据指针
        memory_ = std::move(new_memory);
        data_ptr_ = memory_->data();
        owns_data_ = true;
    }

    void Tensor::allocate(const std::vector<int64_t>& shape, const DataType& dtype, const std::string& name) {
        validate_shape(shape);
        name_ = name;
        shape_ = shape;
        dtype_ = dtype;
        element_size_ = get_element_size(dtype);
        calculate_strides();
        size_t total_size = calculate_total_size();
        memory_ = std::make_shared<MemoryBlock>(total_size);
        data_ptr_ = memory_->data();
        owns_data_ = true;
    }

    void Tensor::from_external_memory(void* data, const std::vector<int64_t>& shape, const DataType dtype,
                                      std::function<void(void*)> deleter, std::string name) {
        name_ = std::move(name);
        shape_ = shape,
            dtype_ = dtype,
            element_size_ = get_element_size(dtype);
        validate_shape(shape);
        size_t total_size = calculate_total_size();
        if (deleter) {
            // 使用外部内存和自定义删除器
            memory_ = std::make_shared<MemoryBlock>(data, total_size, std::move(deleter));
            owns_data_ = false;
        }
        else {
            // 复制外部数据
            memory_ = std::make_shared<MemoryBlock>(total_size);
            std::memcpy(memory_->data(), data, total_size);
            owns_data_ = true;
        }
        data_ptr_ = memory_->data();
        calculate_strides();
    }


    // 友元函数，用于流输出
    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        tensor.print(os);
        return os;
    }

    void Tensor::set_display_max_ele_width(int width) {
        display_max_ele_width_ = width;
    }


    std::string data_ptr_to_string(const void* data, const size_t index, const DataType dtype, int max_ele_width) {
        std::ostringstream oss;
        switch (dtype) {
        case DataType::FP32:
            oss << std::fixed << std::setw(max_ele_width) << std::setprecision(4) << static_cast<const float*>(data)[
                index];
            break;
        case DataType::FP64:
            oss << static_cast<const double*>(data)[index];
            break;
        case DataType::INT32:
            oss << static_cast<const int32_t*>(data)[index];
            break;
        case DataType::INT64:
            oss << static_cast<const int64_t*>(data)[index];
            break;
        case DataType::INT8:
            oss << static_cast<const int8_t*>(data)[index];
            break;
        case DataType::UINT8:
            oss << static_cast<const uint8_t*>(data)[index];
            break;
        default:
            oss << "";
            break;
        }
        return oss.str();
    }

    // 显示张量内容
    void Tensor::print(std::ostream& os) const {
        if (is_empty()) {
            std::cout << "Empty tensor" << std::endl;
            return;
        }
        std::function<void(const void*, const std::vector<int64_t>&, std::vector<size_t>&, size_t)> print_helper;
        print_helper = [&](const void* data, const std::vector<int64_t>& shape,
                           std::vector<size_t>& indices, const size_t dim) {
            if (dim == shape.size() - 1) {
                os << "[";
                if (const size_t size = shape[dim]; size > 10) {
                    // 打印前18个元素
                    for (size_t i = 0; i < 8; ++i) {
                        indices[dim] = i;
                        os << data_ptr_to_string(data, compute_index(indices), dtype_, display_max_ele_width_) << ",";
                    }
                    os << std::setw(display_max_ele_width_) << "...";
                    os << ",";
                    // 打印最后一个元素
                    indices[dim] = size - 1;
                    os << data_ptr_to_string(data, compute_index(indices), dtype_, display_max_ele_width_);
                }
                else {
                    // 打印所有元素
                    for (size_t i = 0; i < size; ++i) {
                        indices[dim] = i;
                        os << data_ptr_to_string(data, compute_index(indices), dtype_, display_max_ele_width_);
                        if (i < size - 1) os << ",";
                    }
                }
                os << "]";
            }
            else {
                os << "[";
                const size_t size = shape[dim];
                if (size > 10) {
                    // 打印前8个元素
                    for (size_t i = 0; i < 8; ++i) {
                        indices[dim] = i;
                        print_helper(data, shape, indices, dim + 1);
                        os << "," << std::endl;
                        for (size_t j = 0; j <= dim; ++j) os << " ";
                    }
                    os << "[";
                    for (size_t i = 0; i < 10; i++) {
                        os << std::setw(display_max_ele_width_) << "...";
                        if (i < 9) os << ",";
                    }
                    os << "],\n";
                    // 打印最后一个元素
                    indices[dim] = size - 1;
                    for (size_t j = 0; j <= dim; ++j) os << " ";
                    print_helper(data, shape, indices, dim + 1);
                }
                else {
                    // 打印所有元素
                    for (size_t i = 0; i < size; ++i) {
                        indices[dim] = i;
                        print_helper(data, shape, indices, dim + 1);
                        if (i < size - 1) {
                            os << "," << std::endl;
                            for (size_t j = 0; j <= dim; ++j) os << " ";
                        }
                    }
                }
                os << "]";
            }
        };
        std::vector<size_t> indices(shape_.size(), 0);
        print_helper(data_ptr_, shape_, indices, 0);
        os << std::endl;
    }


    std::string Tensor::to_string() const {
        std::ostringstream oss;
        print(oss);
        return oss.str();
    }


    Tensor Tensor::concat(const std::vector<Tensor>& tensors, const int axis) {
        if (tensors.empty()) {
            throw std::runtime_error("连接的张量列表为空");
        }

        // 验证所有tensor的维度（除了连接轴）都相同
        const auto& first_shape = tensors[0].shape();
        int64_t concat_size = 0;

        for (const auto& tensor : tensors) {
            if (tensor.dtype() != tensors[0].dtype()) {
                throw std::runtime_error("所有张量必须具有相同的数据类型");
            }
            const auto& current_shape = tensor.shape();
            if (current_shape.size() != first_shape.size()) {
                throw std::runtime_error("所有张量必须具有相同的维度数");
            }
            for (size_t i = 0; i < first_shape.size(); ++i) {
                if (i != static_cast<size_t>(axis) && current_shape[i] != first_shape[i]) {
                    throw std::runtime_error("除了连接轴外，所有维度必须相同");
                }
            }
            concat_size += current_shape[axis];
        }

        // 计算新的形状
        std::vector<int64_t> new_shape = first_shape;
        new_shape[axis] = concat_size;
        // 创建结果张量
        Tensor result(new_shape, tensors[0].dtype(), "concat_result");
        // 计算每个张量的元素大小
        const size_t element_size = tensors[0].element_size_;
        // 计算轴步长
        size_t axis_stride = 1;
        for (size_t i = axis + 1; i < first_shape.size(); ++i) {
            axis_stride *= first_shape[i];
        }
        // 计算每个切片的大小（字节）
        const size_t slice_size = axis_stride * element_size;
        // 计算前面维度的总迭代次数
        size_t outer_iterations = 1;
        for (size_t i = 0; i < axis; ++i) {
            outer_iterations *= first_shape[i];
        }
        // 复制数据
        auto dest_ptr = static_cast<char*>(result.data());

        for (size_t i = 0; i < outer_iterations; ++i) {
            for (const auto& tensor : tensors) {
                const char* src_ptr = static_cast<const char*>(tensor.data()) + i * tensor.shape()[axis] * slice_size;
                const size_t copy_size = tensor.shape()[axis] * slice_size;
                std::memcpy(dest_ptr, src_ptr, copy_size);
                dest_ptr += copy_size;
            }
        }
        return result;
    }

    Tensor Tensor::softmax(int axis) const {
        // 确保数据类型是浮点类型
        if (dtype_ != DataType::FP32) {
            throw std::runtime_error("Softmax操作只支持FP32类型");
        }
        // 处理负轴，转换为正轴
        if (axis < 0) {
            axis += static_cast<int>(shape_.size());
        }
        // 检查轴是否有效
        if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
            throw std::runtime_error("Softmax操作的轴超出范围");
        }
        // 创建结果张量
        Tensor result(shape_, dtype_, name_ + "_softmax");
        // 获取数据指针
        const auto* src_data = static_cast<const float*>(data_ptr_);
        auto* dest_data = static_cast<float*>(result.data());
        // 计算指定轴的维度大小
        const size_t axis_dim = shape_[axis];
        // 计算在内存中相邻元素在这个轴上的步长
        const size_t axis_stride = strides_[axis];
        // 计算这个轴的切片数
        const size_t num_slices = size() / axis_dim;
        // 创建临时缓冲区
        std::vector<float> buffer(axis_dim);
        // 对每个切片应用softmax
        for (size_t slice = 0; slice < num_slices; ++slice) {
            // 计算当前切片的起始索引
            const size_t start_idx = slice / (num_slices / outer_dim(axis)) * strides_[axis - 1] +
                slice % (num_slices / outer_dim(axis));
            // 复制数据到缓冲区
            for (size_t i = 0; i < axis_dim; ++i) {
                buffer[i] = src_data[start_idx + i * axis_stride];
            }
            // 寻找最大值以提高数值稳定性
            const float max_val = *std::max_element(buffer.begin(), buffer.end());
            // 计算指数和
            float sum_exp = 0.0f;
            for (float& val : buffer) {
                val = std::exp(val - max_val);
                sum_exp += val;
            }
            // 归一化
            for (float& val : buffer) {
                val /= sum_exp;
            }
            // 将结果复制回去
            for (size_t i = 0; i < axis_dim; ++i) {
                dest_data[start_idx + i * axis_stride] = buffer[i];
            }
        }
        return result;
    }

    void Tensor::expand_dim(int64_t axis) {
        if (axis < 0) {
            axis += static_cast<int>(shape_.size()) + 1;
        }
        if (axis < 0 || axis > static_cast<int64_t>(shape_.size())) {
            throw std::runtime_error("扩展维度的轴超出范围");
        }
        shape_.insert(shape_.begin() + axis, 1);
        calculate_strides();
    }

    bool Tensor::is_same_shape(const Tensor& other) const {
        return shape_ == other.shape_;
    }

    size_t Tensor::get_dim_size(const size_t dim) const {
        if (dim >= shape_.size()) {
            throw std::runtime_error("维度索引超出范围");
        }
        return shape_[dim];
    }

    size_t Tensor::get_rank() const {
        return shape_.size();
    }

    bool Tensor::is_empty() const {
        return memory_ == nullptr || data_ptr_ == nullptr;
    }

    size_t Tensor::compute_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Index dimension mismatch");
        }

        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of range");
            }
            index += indices[i] * strides_[i];
        }
        return index;
    }

    void Tensor::validate_shape(const std::vector<int64_t>& shape) {
        if (shape.empty()) {
            throw std::runtime_error("不允许空形状");
        }

        if (std::any_of(shape.begin(), shape.end(), [](const int64_t dim) { return dim <= 0; })) {
            throw std::runtime_error("所有维度必须为正数");
        }
    }

    size_t Tensor::calculate_total_size() const {
        return std::accumulate(shape_.begin(), shape_.end(),
                               1LL, std::multiplies<>()) * element_size_;
    }

    // 计算轴之前的维度乘积
    size_t Tensor::outer_dim(const int axis) const {
        size_t result = 1;
        for (int i = 0; i < axis; ++i) {
            result *= shape_[i];
        }
        return result;
    }


    // TensorView实现
    TensorView::TensorView(const Tensor& tensor)
        : base_tensor_(std::make_shared<const Tensor>(tensor)),
          shape_(tensor.shape_),
          strides_(tensor.strides_),
          data_ptr_(tensor.data_ptr_) {
    }

    TensorView::TensorView(const Tensor& tensor,
                           const std::vector<int64_t>& shape,
                           const std::vector<int64_t>& strides,
                           void* data_ptr)
        : base_tensor_(std::make_shared<const Tensor>(tensor)),
          shape_(shape),
          strides_(strides),
          data_ptr_(data_ptr) {
    }

    size_t TensorView::size() const {
        return std::accumulate(shape_.begin(), shape_.end(),
                               1LL, std::multiplies<>());
    }

    size_t TensorView::byte_size() const {
        return size() * get_element_size();
    }

    size_t TensorView::get_element_size() const {
        return base_tensor_->element_size_;
    }

    Tensor TensorView::to_tensor() const {
        // 创建新的张量，复制视图中的数据
        Tensor result(shape_, base_tensor_->dtype(), base_tensor_->name() + "_from_view");
        // 如果数据在内存中是连续的，可以一次性复制
        if (is_contiguous()) {
            std::memcpy(result.data(), data_ptr_, byte_size());
            return result;
        }
        // 否则需要逐元素复制
        const size_t element_size = get_element_size();
        std::vector<int64_t> indices(shape_.size(), 0);
        std::function<void(size_t)> copy_recursively = [&](const size_t dim) {
            if (dim == shape_.size()) {
                // 计算源偏移量
                size_t src_offset = 0;
                for (size_t i = 0; i < indices.size(); ++i) {
                    src_offset += indices[i] * strides_[i];
                }
                src_offset *= element_size;
                // 计算目标偏移量
                size_t dest_offset = 0;
                size_t dest_stride = 1;
                for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
                    dest_offset += indices[i] * dest_stride;
                    dest_stride *= shape_[i];
                }
                dest_offset *= element_size;
                // 复制单个元素
                std::memcpy(
                    static_cast<char*>(result.data()) + dest_offset,
                    static_cast<char*>(data_ptr_) + src_offset,
                    element_size
                );
                return;
            }
            for (indices[dim] = 0; indices[dim] < shape_[dim]; ++indices[dim]) {
                copy_recursively(dim + 1);
            }
        };
        copy_recursively(0);
        return result;
    }

    TensorView TensorView::reshape(const std::vector<int64_t>& new_shape) const {
        // 验证元素数量是否匹配
        const size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                                1LL, std::multiplies<>());
        if (new_size != size()) {
            throw std::runtime_error("无法重塑：元素数量不匹配");
        }
        // 如果当前视图是连续的，可以直接创建新形状的视图
        if (is_contiguous()) {
            std::vector<int64_t> new_strides(new_shape.size());
            if (!new_strides.empty()) {
                new_strides.back() = 1;
                for (int i = static_cast<int>(new_shape.size()) - 2; i >= 0; --i) {
                    new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
                }
            }
            return TensorView{*base_tensor_, new_shape, new_strides, data_ptr_};
        }
        // 如果不连续，需要先物化然后再创建视图
        return to_tensor().view().reshape(new_shape);
    }

    // TensorView::transpose方法的实现
    TensorView TensorView::transpose(const std::vector<int64_t>& axes) const {
        if (axes.size() != shape_.size()) {
            throw std::runtime_error("转置的轴数与视图维度不匹配");
        }
        // 检查轴是否有效
        std::vector used(shape_.size(), false);
        for (auto& axe : axes) {
            if (axe < 0 || axe >= static_cast<int64_t>(shape_.size())) {
                throw std::runtime_error("转置轴超出范围");
            }
            if (used[axe]) {
                throw std::runtime_error("转置轴重复");
            }
            used[axe] = true;
        }
        // 创建新的形状和步长
        std::vector<int64_t> new_shape(shape_.size());
        std::vector<int64_t> new_strides(shape_.size());
        for (size_t i = 0; i < axes.size(); ++i) {
            new_shape[i] = shape_[axes[i]];
            new_strides[i] = strides_[axes[i]];
        }
        return TensorView{*base_tensor_, new_shape, new_strides, data_ptr_};
    }

    TensorView TensorView::slice(const std::vector<int64_t>& starts, const std::vector<int64_t>& ends) const {
        if (starts.size() != shape_.size() || ends.size() != shape_.size()) {
            throw std::runtime_error("切片的起始和结束索引必须与视图维度匹配");
        }
        // 验证切片范围
        std::vector<int64_t> new_shape(shape_.size());
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (starts[i] < 0 || starts[i] >= shape_[i] || ends[i] > shape_[i] || starts[i] >= ends[i]) {
                throw std::runtime_error("无效的切片范围");
            }
            new_shape[i] = ends[i] - starts[i];
        }
        // 计算新的起始数据指针
        size_t offset = 0;
        for (size_t i = 0; i < shape_.size(); ++i) {
            offset += starts[i] * strides_[i];
        }
        char* new_data_ptr = static_cast<char*>(data_ptr_) + offset * get_element_size();

        return TensorView{*base_tensor_, new_shape, strides_, static_cast<void*>(new_data_ptr)};
    }

    bool TensorView::is_contiguous() const {
        // 检查步长是否表示连续内存
        int64_t expected_stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            if (strides_[i] != expected_stride) {
                return false;
            }
            expected_stride *= shape_[i];
        }
        return true;
    }

    template <typename T>
    T& TensorView::at(const std::vector<int64_t>& indices) {
        if (indices.size() != shape_.size()) {
            throw std::runtime_error("索引数量与视图维度不匹配");
        }
        // 验证索引范围
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || indices[i] >= shape_[i]) {
                throw std::runtime_error("索引超出范围");
            }
        }
        // 计算偏移量
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * strides_[i];
        }
        return *static_cast<T*>(static_cast<void*>(static_cast<char*>(data_ptr_) + offset * get_element_size()));
    }

    template <typename T>
    const T& TensorView::at(const std::vector<int64_t>& indices) const {
        return const_cast<TensorView*>(this)->at<T>(indices);
    }

    // 不进行模板特化那么模板的定义必须在头文件中
    // 显式实例化TensorView的常用类型
    template float& TensorView::at<float>(const std::vector<int64_t>&);
    template int32_t& TensorView::at<int32_t>(const std::vector<int64_t>&);
    template int64_t& TensorView::at<int64_t>(const std::vector<int64_t>&);
    template uint8_t& TensorView::at<uint8_t>(const std::vector<int64_t>&);
    template int8_t& TensorView::at<int8_t>(const std::vector<int64_t>&);

    template const float& TensorView::at<float>(const std::vector<int64_t>&) const;
    template const int32_t& TensorView::at<int32_t>(const std::vector<int64_t>&) const;
    template const int64_t& TensorView::at<int64_t>(const std::vector<int64_t>&) const;
    template const uint8_t& TensorView::at<uint8_t>(const std::vector<int64_t>&) const;
    template const int8_t& TensorView::at<int8_t>(const std::vector<int64_t>&) const;

    // 这里添加了Tensor类中的数据操作接口的显式实例化
    template void Tensor::set_data<double>(const double*, size_t);
    template void Tensor::set_data<double>(std::vector<double>&&);
    template const double* Tensor::data_ptr<double>() const;
    template std::vector<double> Tensor::get_data<double>() const;

    // 显式实例化TensorView的常用类型
    template float& Tensor::at<float>(const std::vector<int64_t>&);
    template int32_t& Tensor::at<int32_t>(const std::vector<int64_t>&);
    template int64_t& Tensor::at<int64_t>(const std::vector<int64_t>&);
    template uint8_t& Tensor::at<uint8_t>(const std::vector<int64_t>&);
    template int8_t& Tensor::at<int8_t>(const std::vector<int64_t>&);

    template const float& Tensor::at<float>(const std::vector<int64_t>&) const;
    template const int32_t& Tensor::at<int32_t>(const std::vector<int64_t>&) const;
    template const int64_t& Tensor::at<int64_t>(const std::vector<int64_t>&) const;
    template const uint8_t& Tensor::at<uint8_t>(const std::vector<int64_t>&) const;
    template const int8_t& Tensor::at<int8_t>(const std::vector<int64_t>&) const;
} // namespace modeldeploy
