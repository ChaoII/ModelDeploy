//
// Created by aichao on 2025/2/20.
//
#pragma once

#include <string>
#include <vector>

#include "csrc/core/md_decl.h"
#include "csrc/core/md_scalar.h"
#include "csrc/core/md_type.h"


namespace modeldeploy {
    struct MODELDEPLOY_CXX_EXPORT MDTensor {
        void set_data(const std::vector<int64_t>& tensor_shape,
                      const MDDataType::Type& data_type,
                      void* data_buffer, bool copy = false) {
            set_external_data(tensor_shape, data_type, data_buffer);
            if (copy) {
                stop_sharing();
            }
        }

        /// Get data pointer of tensor
        void* get_data() {
            return mutable_data();
        }

        /// Get data pointer of tensor
        const void* get_data() const {
            return data();
        }

        /// Expand the shape of tensor, it will not change the data memory, just modify its attribute `shape`
        void expand_dim(int64_t axis = 0);

        /// Squeeze the shape of tensor, it will not change the data memory, just modify its attribute `shape`
        void squeeze(int64_t axis = 0);

        /// Reshape the tensor, it will not change the data memory, just modify its attribute `shape`
        bool reshape(const std::vector<int64_t>& new_shape);

        /// Total size of tensor memory buffer in bytes
        int total_bytes() const;

        /// Total number of elements in tensor
        int total() const;

        /// Get shape of tensor
        std::vector<int64_t> get_shape() const { return shape; }

        /// Get dtype of tensor
        MDDataType::Type get_dtype() const { return dtype; }

        /** \brief Allocate cpu data buffer for a MDTensor, e.g
         *  ```
         *  MDTensor tensor;
         *  tensor.Allocate(FDDataType::FLOAT, {1, 3, 224, 224};
         *  ```
         * \param[in] data_type The data type of tensor
         * \param[in] tensor_shape The shape of tensor
         */
        void allocate(const MDDataType::Type& data_type, const std::vector<int64_t>& tensor_shape) {
            allocate(tensor_shape, data_type, name);
        }


        /// Name of tensor, while feed to runtime, this need be defined
        std::string name;

        /// Whether the tensor is owned the data buffer or share the data buffer from outside
        [[nodiscard]] bool is_shared() const { return external_data_ptr_ != nullptr; }
        /// If the tensor is share the data buffer from outside, `StopSharing` will copy to its own structure; Otherwise, do nothing
        void stop_sharing();


        // ******************************************************
        // The following member and function only used by inside FastDeploy, maybe removed in next version

        void* buffer_ = nullptr;
        std::vector<int64_t> shape = {0};
        MDDataType::Type dtype = MDDataType::Type::INT8;

        // This use to skip memory copy step
        // the external_data_ptr will point to the user allocated memory
        // user has to maintain the memory, allocate and release
        void* external_data_ptr_ = nullptr;


        // if the external data is not on CPU, we use this temporary buffer
        // to transfer data to CPU at some cases we need to visit the
        // other devices' data
        std::vector<int8_t> temporary_cpu_buffer_;

        // The number of bytes allocated so far.
        // When resizing GPU memory, we will free and realloc the memory only if the
        // required size is larger than this value.
        size_t total_bytes_allocated_ = 0;

        // Get data buffer pointer
        void* mutable_data();

        void* data();

        const void* data() const;

        // void SetDataBuffer(const std::vector<int64_t>& new_shape, const FDDataType& data_type, void* data_buffer, bool copy = false, const Device& new_device = Device::CPU, int new_device_id = -1);
        // Set user memory buffer for Tensor, the memory is managed by
        // the user it self, but the Tensor will share the memory with user
        // So take care with the user buffer
        void set_external_data(const std::vector<int64_t>& new_shape,
                               const MDDataType::Type& data_type, void* data_buffer);
        // Initialize Tensor
        // Include setting attribute for tensor
        // and allocate cpu memory buffer
        void allocate(const std::vector<int64_t>& new_shape,
                      const MDDataType::Type& data_type,
                      const std::string& tensor_name = "");

        void resize(size_t total_bytes);

        void resize(const std::vector<int64_t>& new_shape);

        void resize(const std::vector<int64_t>& new_shape,
                    const MDDataType::Type& data_type, const std::string& tensor_name = "");

        bool re_alloc_fn(size_t total_bytes);

        void free_fn();

        MDTensor() {
        }

        explicit MDTensor(const std::string& tensor_name);
        explicit MDTensor(const char* tensor_name);

        // Deep copy
        MDTensor(const MDTensor& other);
        // Move constructor
        MDTensor(MDTensor&& other) noexcept ;

        // Deep copy assignment
        MDTensor& operator=(const MDTensor& other);
        // Move assignment
        MDTensor& operator=(MDTensor&& other) noexcept ;

        // Scalar to MDTensor
        explicit MDTensor(const Scalar& scalar);

        ~MDTensor() { free_fn(); }

        static void copy_buffer(void* dst, const void* src, size_t num_bytes);
    };
}
