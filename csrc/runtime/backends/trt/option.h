//
// Created by aichao on 2025/6/27.
//

#pragma once
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace modeldeploy {
    /*! @brief Option object to configure TensorRT backend
     */
    struct TrtBackendOption {
        /// `max_batch_size`, it's deprecated in TensorRT 8.x
        size_t max_batch_size = 32;

        /// `max_workspace_size` for TensorRT
        size_t max_workspace_size = 1 << 30;

        /// Enable log while converting onnx model to tensorrt
        bool enable_log_info = false;


        /// Enable half precison inference, on some device not support half precision,
        /// it will fallback to float32 mode
        bool enable_fp16 = false;

        /** \brief Set shape range of input tensor for the model that contain dynamic input shape while using TensorRT backend
         *
         * \param[in] tensor_name The name of input for the model which is dynamic shape
         * \param[in] min The minimal shape for the input tensor
         * \param[in] opt The optimized shape for the input tensor, just set the most common shape, if set as default value, it will keep same with min_shape
         * \param[in] max The maximum shape for the input tensor, if set as default value, it will keep same with min_shape
         */
        void SetShape(const std::string& tensor_name,
                      const std::vector<int32_t>& min,
                      const std::vector<int32_t>& opt = std::vector<int32_t>(),
                      const std::vector<int32_t>& max = std::vector<int32_t>()) {
            if (opt.empty()) {
                opt_shape[tensor_name] = min;
            }
            else {
                opt_shape[tensor_name] = opt;
            }
            if (max.empty()) {
                max_shape[tensor_name] = opt_shape[tensor_name];
            }
            else {
                max_shape[tensor_name] = max;
            }
        }

        /// Set cache file path while use TensorRT backend.
        /// take a long time,
        /// by this interface it will save the tensorrt engine to `cache_file_path`,
        /// and load it directly while execute the code again
        std::string cache_file_path;

        std::map<std::string, std::vector<int32_t>> max_shape;
        std::map<std::string, std::vector<int32_t>> min_shape;
        std::map<std::string, std::vector<int32_t>> opt_shape;
        bool enable_pinned_memory = false;
        void* external_stream_ = nullptr;
        int gpu_id = 0;
        std::string model_file;
    };
} // namespace fastdeploy
