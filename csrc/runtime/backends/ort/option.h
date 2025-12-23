//
// Created by aichao on 2025/2/20.
//

#pragma once
#include <string>
#include "core/md_decl.h"
#include "core/enum_variables.h"

namespace modeldeploy {
    /*! @brief Option object to configure ONNX Runtime backend
     */
    struct MODELDEPLOY_CXX_EXPORT OrtBackendOption {
        bool model_from_memory = false;

        /// Level of graph optimization,
        ///         /-1: mean default(Enable all the optimization strategy)
        ///         /0: disable all the optimization strategy/1: enable basic strategy
        ///         /2:enable extend strategy/99: enable all
        int graph_optimization_level = -1;
        /// Number of threads to execute the operator, -1: default
        int intra_op_num_threads = -1;
        /// Number of threads to execute the graph,
        ///         -1: default. This parameter only will bring effects
        ///         while the `OrtBackendOption::execution_mode` set to 1.
        int inter_op_num_threads = -1;
        /// Execution mode for the graph, -1: default(Sequential mode)
        ///         /0: Sequential mode, execute the operators in graph one by one.
        ///         /1: Parallel mode, execute the operators in graph parallely.
        int execution_mode = -1;
        /// Inference device, OrtBackend supports CPU/GPU
        Device device = Device::CPU;
        /// Inference device id
        int device_id = 0;

        void* external_stream = nullptr;
        /// Use fp16 to infer
        bool enable_fp16 = false;
        /// file path for optimized model
        std::string optimized_model_filepath;

        std::string trt_engine_cache_path = "./";

        std::string model_buffer;

        std::string model_filepath;

        bool enable_trt = true;

        // 动态 shape 支持（如不需要可以留空）
        std::string trt_min_shape;

        std::string trt_opt_shape;

        std::string trt_max_shape;

        void set_cpu_thread_num(const int num) {
            intra_op_num_threads = num;
            inter_op_num_threads = num;
        }
    };
}
