//
// Created by aichao on 2025/5/22.
//

#pragma once

#include "csrc/runtime/backends/mnn/option.h"
#include "csrc/runtime/backends/ort/option.h"
#include "csrc/runtime/enum_variables.h"


namespace modeldeploy {
    struct MODELDEPLOY_CXX_EXPORT RuntimeOption {
        void set_model_path(const std::string& model_path);

        void use_cpu();

        void use_gpu(int gpu_id = 0);

        void set_external_stream(void* external_stream);

        void set_cpu_thread_num(int thread_num);

        void use_ort_backend();

        void use_mnn_backend();

        void set_trt_min_shape(const std::string& trt_min_shape);

        void set_trt_opt_shape(const std::string& trt_opt_shape);

        void set_trt_max_shape(const std::string& trt_max_shape);

        OrtBackendOption ort_option;

        MnnBackendOption mnn_option;

        bool enable_fp16 = false;
        int cpu_thread_num = -1;
        int device_id = 0;
        bool enable_trt = false;
        bool model_from_memory = false;
        std::string model_buffer;
        Device device = Device::CPU;
        std::string model_file;
        Backend backend = Backend::ORT;
        void set_ort_graph_opt_level(int level = -1);
    };
}
