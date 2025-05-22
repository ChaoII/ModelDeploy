//
// Created by aichao on 2025/5/22.
//

#pragma once

#include "csrc/runtime/backends/ort/option.h"
#include "csrc/runtime/enum_variables.h"


namespace modeldeploy {

    struct RuntimeOption {

        void set_model_path(const std::string& model_path);

        void use_cpu();

        void use_gpu(int gpu_id = 0);

        void set_external_stream(void* external_stream);

        void set_cpu_thread_num(int thread_num);

        void use_ort_backend();

        OrtBackendOption ort_option;

        int cpu_thread_num = -1;
        int device_id = 0;
        bool model_from_memory = false;
        std::string model_buffer;
        Device device = Device::CPU;
        std::string model_filepath;
        Backend backend = Backend::ORT;
        void set_ort_graph_opt_level(int level = -1);
    };
}
