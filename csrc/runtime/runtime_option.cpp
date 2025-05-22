//
// Created by aichao on 2025/5/22.
//

#include "csrc/runtime/runtime_option.h"
#include "csrc/utils/utils.h"


namespace modeldeploy {
    void RuntimeOption::set_model_path(const std::string& model_path) {
        model_filepath = model_path;
    }


    void RuntimeOption::use_cpu() { device = Device::CPU; }


    void RuntimeOption::use_gpu(const int gpu_id) {
        device = Device::GPU;
        device_id = gpu_id;
    }


    void RuntimeOption::set_external_stream(void* external_stream) {
        ort_option.external_stream_ = external_stream;
    }

    void RuntimeOption::set_cpu_thread_num(int thread_num) {
        ort_option.set_cpu_thread_num(thread_num);
    }

    void RuntimeOption::use_ort_backend() {
        backend = Backend::ORT;
    }

    void RuntimeOption::set_ort_graph_opt_level(int level) {
        std::vector<int> supported_level{-1, 0, 1, 2};
        ort_option.graph_optimization_level = level;
    }
}
