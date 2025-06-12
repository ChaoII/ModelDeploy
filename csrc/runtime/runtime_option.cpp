//
// Created by aichao on 2025/5/22.
//

#include "csrc/runtime/runtime_option.h"
#include "csrc/utils/utils.h"


namespace modeldeploy {
    void RuntimeOption::set_model_path(const std::string& model_path) {
        model_file = model_path;
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

    void RuntimeOption::set_ort_graph_opt_level(const int level) {
        const std::vector supported_level{-1, 0, 1, 2};
        if (std::ranges::find(supported_level, level) == supported_level.end()) {
            MD_LOG_ERROR << "Invalid graph optimization level: " << level << ", supported levels are: "
                << vector_to_string(supported_level) << std::endl;
        }
        ort_option.graph_optimization_level = level;
    }

    //input:8x3x224x224
    void RuntimeOption::set_trt_min_shape(const std::string& trt_min_shape) {
        ort_option.trt_min_shape = trt_min_shape;
    }

    //input:8x3x224x224
    void RuntimeOption::set_trt_opt_shape(const std::string& trt_opt_shape) {
        ort_option.trt_opt_shape = trt_opt_shape;
    }

    //input:8x3x224x224
    void RuntimeOption::set_trt_max_shape(const std::string& trt_max_shape) {
        ort_option.trt_max_shape = trt_max_shape;
    }
}
