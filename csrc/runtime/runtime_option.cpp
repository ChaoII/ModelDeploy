//
// Created by aichao on 2025/5/22.
//

#include <algorithm>
#include "runtime/runtime_option.h"
#include "utils/utils.h"


namespace modeldeploy {
    void RuntimeOption::set_model_path(const std::string& model_path) {
        model_file = model_path;
    }


    void RuntimeOption::use_cpu() { device = Device::CPU; }


    void RuntimeOption::use_gpu(const int gpu_id) {
        device = Device::GPU;
        device_id = gpu_id;
    }

    void RuntimeOption::use_opencl(const int device_id) {
        device = Device::OPENCL;
        this->device_id = device_id;
    }


    void RuntimeOption::set_external_stream(void* external_stream) {
        ort_option.external_stream_ = external_stream;
    }

    void RuntimeOption::set_cpu_thread_num(const int thread_num) {
        ort_option.set_cpu_thread_num(thread_num);
        mnn_option.cpu_thread_num = thread_num;
    }

    void RuntimeOption::use_ort_backend() {
#ifdef ENABLE_ORT
        backend = Backend::ORT;
#else
        MD_LOG_FATAL << "The ModelDeploy didn't compile with OnnxRuntime backend." << std::endl;
#endif
    }

    void RuntimeOption::use_mnn_backend() {
#ifdef ENABLE_MNN
        backend = Backend::MNN;
#else
        MD_LOG_FATAL << "The ModelDeploy didn't compile with MNN backend." << std::endl;
#endif
    }

    void RuntimeOption::use_trt_backend() {
#ifdef ENABLE_TRT
        backend = Backend::TRT;
#else
        MD_LOG_FATAL << "The ModelDeploy didn't compile with TRT backend." << std::endl;
#endif
    }

    void RuntimeOption::set_ort_graph_opt_level(const int level) {
        const std::vector supported_level{-1, 0, 1, 2};
        if (std::find(supported_level.begin(), supported_level.end(), level) == supported_level.end()) {
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
