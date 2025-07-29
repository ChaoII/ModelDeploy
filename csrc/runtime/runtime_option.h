//
// Created by aichao on 2025/5/22.
//

#pragma once

#include "backends/trt/option.h"
#include "runtime/backends/mnn/option.h"
#include "runtime/backends/ort/option.h"
#include "runtime/enum_variables.h"


namespace modeldeploy {
    struct MODELDEPLOY_CXX_EXPORT RuntimeOption {
        void set_model_path(const std::string& model_path, const std::string& password = "");

        void use_cpu();

        void use_gpu(int gpu_id = 0);

        void use_opencl(int device_id = 0);

        void set_external_stream(void* external_stream);

        void set_cpu_thread_num(int thread_num);

        void use_ort_backend();

        void use_mnn_backend();

        void use_trt_backend();

        //images:1x3x224x224
        void set_trt_min_shape(const std::string& trt_min_shape);
        //images:4x3x640x640
        void set_trt_opt_shape(const std::string& trt_opt_shape);
        //images:8x3x1280x1280
        void set_trt_max_shape(const std::string& trt_max_shape);

        OrtBackendOption ort_option;
        MnnBackendOption mnn_option;
        TrtBackendOption trt_option;
        std::string password;
        bool enable_fp16 = false;
        bool enable_trt = false;
        bool model_from_memory = false;
        int cpu_thread_num = -1;
        int device_id = -1;
        std::string model_buffer;
        Device device = Device::CPU;
        std::string model_file;
        Backend backend = Backend::ORT;
    };
}
