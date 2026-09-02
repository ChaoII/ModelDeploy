//
// Created by aichao on 2025/5/22.
//

#pragma once

#include "backends/trt/option.h"
#include "runtime/backends/mnn/option.h"
#include "runtime/backends/ort/option.h"
#include "runtime/backends/sophgo/option.h"
#include "runtime/backends/ncnn/option.h"
#include "core/enum_variables.h"


namespace modeldeploy {
    struct MODELDEPLOY_CXX_EXPORT RuntimeOption {
        void set_model_path(const std::string& model_path, const std::string& password = "");

        // 格式 → 唯一后端映射（capi md_option_set_model_buffer 用）
        static Backend backend_for_format(const std::string& fmt);

        // 设备：唯一入口，device + id 一步设置
        void set_device(Device dev, int device_id = 0);
        // 以下为废弃转发（兼容旧代码），新代码请用 set_device
        [[deprecated("use set_device()")]] void use_cpu();
        [[deprecated("use set_device(Device::GPU, id)")]] void use_gpu(int gpu_id = 0);
        [[deprecated("use set_device(Device::OPENCL, id)")]] void use_opencl(int device_id = 0);
        void set_password(const std::string& pwd) { password = pwd; }

        void set_external_stream(void* external_stream);
        void set_cpu_thread_num(int thread_num);

        void use_ort_backend();
        void use_mnn_backend();
        void use_trt_backend();
        void use_sophgo_backend();          // 无参，隐含 device=TPU
        void use_ncnn_backend();
        void validate();                     // 配置期校验 + 快照到激活后端

        void set_trt_min_shape(const std::string&);
        void set_trt_opt_shape(const std::string&);
        void set_trt_max_shape(const std::string&);

        OrtBackendOption ort_option;
        MnnBackendOption mnn_option;
        TrtBackendOption trt_option;
        SophgoBackendOption sophgo_option;
        NcnnBackendOption ncnn_option;
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
