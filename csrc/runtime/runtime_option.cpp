//
// Created by aichao on 2025/5/22.
//

#include <algorithm>
#include <filesystem>
#include "utils/utils.h"
#include "runtime/runtime_option.h"
#include "encryption/encryption.h"


namespace modeldeploy {
    // 各后端支持的模型格式
    static bool backend_supports_format(Backend backend, const std::string& ext) {
        switch (backend) {
            case Backend::ORT: return ext == ".onnx";
            case Backend::MNN: return ext == ".mnn";
            case Backend::TRT: return ext == ".onnx" || ext == ".engine";
            default: return false;
        }
    }

    void RuntimeOption::set_model_path(const std::string& model_path, const std::string& password) {
        model_file = model_path;

        // 解密加密模型（如适用）
        if (is_encrypted_model_file(model_path)) {
            std::string buffer, format;
            auto decrypt_password = password;
            if (password.empty()) {
                decrypt_password = this->password;
            }
            if (!read_encrypted_model_to_buffer(model_path, decrypt_password, &buffer, &format)) {
                MD_LOG_FATAL << "Model decryption failed. Check password (set via option.password) "
                    << "or that the file is not corrupted." << std::endl;
            }
            // 同步到三个后端 option（各后端从各自的 option 或 RuntimeOption 读取）
            model_from_memory = true;
            model_buffer = buffer;
            ort_option.model_from_memory = true;
            ort_option.model_buffer = buffer;
            trt_option.model_from_memory = true;
            trt_option.model_buffer = buffer;
            mnn_option.model_from_memory = true;
            mnn_option.model_buffer = buffer;
            return;
        }

        // 非加密模型
        const std::filesystem::path path(model_path);
        model_from_memory = false;

        if (!password.empty() || !this->password.empty()) {
            MD_LOG_WARN << "Password provided but model file is not encrypted: "
                << model_path << ". The password will be ignored." << std::endl;
        }

        if (!path.has_extension()) return;

        // 校验当前 backend 是否支持该格式；仅在默认 ORT 且不匹配时自动推断
        const std::string ext = path.extension().string();
        if (backend_supports_format(backend, ext)) return;

        if (backend == Backend::ORT) {
            if (ext == ".mnn") {
#ifdef ENABLE_MNN
                backend = Backend::MNN;
#endif
            }
            else if (ext == ".engine") {
#ifdef ENABLE_TRT
                backend = Backend::TRT;
#endif
            }
            // .onnx → ORT（已经是默认，不需要切换）
        }
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
        ort_option.external_stream = external_stream;
        trt_option.external_stream = external_stream;
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

    // for onnxruntime-trt
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
