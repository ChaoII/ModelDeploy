//
// Created by aichao on 2025/5/22.
//

#include <algorithm>
#include <filesystem>
#include "utils/utils.h"
#include "runtime/runtime_option.h"
#include "encryption/encryption.h"


namespace modeldeploy {
    void RuntimeOption::set_model_path(const std::string& model_path, const std::string& password) {
        model_file = model_path;
        if (is_encrypted_model_file(model_path)) {
            std::string buffer, format;
            auto decrypt_password = password;
            if (password.empty()) {
                decrypt_password = this->password;
            }
            if (!read_encrypted_model_to_buffer(model_path, decrypt_password, &buffer, &format)) {
                MD_LOG_FATAL << "decrypt model failed" << std::endl;
            }
            model_from_memory = true;
            model_buffer = buffer;
            // 你可以根据format自动切换后端
            if (format == "onnx") {
                use_ort_backend();
            }
            else if (format == "mnn") {
                use_mnn_backend();
            }
            else if (format == "engine") {
                use_trt_backend();
            }
            else {
                MD_LOG_FATAL << "model format error" << std::endl;
            }
        }
        else {
            const std::filesystem::path path(model_path);
            if (path.has_extension()) {
                if (path.extension() == ".onnx") {
                    use_ort_backend();
                }
                else if (path.extension() == ".mnn") {
                    use_mnn_backend();
                }
                else if (path.extension() == ".engine") {
                    use_trt_backend();
                }
                else {
                    MD_LOG_FATAL << "model format error" << std::endl;
                }
            }
            model_from_memory = false;
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
        const std::vector<int> supported_level{-1, 0, 1, 2};
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
