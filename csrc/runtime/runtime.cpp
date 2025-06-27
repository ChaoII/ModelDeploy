//
// Created by aichao on 2025/5/22.
//


#include "csrc/runtime/runtime.h"
#ifdef ENABLE_TRT
#include "backends/trt/trt.h"
#endif

#ifdef ENABLE_ORT
#include "csrc/runtime/backends/ort/ort.h"
#endif
#ifdef ENABLE_MNN
#include "csrc/runtime/backends/mnn/mnn_module_api.h"
#endif


namespace modeldeploy {
    bool Runtime::init(const RuntimeOption& _option) {
        option = _option;

        if (option.backend == Backend::ORT) {
            create_ort_backend();
        }
        else if (option.backend == Backend::MNN) {
            create_mnn_backend();
        }
        else if (option.backend == Backend::TRT) {
            create_trt_backend();
        }
        else {
            MD_LOG_ERROR << "The " << option.backend << " backend is not supported now." << std::endl;
            return false;
        }
        return true;
    }

    TensorInfo Runtime::get_input_info(const int index) const {
        return backend_->get_input_info(index);
    }

    TensorInfo Runtime::get_output_info(const int index) const {
        return backend_->get_output_info(index);
    }

    std::vector<TensorInfo> Runtime::get_input_infos() const {
        return backend_->get_input_infos();
    }

    std::vector<TensorInfo> Runtime::get_output_infos() const {
        return backend_->get_output_infos();
    }

    bool Runtime::infer(std::vector<Tensor>& input_tensors,
                        std::vector<Tensor>* output_tensors) const {
        return backend_->infer(input_tensors, output_tensors);
    }

    bool Runtime::infer() {
        bool result = false;
        result = backend_->infer(input_tensors_, &output_tensors_);
        return result;
    }

    Runtime* Runtime::clone(void* stream, const int device_id) {
        auto* runtime = new Runtime();
        MD_LOG_INFO << "Runtime Clone with Backend:: " << option.backend << " in " << option.device << "." << std::endl;
        runtime->option = option;
        runtime->backend_ = backend_->clone(option, stream, device_id);
        return runtime;
    }


    void Runtime::bind_input_tensor(const std::string& name, Tensor& input) {
        bool is_exist = false;
        for (auto& t : input_tensors_) {
            if (t.get_name() == name) {
                is_exist = true;
                t.from_external_memory(input.data(), input.shape(), input.dtype());
                break;
            }
        }
        if (!is_exist) {
            Tensor new_tensor(input.data(), input.shape(), input.dtype());
            input_tensors_.emplace_back(std::move(new_tensor));
        }
    }

    void Runtime::bind_output_tensor(const std::string& name, Tensor& output) {
        bool is_exist = false;
        for (auto& t : output_tensors_) {
            if (t.get_name() == name) {
                is_exist = true;
                t.from_external_memory(output.data(), output.shape(), output.dtype());
                break;
            }
        }
        if (!is_exist) {
            Tensor new_tensor(output.data(), output.shape(), output.dtype());
            input_tensors_.emplace_back(std::move(new_tensor));
        }
    }

    Tensor* Runtime::get_output_tensor(const std::string& name) {
        for (auto& t : output_tensors_) {
            if (t.get_name() == name) {
                return &t;
            }
        }
        MD_LOG_WARN << "The output name [" << name << "] don't exist." << std::endl;
        return nullptr;
    }

    void Runtime::create_ort_backend() {
#ifdef ENABLE_ORT
        backend_ = std::make_unique<OrtBackend>();
        if (!backend_->init(option)) {
            MD_LOG_ERROR << "Failed to initialize " << option.backend << "." << std::endl;
        }
#else
        MD_LOG_FATAL << "OrtBackend is not available, please compiled with ENABLE_ORT=ON." << std::endl;
#endif
        MD_LOG_INFO << "Runtime initialized with " << option.backend << " in " << option.device << "." << std::endl;
    }

    void Runtime::create_mnn_backend() {
#ifdef ENABLE_MNN
        backend_ = std::make_unique<MnnBackend>();
        if (!backend_->init(option)) {
            MD_LOG_ERROR << "Failed to initialize " << option.backend << "." << std::endl;
        }
#else
        MD_LOG_FATAL << "MNNBackend is not available, please compiled with ENABLE_MNN=ON." << std::endl;
#endif
        MD_LOG_INFO << "Runtime initialized with " << option.backend << " in " << option.device << "." << std::endl;
    }

    void Runtime::create_trt_backend() {
#ifdef ENABLE_TRT
        backend_ = std::make_unique<TrtBackend>();
        if (!backend_->init(option)) {
            MD_LOG_ERROR << "Failed to initialize " << option.backend << "." << std::endl;
        }
#else
        MD_LOG_FATAL << "MNNBackend is not available, please compiled with ENABLE_TRT=ON." << std::endl;
#endif
        MD_LOG_INFO << "Runtime initialized with " << option.backend << " in " << option.device << "." << std::endl;
    }
}
