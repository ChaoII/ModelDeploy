//
// Created by aichao on 2025/5/22.
//


#include "runtime/runtime.h"
#ifdef ENABLE_TRT
#include "backends/trt/trt_backend.h"
#endif

#ifdef ENABLE_ORT
#include "runtime/backends/ort/ort_backend.h"
#endif
#ifdef ENABLE_MNN
#include "runtime/backends/mnn/mnn_backend.h"
#endif


namespace modeldeploy {
    bool Runtime::init(const RuntimeOption& _option) {
        option = _option;
        if (option.backend == Backend::ORT) {
            return create_ort_backend();
        }
        if (option.backend == Backend::MNN) {
            return create_mnn_backend();
        }
        if (option.backend == Backend::TRT) {
            return create_trt_backend();
        }
        MD_LOG_ERROR << "The " << option.backend << " backend is not supported now." << std::endl;
        return false;
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

    std::unique_ptr<Runtime> Runtime::clone(void* stream, const int device_id) const {
        auto cloned_backend = backend_->clone(this->option, stream, device_id);
        if (!cloned_backend) {
            MD_LOG_ERROR << "Failed to clone backend" << std::endl;
            return nullptr;
        }
        auto new_runtime = std::make_unique<Runtime>();
        new_runtime->backend_ = std::move(cloned_backend);
        new_runtime->option = this->option;
        new_runtime->option.set_external_stream(stream);
        new_runtime->option.device_id = device_id;
        MD_LOG_INFO << "Runtime cloned with Backend::" << option.backend
            << " on device " << (device_id >= 0 ? device_id : option.device_id)
            << "." << std::endl;
        return new_runtime;
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

    bool Runtime::create_ort_backend() {
#ifdef ENABLE_ORT
        backend_ = std::make_unique<OrtBackend>();
        if (!backend_->init(option)) {
            MD_LOG_ERROR << "Failed to initialize " << option.backend << "." << std::endl;
            return false;
        }
#else
        MD_LOG_FATAL << "OrtBackend is not available, please compiled with ENABLE_ORT=ON." << std::endl;
        return false;
#endif
        MD_LOG_INFO << "Runtime initialized with " << option.backend << " in " << option.device << "." << std::endl;
        return true;
    }

    bool Runtime::create_mnn_backend() {
#ifdef ENABLE_MNN
        backend_ = std::make_unique<MnnBackend>();
        if (!backend_->init(option)) {
            MD_LOG_ERROR << "Failed to initialize " << option.backend << "." << std::endl;
            return false;
        }
#else
        MD_LOG_FATAL << "MNNBackend is not available, please compiled with ENABLE_MNN=ON." << std::endl;
        return false;
#endif
        MD_LOG_INFO << "Runtime initialized with " << option.backend << " in " << option.device << "." << std::endl;
        return true;
    }

    bool Runtime::create_trt_backend() {
#ifdef ENABLE_TRT
        backend_ = std::make_unique<TrtBackend>();
        if (!backend_->init(option)) {
            MD_LOG_ERROR << "Failed to initialize " << option.backend << "." << std::endl;
            return false;
        }
#else
        MD_LOG_FATAL << "MNNBackend is not available, please compiled with ENABLE_TRT=ON." << std::endl;
        return false;
#endif
        MD_LOG_INFO << "Runtime initialized with " << option.backend << " in " << option.device << "." << std::endl;
        return true;
    }
}
