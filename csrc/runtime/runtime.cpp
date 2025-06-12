//
// Created by aichao on 2025/5/22.
//


#include "csrc/runtime/backends/ort/ort.h"
#include "csrc/runtime/runtime.h"


namespace modeldeploy {
    bool Runtime::init(const RuntimeOption& _option) {
        option = _option;

        if (option.backend == Backend::NONE) {
            return false;
        }
        if (option.backend == Backend::ORT) {
            CreateOrtBackend();
        }
        else {
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


    void Runtime::CreateOrtBackend() {
        backend_ = std::make_unique<OrtBackend>();
        if (!backend_->init(option)) {
            MD_LOG_ERROR << "Failed to initialize Backend::ORT." << std::endl;
        }
        MD_LOG_INFO << "Runtime initialized with Backend::ORT in " << option.device << "." << std::endl;
    }
}
