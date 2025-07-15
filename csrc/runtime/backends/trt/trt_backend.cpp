//
// Created by aichao on 2025/6/27.
//
#include "csrc/runtime/backends/trt/trt_backend.h"
#include "csrc/runtime/backends/trt/trt_backend_impl.h"


namespace modeldeploy {
    TrtBackend::TrtBackend(): impl_(std::make_unique<TrtBackendImpl>()) {
    }

    TrtBackend::~TrtBackend() = default;


    bool TrtBackend::init(const RuntimeOption& option) {
        return impl_->init(option);
    }


    bool TrtBackend::infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) {
        return impl_->infer(inputs, outputs);
    }


    size_t TrtBackend::num_inputs() const {
        return impl_->num_inputs();
    }

    size_t TrtBackend::num_outputs() const {
        return impl_->num_outputs();
    }


    TensorInfo TrtBackend::get_input_info(int index) {
        return impl_->get_input_info(index);
    }

    std::vector<TensorInfo> TrtBackend::get_input_infos() {
        return impl_->get_input_infos();
    }

    TensorInfo TrtBackend::get_output_info(int index) {
        return impl_->get_output_info(index);
    }

    std::vector<TensorInfo> TrtBackend::get_output_infos() {
        return impl_->get_output_infos();
    }

    std::unique_ptr<BaseBackend> TrtBackend::clone(RuntimeOption& runtime_option,
                                                   void* stream, int device_id) {
        auto backend = std::make_unique<TrtBackend>();
        backend->impl_ = impl_->clone(runtime_option, stream, device_id);
        return backend;
    }
} // namespace modeldeploy
