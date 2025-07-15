//
// Created by aichao on 2025/2/20.
//

#include "csrc/runtime/backends/ort/ort_backend.h"
#include "csrc/runtime/backends/ort/ort_backend_impl.h"


namespace modeldeploy {
    OrtBackend::OrtBackend(): impl_(std::make_unique<OrtBackendImpl>()) {
    }

    OrtBackend::~OrtBackend() = default;


    bool OrtBackend::init(const RuntimeOption& option) {
        return impl_->init(option);
    }


    bool OrtBackend::infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) {
        return impl_->infer(inputs, outputs);
    }

    std::unique_ptr<BaseBackend> OrtBackend::clone(RuntimeOption& runtime_option,
                                                   void* stream,
                                                   const int device_id) {
        auto backend = std::make_unique<OrtBackend>();
        backend->impl_ = impl_->clone(runtime_option, stream, device_id);
        return backend;
    }

    size_t OrtBackend::num_inputs() const {
        return impl_->num_inputs();
    }

    size_t OrtBackend::num_outputs() const {
        return impl_->num_outputs();
    }

    TensorInfo OrtBackend::get_input_info(const int index) {
        return impl_->get_input_info(index);
    }

    std::vector<TensorInfo> OrtBackend::get_input_infos() {
        return impl_->get_input_infos();
    }

    TensorInfo OrtBackend::get_output_info(const int index) {
        return impl_->get_output_info(index);
    }

    std::vector<TensorInfo> OrtBackend::get_output_infos() {
        return impl_->get_output_infos();
    }

    std::map<std::string, std::string> OrtBackend::get_custom_meta_data() const {
        return impl_->get_custom_meta_data();
    }
}
