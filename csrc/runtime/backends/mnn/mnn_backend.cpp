//
// Created by aichao on 2025/2/20.
//

#include "csrc/runtime/backends/mnn/mnn_backend_impl.h"
#include "csrc/runtime/backends/mnn/mnn_backend.h"


namespace modeldeploy {
    MnnBackend::MnnBackend(): impl_(std::make_unique<MnnBackendImpl>()) {
    }

    MnnBackend::~MnnBackend() = default;


    bool MnnBackend::init(const RuntimeOption& option) {
        return impl_->init(option);
    }


    bool MnnBackend::infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) {
        return impl_->infer(inputs, outputs);
    }

    std::unique_ptr<BaseBackend> MnnBackend::clone(RuntimeOption& runtime_option,
                                                   void* stream,
                                                   const int device_id) {
        auto backend = std::make_unique<MnnBackend>();
        backend->impl_ = std::move(impl_);
        return backend;
    }

    size_t MnnBackend::num_inputs() const {
        return impl_->num_inputs();
    }

    size_t MnnBackend::num_outputs() const {
        return impl_->num_outputs();
    }

    TensorInfo MnnBackend::get_input_info(const int index) {
        return impl_->get_input_info(index);
    }

    std::vector<TensorInfo> MnnBackend::get_input_infos() {
        return impl_->get_input_infos();
    }

    TensorInfo MnnBackend::get_output_info(const int index) {
        return impl_->get_output_info(index);
    }

    std::vector<TensorInfo> MnnBackend::get_output_infos() {
        return impl_->get_output_infos();
    }

    std::map<std::string, std::string> MnnBackend::get_custom_meta_data() const {
        return impl_->get_custom_meta_data();
    }
}
