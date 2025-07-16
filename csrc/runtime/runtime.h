//
// Created by aichao on 2025/5/22.
//
#pragma once

#include "core/tensor.h"
#include "runtime/runtime_option.h"
#include "runtime/backends/backend.h"

namespace modeldeploy {
    struct MODELDEPLOY_CXX_EXPORT Runtime {
        bool init(const RuntimeOption& _option);

        bool infer(std::vector<Tensor>& input_tensors, std::vector<Tensor>* output_tensors) const;

        bool infer();

        Runtime* clone(void* stream = nullptr, int device_id = -1);

        [[nodiscard]] size_t num_inputs() const { return backend_->num_inputs(); }

        [[nodiscard]] size_t num_outputs() const { return backend_->num_outputs(); }

        [[nodiscard]] TensorInfo get_input_info(int index) const;

        [[nodiscard]] TensorInfo get_output_info(int index) const;

        [[nodiscard]] std::vector<TensorInfo> get_input_infos() const;

        [[nodiscard]] std::vector<TensorInfo> get_output_infos() const;

        void bind_input_tensor(const std::string& name, Tensor& input);

        void bind_output_tensor(const std::string& name, Tensor& output);

        Tensor* get_output_tensor(const std::string& name);

        [[nodiscard]] std::map<std::string, std::string> get_custom_meta_data() const {
            return backend_->get_custom_meta_data();
        }

        [[nodiscard]] bool is_initialized() const { return backend_->is_initialized(); }

        RuntimeOption option;

    private:
        void create_ort_backend();

        void create_mnn_backend();

        void create_trt_backend();

        std::unique_ptr<BaseBackend> backend_;
        std::vector<Tensor> input_tensors_;
        std::vector<Tensor> output_tensors_;
    };
}
