//
// Created by aichao on 2025/6/27.
//

#pragma once

#include <vector>

#include "csrc/runtime/backends/backend.h"

namespace modeldeploy {
    class TrtBackendImpl;

    class MODELDEPLOY_CXX_EXPORT TrtBackend : public BaseBackend {
    public:
        TrtBackend();
        ~TrtBackend() override;
        bool init(const RuntimeOption& runtime_option) override;

        bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) override;

        [[nodiscard]] size_t num_inputs() const override;

        [[nodiscard]] size_t num_outputs() const override;

        TensorInfo get_input_info(int index) override;

        TensorInfo get_output_info(int index) override;

        std::vector<TensorInfo> get_input_infos() override;

        std::vector<TensorInfo> get_output_infos() override;

        std::unique_ptr<BaseBackend> clone(RuntimeOption& runtime_option,
                                           void* stream = nullptr,
                                           int device_id = -1) override;

    private:
        std::unique_ptr<TrtBackendImpl> impl_;
    };
} // namespace modeldeploy
