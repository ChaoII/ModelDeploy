//
// Created by aichao on 2025/2/20.
//

#pragma once


#include "csrc/runtime/backends/backend.h"

namespace modeldeploy {
    class OrtBackendImpl;

    class MODELDEPLOY_CXX_EXPORT OrtBackend : public BaseBackend {
    public:
        OrtBackend();

        ~OrtBackend() override;

        bool init(const RuntimeOption& option) override;

        bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) override;

        std::unique_ptr<BaseBackend> clone(RuntimeOption& runtime_option,
                                           void* stream = nullptr, int device_id = -1) override;

        [[nodiscard]] size_t num_inputs() const override;

        [[nodiscard]] size_t num_outputs() const override;

        TensorInfo get_input_info(int index) override;

        TensorInfo get_output_info(int index) override;

        std::vector<TensorInfo> get_input_infos() override;

        std::vector<TensorInfo> get_output_infos() override;

        [[nodiscard]] std::map<std::string, std::string> get_custom_meta_data() const override;

    private:
        std::unique_ptr<OrtBackendImpl> impl_;
    };
}
