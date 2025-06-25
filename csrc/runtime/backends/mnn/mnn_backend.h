//
// Created by aichao on 2025/6/20.
//

#pragma once

#include <memory>
#include <vector>
#include <MNN/Interpreter.hpp>
#include "csrc/core/tensor.h"
#include "csrc/runtime/backends/backend.h"
#include "csrc/runtime/backends/mnn/option.h"


namespace modeldeploy {
    class MnnBackend : public BaseBackend {
    public:
        MnnBackend() = default;
        ~MnnBackend() override;
        bool init(const RuntimeOption& runtime_option) override;

        [[nodiscard]] size_t num_inputs() const override {
            return inputs_desc_.size();
        }

        [[nodiscard]] size_t num_outputs() const override {
            return outputs_desc_.size();
        }

        TensorInfo get_input_info(int index) override;
        TensorInfo get_output_info(int index) override;
        std::vector<TensorInfo> get_input_infos() override;
        std::vector<TensorInfo> get_output_infos() override;
        bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) override;

    private:
        void build_option(const RuntimeOption& option);

        MnnBackendOption option_;
        std::string model_buffer_;
        MNN::Session* session_ = nullptr;
        std::shared_ptr<MNN::Interpreter> net_;
        std::vector<TensorInfo> inputs_desc_;
        std::vector<TensorInfo> outputs_desc_;
    };
} // namespace fastdeploy
