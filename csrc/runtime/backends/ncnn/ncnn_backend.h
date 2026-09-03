#pragma once

#include <memory>
#include <vector>
#include <ncnn/net.h>
#include "core/tensor.h"
#include "runtime/backends/backend.h"
#include "runtime/backends/ncnn/option.h"

namespace modeldeploy {
    class NcnnBackend : public BaseBackend {
    public:
        NcnnBackend() = default;
        ~NcnnBackend() override = default;
        bool init(const RuntimeOption& runtime_option) override;
        [[nodiscard]] size_t num_inputs() const override { return input_names_.size(); }
        [[nodiscard]] size_t num_outputs() const override { return output_names_.size(); }
        TensorInfo get_input_info(int index) override;
        TensorInfo get_output_info(int index) override;
        std::vector<TensorInfo> get_input_infos() override;
        std::vector<TensorInfo> get_output_infos() override;
        bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) override;
        std::unique_ptr<BaseBackend> clone(const RuntimeOption& runtime_option,
                                           void* stream = nullptr,
                                           int device_id = -1) override;

    private:
        std::unique_ptr<ncnn::Net> net_;
        NcnnBackendOption option_;
        RuntimeOption saved_option_;
        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;
    };
} // namespace modeldeploy
