//
// Created by aichao on 2025/8/2.
//
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "runtime/backends/backend.h"
#include "runtime/backends/sophgo/option.h"

namespace modeldeploy {
    class SophgoBackend : public BaseBackend {
    public:
        SophgoBackend() = default;
        ~SophgoBackend() override;

        bool init(const RuntimeOption& option) override;

        bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) override;

        std::unique_ptr<BaseBackend> clone(const RuntimeOption& runtime_option,
                                           void* stream = nullptr,
                                           int device_id = -1) override;

        [[nodiscard]] size_t num_inputs() const override { return inputs_desc_.size(); }
        [[nodiscard]] size_t num_outputs() const override { return outputs_desc_.size(); }

        TensorInfo get_input_info(int index) override;
        TensorInfo get_output_info(int index) override;
        std::vector<TensorInfo> get_input_infos() override;
        std::vector<TensorInfo> get_output_infos() override;
        [[nodiscard]] std::map<std::string, std::string> get_custom_meta_data() const override;

    private:
        std::string bmodel_path_;
        std::string graph_name_;
        // 不透明句柄：实际是 sail::Engine / sail::Handle（见 .cpp，避免在头文件引入 sail 依赖）
        void* engine_ = nullptr;
        void* handle_ = nullptr;
        std::vector<TensorInfo> inputs_desc_;
        std::vector<TensorInfo> outputs_desc_;
    };
} // namespace modeldeploy
