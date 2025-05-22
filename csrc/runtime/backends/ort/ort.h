//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "option.h"
#include "utils.h"
#include "csrc/core/md_decl.h"
#include "onnxruntime_cxx_api.h"
#include "csrc/runtime/backends/backend.h"

namespace modeldeploy {
    struct OrtValueInfo {
        std::string name;
        std::vector<int64_t> shape;
        ONNXTensorElementDataType dtype;
    };

    class MODELDEPLOY_CXX_EXPORT OrtBackend : public BaseBackend {
    public:
        OrtBackend() = default;

        ~OrtBackend() override = default;

        void build_option(const OrtBackendOption& option);

        bool init(const RuntimeOption& option) override;

        bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) override;

        [[nodiscard]] size_t num_inputs() const override { return inputs_desc_.size(); }

        [[nodiscard]] size_t num_outputs() const override { return outputs_desc_.size(); }

        TensorInfo get_input_info(int index) override;
        TensorInfo get_output_info(int index) override;
        std::vector<TensorInfo> get_input_infos() override;
        std::vector<TensorInfo> get_output_infos() override;
        [[nodiscard]] std::map<std::string, std::string> get_custom_meta_data() const override;

    private:
        bool init_from_onnx(const std::string& model_buffer,
                            const OrtBackendOption& option = OrtBackendOption());

        Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "MD"};
        Ort::Session session_{nullptr};
        Ort::SessionOptions session_options_;
        std::unique_ptr<Ort::IoBinding> binding_;
        std::vector<OrtValueInfo> inputs_desc_;
        std::vector<OrtValueInfo> outputs_desc_;
        bool initialized_ = false;
        std::string model_file_name;
        OrtBackendOption option_;
    };
}
