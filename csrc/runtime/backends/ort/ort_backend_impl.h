//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "onnxruntime_cxx_api.h"
#include "runtime/backends/backend.h"
#include "runtime/backends/ort/option.h"

namespace modeldeploy {
    struct OrtValueInfo {
        std::string name;
        std::vector<int64_t> shape;
        ONNXTensorElementDataType dtype;
    };

    class OrtBackendImpl  {
    public:
        OrtBackendImpl() = default;

        ~OrtBackendImpl()  = default;

        void build_option(const OrtBackendOption& option);

        bool init(const RuntimeOption& option) ;

        bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) const;

        std::unique_ptr<OrtBackendImpl> clone(RuntimeOption& runtime_option,
                                           void* stream = nullptr, int device_id = -1) const;

        [[nodiscard]] size_t num_inputs() const  { return inputs_desc_.size(); }

        [[nodiscard]] size_t num_outputs() const  { return outputs_desc_.size(); }

        TensorInfo get_input_info(int index) ;
        TensorInfo get_output_info(int index) ;
        std::vector<TensorInfo> get_input_infos() ;
        std::vector<TensorInfo> get_output_infos() ;
        [[nodiscard]] std::map<std::string, std::string> get_custom_meta_data() const ;

    private:
        bool init_from_onnx(const std::string& model_buffer,
                            const OrtBackendOption& option = OrtBackendOption());

        Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "MD"};
        std::shared_ptr<Ort::Session> shared_session_{nullptr};
        Ort::SessionOptions session_options_;
        std::unique_ptr<Ort::IoBinding> binding_;
        std::vector<OrtValueInfo> inputs_desc_;
        std::vector<OrtValueInfo> outputs_desc_;
        bool initialized_ = false;
        std::string model_file_name_;
        std::string model_buffer_;
        OrtBackendOption option_;
    };
}
