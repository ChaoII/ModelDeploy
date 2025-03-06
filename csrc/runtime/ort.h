//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "option.h"
#include "utils.h"
#include "onnxruntime_cxx_api.h"
namespace modeldeploy {

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    ONNXTensorElementDataType dtype;
};

class  OrtBackend {
public:
    OrtBackend() = default;

    virtual ~OrtBackend() = default;

    void build_option(const RuntimeOption& option);

    bool init(const RuntimeOption& option);

    bool infer(std::vector<MDTensor>& inputs, std::vector<MDTensor>* outputs,bool copy_to_fd=true) ;

    [[nodiscard]] size_t num_inputs() const { return inputs_desc_.size(); }

    [[nodiscard]] size_t num_outputs() const { return outputs_desc_.size(); }

    TensorInfo get_input_info(int index);
    TensorInfo get_output_info(int index);
    std::vector<TensorInfo> get_input_infos();
    std::vector<TensorInfo> get_output_infos();

private:
    bool init_from_onnx(const std::string& model_buffer,
                        const RuntimeOption& option = RuntimeOption());
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "MD"};
    Ort::Session session_{nullptr};
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::IoBinding> binding_;
    std::vector<TensorInfo> inputs_desc_;
    std::vector<TensorInfo> outputs_desc_;
    bool initialized_ = false;
    std::string model_file_name;
    RuntimeOption option_;
};
}