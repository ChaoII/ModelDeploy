//
// Created by aichao on 2025/2/19.
//

#pragma once
#include <string>
#include <onnxruntime_cxx_api.h>
#include "csrc/core/tensor.h"

namespace modeldeploy {
    ONNXTensorElementDataType get_ort_dtype(const DataType& dtype);

    // Convert OrtDataType to FDDataType
    DataType get_md_dtype(const ONNXTensorElementDataType& ort_dtype);

    // Create Ort::Value
    // is_backend_cuda specify if the onnxruntime use CUDAExectionProvider
    // While is_backend_cuda = true, and tensor.device = Device::GPU
    // Will directly share the cuda data in tensor to OrtValue
    Ort::Value create_ort_value(Tensor& tensor, const Ort::MemoryInfo& memory_info);


    void ort_value_to_md_tensor(const Ort::Value& value, Tensor* tensor, const std::string& name);


    std::string onnx_type_to_string(ONNXTensorElementDataType type);

    // 辅助函数：形状转字符串
    std::string shape_to_string(const std::vector<int64_t>& shape);
}
