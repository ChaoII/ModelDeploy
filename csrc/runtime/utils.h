//
// Created by aichao on 2025/2/19.
//

#pragma once
#include <onnxruntime_cxx_api.h>
#include "../core/md_tensor.h"
namespace modeldeploy {
    ONNXTensorElementDataType get_ort_dtype(const MDDataType& fd_dtype);

    // Convert OrtDataType to FDDataType
    MDDataType get_md_dtype(const ONNXTensorElementDataType& ort_dtype);

    // Create Ort::Value
    // is_backend_cuda specify if the onnxruntime use CUDAExectionProvider
    // While is_backend_cuda = true, and tensor.device = Device::GPU
    // Will directly share the cuda data in tensor to OrtValue
    Ort::Value create_ort_value(MDTensor& tensor, bool is_backend_cuda = false);


    void ort_value_to_md_tensor(const Ort::Value& value, MDTensor* tensor, const std::string& name, bool copy_to_fd);
}