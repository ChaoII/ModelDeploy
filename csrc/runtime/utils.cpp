//
// Created by aichao on 2025/2/19.
//
#include "utils.h"

namespace modeldeploy {
    ONNXTensorElementDataType get_ort_dtype(const MDDataType& fd_dtype) {
        if (fd_dtype == MDDataType::FP32) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        else if (fd_dtype == MDDataType::FP64) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        }
        else if (fd_dtype == MDDataType::INT32) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        }
        else if (fd_dtype == MDDataType::INT64) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        }
        else if (fd_dtype == MDDataType::UINT8) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        }
        else if (fd_dtype == MDDataType::INT8) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        }
        std::cerr << "Unrecognized fastdeply data type:" << str(fd_dtype) << "."
            << std::endl;
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }

    MDDataType get_md_dtype(const ONNXTensorElementDataType& ort_dtype) {
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return MDDataType::FP32;
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
            return MDDataType::FP64;
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            return MDDataType::INT32;
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            return MDDataType::INT64;
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            return MDDataType::UINT8;
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
            return MDDataType::INT8;
        }
        std::cerr << "Unrecognized ort data type:" << ort_dtype << "." << std::endl;
        return MDDataType::FP32;
    }

    Ort::Value create_ort_value(MDTensor& tensor, bool is_backend_cuda) {
        if (is_backend_cuda) {
            Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator, 0,
                                        OrtMemTypeDefault);
            auto ort_value = Ort::Value::CreateTensor(
                memory_info, tensor.mutable_data(), tensor.total_bytes(), tensor.shape.data(),
                tensor.shape.size(), get_ort_dtype(tensor.dtype));
            return ort_value;
        }
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        auto ort_value = Ort::Value::CreateTensor(
            memory_info, tensor.data(), tensor.total_bytes(), tensor.shape.data(),
            tensor.shape.size(), get_ort_dtype(tensor.dtype));
        return ort_value;
    }

    void ort_value_to_md_tensor(const Ort::Value& value, MDTensor* tensor, const std::string& name, bool copy_to_fd) {
        const auto info = value.GetTensorTypeAndShapeInfo();
        const auto data_type = info.GetElementType();
        size_t numel = info.GetElementCount();
        auto shape = info.GetShape();
        MDDataType dtype;

        if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            dtype = MDDataType::FP32;
            numel *= sizeof(float);
        }
        else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            dtype = MDDataType::INT32;
            numel *= sizeof(int32_t);
        }
        else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            dtype = MDDataType::INT64;
            numel *= sizeof(int64_t);
        }
        else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
            dtype = MDDataType::FP64;
            numel *= sizeof(double);
        }
        else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            dtype = MDDataType::UINT8;
            numel *= sizeof(uint8_t);
        }
        else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
            dtype = MDDataType::INT8;
            numel *= sizeof(int8_t);
        }
        else {
            std::cerr << "Unrecognized data type of " << data_type
                << " while calling OrtBackend::CopyToCpu()." << std::endl;
        }
        const void* value_ptr = value.GetTensorData<void*>();
        if (copy_to_fd) {
            tensor->resize(shape, dtype, name);
            memcpy(tensor->mutable_data(), value_ptr, numel);
        }
        else {
            tensor->name = name;
            tensor->set_external_data(shape, dtype, const_cast<void*>(value_ptr));
        }
    }

    std::string onnx_type_to_string(const ONNXTensorElementDataType type) {
        static const std::map<ONNXTensorElementDataType, std::string> type_map = {
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "FLOAT"},
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "INT32"},
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, "STRING"},
            // 添加更多类型映射...
        };
        return type_map.count(type) ? type_map.at(type) : "UNKNOWN";
    }

    // 辅助函数：形状转字符串
    std::string shape_to_string(const std::vector<int64_t>& shape) {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << shape[i];
        }
        ss << "]";
        return ss.str();
    }
}
