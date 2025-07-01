//
// Created by aichao on 2025/2/19.
//

#include "csrc/core/md_log.h"
#include "csrc/runtime/backends/ort/utils.h"

#include <map>


namespace modeldeploy {
    ONNXTensorElementDataType get_ort_dtype(const DataType& dtype) {
        if (dtype == DataType::FP32) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        if (dtype == DataType::FP64) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        }
        if (dtype == DataType::INT32) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        }
        if (dtype == DataType::INT64) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        }
        if (dtype == DataType::UINT8) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        }
        if (dtype == DataType::INT8) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        }
        MD_LOG_ERROR << "Unrecognized modeldeploy data type." << std::endl;
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }

    DataType get_md_dtype(const ONNXTensorElementDataType& ort_dtype) {
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return DataType::FP32;
        }
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
            return DataType::FP64;
        }
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            return DataType::INT32;
        }
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            return DataType::INT64;
        }
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            return DataType::UINT8;
        }
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
            return DataType::INT8;
        }
        MD_LOG_ERROR << "Unrecognized ort data type:" << ort_dtype << "." << std::endl;
        return DataType::FP32;
    }

    Ort::Value create_ort_value(Tensor& tensor, const Ort::MemoryInfo& memory_info) {
        auto ort_value = Ort::Value::CreateTensor(
            memory_info, tensor.data(), tensor.byte_size(), tensor.shape().data(),
            tensor.shape().size(), get_ort_dtype(tensor.dtype()));
        return ort_value;
    }

    void ort_value_to_md_tensor(const Ort::Value& value, Tensor* tensor,
                                const std::string& name) {
        const auto info = value.GetTensorTypeAndShapeInfo();
        const auto ort_dtype = info.GetElementType();
        size_t num_el = info.GetElementCount();
        const auto shape = info.GetShape();
        DataType dtype;

        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            dtype = DataType::FP32;
            num_el *= sizeof(float);
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            dtype = DataType::INT32;
            num_el *= sizeof(int32_t);
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            dtype = DataType::INT64;
            num_el *= sizeof(int64_t);
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
            dtype = DataType::FP64;
            num_el *= sizeof(double);
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            dtype = DataType::UINT8;
            num_el *= sizeof(uint8_t);
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
            dtype = DataType::INT8;
            num_el *= sizeof(int8_t);
        }
        else {
            MD_LOG_ERROR << "Unrecognized data type of " << onnx_type_to_string(ort_dtype) <<
                " while calling OrtBackend::CopyToCpu()." << std::endl;
        }
        const void* value_ptr = value.GetTensorData<void*>();
        tensor->allocate(shape, dtype, name);
        memcpy(tensor->data(), value_ptr, num_el);
    }

    std::string onnx_type_to_string(const ONNXTensorElementDataType type) {
        static const std::map<ONNXTensorElementDataType, std::string> type_map = {
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, "FLOAT16"},
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "FLOAT"},
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "INT32"},
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, "STRING"},
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, "DOUBLE"},
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "UINT8"},
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, "INT8"},
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "INT64"},
            // 添加更多类型映射...
        };
        auto it = type_map.find(type);
        return it != type_map.end() ? it->second : "UNKNOWN";
    }

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
