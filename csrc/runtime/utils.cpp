//
// Created by aichao on 2025/2/19.
//
#include "csrc/runtime/utils.h"
#include "csrc/core/md_log.h"

namespace modeldeploy {
    ONNXTensorElementDataType get_ort_dtype(const MDDataType::Type& md_dtype) {
        if (md_dtype == MDDataType::Type::FP32) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        if (md_dtype == MDDataType::Type::FP64) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        }
        if (md_dtype == MDDataType::Type::INT32) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        }
        if (md_dtype == MDDataType::Type::INT64) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        }
        if (md_dtype == MDDataType::Type::UINT8) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        }
        if (md_dtype == MDDataType::Type::INT8) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        }
        MD_LOG_ERROR("Unrecognized modeldeploy data type:{}.", MDDataType::str(md_dtype));
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }

    MDDataType::Type get_md_dtype(const ONNXTensorElementDataType& ort_dtype) {
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return MDDataType::Type::FP32;
        }
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
            return MDDataType::Type::FP64;
        }
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            return MDDataType::Type::INT32;
        }
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            return MDDataType::Type::INT64;
        }
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            return MDDataType::Type::UINT8;
        }
        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
            return MDDataType::Type::INT8;
        }
        std::cerr << "Unrecognized ort data type:" << ort_dtype << "." << std::endl;
        return MDDataType::Type::FP32;
    }

    Ort::Value create_ort_value(MDTensor& tensor, const bool is_backend_cuda) {
        if (is_backend_cuda) {
            const Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator, 0,
                                              OrtMemTypeDefault);
            auto ort_value = Ort::Value::CreateTensor(
                memory_info, tensor.mutable_data(), tensor.total_bytes(), tensor.shape.data(),
                tensor.shape.size(), get_ort_dtype(tensor.dtype));
            return ort_value;
        }
        const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        auto ort_value = Ort::Value::CreateTensor(
            memory_info, tensor.data(), tensor.total_bytes(), tensor.shape.data(),
            tensor.shape.size(), get_ort_dtype(tensor.dtype));
        return ort_value;
    }

    void ort_value_to_md_tensor(const Ort::Value& value, MDTensor* tensor,
                                const std::string& name, const bool copy_to_fd) {
        const auto info = value.GetTensorTypeAndShapeInfo();
        const auto ort_dtype = info.GetElementType();
        size_t num_el = info.GetElementCount();
        const auto shape = info.GetShape();
        MDDataType::Type dtype;

        if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            dtype = MDDataType::Type::FP32;
            num_el *= sizeof(float);
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            dtype = MDDataType::Type::INT32;
            num_el *= sizeof(int32_t);
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            dtype = MDDataType::Type::INT64;
            num_el *= sizeof(int64_t);
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
            dtype = MDDataType::Type::FP64;
            num_el *= sizeof(double);
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            dtype = MDDataType::Type::UINT8;
            num_el *= sizeof(uint8_t);
        }
        else if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
            dtype = MDDataType::Type::INT8;
            num_el *= sizeof(int8_t);
        }
        else {
            MD_LOG_ERROR("Unrecognized data type of {} while calling OrtBackend::CopyToCpu().",
                         onnx_type_to_string(ort_dtype));
        }
        const void* value_ptr = value.GetTensorData<void*>();
        if (copy_to_fd) {
            tensor->resize(shape, dtype, name);
            memcpy(tensor->mutable_data(), value_ptr, num_el);
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
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, "DOUBLE"},
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "UINT8"},
            {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, "INT8"},
            // 添加更多类型映射...
        };
        return type_map.count(type) ? type_map.at(type) : "UNKNOWN";
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
