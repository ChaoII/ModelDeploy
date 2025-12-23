//
// Created by aichao on 2025/5/22.
//

#pragma once

namespace modeldeploy {
    enum Backend {
        ORT,
        MNN,
        TRT,
        NONE
    };

    enum class Device {
        CPU,
        GPU,
        OPENCL,
        VULKAN,
    };

   enum class DataType {
        FP32,
        FP64,
        INT32,
        INT64,
        UINT8,
        INT8,
        UNKNOWN
    };

    // 辅助函数实现
    inline std::string datatype_to_string(const DataType dtype) {
        switch (dtype) {
        case DataType::FP32: return "FP32";
        case DataType::FP64: return "FP64";
        case DataType::INT32: return "INT32";
        case DataType::INT64: return "INT64";
        case DataType::UINT8: return "UINT8";
        case DataType::INT8: return "INT8";
        case DataType::UNKNOWN: return "UNKNOWN";
        default: return "";
        }
    }

    inline std::string device_to_string(const Device device) {
        switch (device) {
        case Device::CPU: return "CPU";
        case Device::GPU: return "GPU";
        case Device::OPENCL: return "OPENCL";
        case Device::VULKAN: return "VULKAN";
        default: return "Unknown";
        }
    }

    inline std::string backend_to_string(const Backend backend) {
        switch (backend) {
        case Backend::ORT: return "ORT";
        case Backend::MNN: return "MNN";
        case Backend::TRT: return "TRT";
        case Backend::NONE: return "NONE";
        default: return "Unknown";
        }
    }

    inline std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
        return os << datatype_to_string(dtype);
    }

    inline std::ostream& operator<<(std::ostream& os, const Device device) {
        return os << device_to_string(device);
    }

    inline std::ostream& operator<<(std::ostream& os, const Backend backend) {
        return os << backend_to_string(backend);
    }
}
