//
// Created by aichao on 2025/5/22.
//

#pragma once

namespace modeldeploy {
    enum Backend {
        ORT,
        MNN,
        NONE
    };

    enum Device {
        CPU,
        GPU,
        OPENCL,
        VULKAN,
    };

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
        case Backend::NONE: return "NONE";
        default: return "Unknown";
        }
    }

    inline std::ostream& operator<<(std::ostream& os, const Device device) {
        return os << device_to_string(device);
    }

    inline std::ostream& operator<<(std::ostream& os, const Backend backend) {
        return os << backend_to_string(backend);
    }
}
