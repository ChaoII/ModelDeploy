//
// Created by aichao on 2025/2/20.
//
#include "csrc/function/eigen.h"

namespace modeldeploy::function {
    std::shared_ptr<EigenDeviceWrapper> EigenDeviceWrapper::instance_ = nullptr;

    std::shared_ptr<EigenDeviceWrapper> EigenDeviceWrapper::GetInstance() {
        if (instance_ == nullptr) {
            instance_ = std::make_shared<EigenDeviceWrapper>();
        }
        return instance_;
    }

    const Eigen::DefaultDevice* EigenDeviceWrapper::GetDevice() const {
        return &device_;
    }
}
