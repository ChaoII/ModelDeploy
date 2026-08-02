//
// Created by aichao on 2025/8/2.
//
#pragma once

#include <memory>
#include "core/enum_variables.h"
#include "vision/processors/processor_backend.h"

namespace modeldeploy::vision {

// 根据设备与后端创建对应的前后处理 backend
std::unique_ptr<VisionProcessorBackend> create_processor_backend(
    Device device, Backend backend, int device_id = -1);

} // namespace modeldeploy::vision
