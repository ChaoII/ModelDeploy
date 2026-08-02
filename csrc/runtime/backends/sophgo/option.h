//
// Created by aichao on 2025/8/2.
//
#pragma once

#include <string>

namespace modeldeploy {
    struct SophgoBackendOption {
        // TPU 设备 id（0 为第一个）
        int device_id = 0;
        // .bmodel 文件路径；为空时取 RuntimeOption.model_file
        std::string bmodel_path;
        // 是否为设备内存直用（硬解码零拷贝路径预留）
        bool use_device_input = false;
    };
} // namespace modeldeploy
