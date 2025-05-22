//
// Created by aichao on 2025/5/22.
//

#pragma once
#include <ostream>
#include <map>

namespace modeldeploy {
    /*! Inference backend supported in FastDeploy */
    enum Backend {
        ORT, //< ONNX Runtime, support Paddle/ONNX format model,
        NONE
    };

    enum Device {
        CPU,
        GPU
    };
}
