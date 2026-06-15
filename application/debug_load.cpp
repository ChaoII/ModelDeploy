#include <iostream>
#include "inference_engine.hpp"

int main() {
    ModelConfig mcfg;
    mcfg.name = "yolo11n";
    mcfg.type = "detection";
    mcfg.path = "E:/CLionProjects/ModelDeploy/test_data/test_models/yolo11n_nms.onnx";
    mcfg.backend = "ort";
    mcfg.device = "gpu";

    InferenceEngine engine;
    if (!engine.load(mcfg)) {
        std::cerr << "FAIL: load model" << std::endl;
        return 1;
    }
    std::cout << "OK: model loaded" << std::endl;
    return 0;
}
