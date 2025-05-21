//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include "csrc/vision.h"

int main() {
    auto face_antispoof_model = modeldeploy::vision::face::SeetaFaceAsSecond(
        "../../test_data/test_models/face/fas_second.onnx");
    assert(face_antispoof_model.is_initialized());
    auto im0 = cv::imread("../../test_data/test_images/test_face_as_second2.jpg");
    std::vector<std::tuple<int, float>> results;
    if (!face_antispoof_model.predict(im0, &results)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }

    // 如果检测到对象，那么可以初步认为是攻击人脸(全局性的)
    if (!results.empty()) {
        std::cout << "Spoof" << std::endl;
    }
    else {
        std::cout << "real" << std::endl;
    }
    return 0;
}
