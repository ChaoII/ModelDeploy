//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include "csrc/vision.h"

int main() {
    auto face_antispoof_model = modeldeploy::vision::face::SeetaFaceAntiSpoofSecond(
        "../../test_data/test_models/face/fas_second.onnx");
    assert(table_model.Initialized());
    auto im0 = cv::imread("../../test_data/test_images/test_face_antispoof1.jpg");
    std::vector<std::tuple<int, float>> results;
    if (!face_antispoof_model.predict(&im0, &results)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    return 0;
}
