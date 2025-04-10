//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include "csrc/vision.h"

int main() {
    auto faceid_model = modeldeploy::vision::face::SeetaFaceAsFirst(
        "../../test_data/test_models/face/fas_first.onnx");
    assert(table_model.Initialized());
    auto im0 = cv::imread("../../test_data/test_images/test_face_id4.jpg");
    float result;
    if (!faceid_model.predict(im0, &result)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    std::cout << "result: " << result << std::endl;
    return 0;
}
