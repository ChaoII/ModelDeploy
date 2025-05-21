//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include "csrc/vision.h"

int main() {
    auto faceid_model = modeldeploy::vision::face::SeetaFaceAge(
        "../../test_data/test_models/face/age_predictor.onnx");
    assert(faceid_model.is_initialized());
    auto im0 = cv::imread("../../test_data/test_images/test_face_id.jpg");
    int age = 0;
    if (!faceid_model.predict(im0, &age)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    std::cout << "age: " << age << std::endl;
    return 0;
}
