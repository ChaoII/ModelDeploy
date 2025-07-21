//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include "csrc/vision.h"

int main() {
    auto faceid_model = modeldeploy::vision::face::SeetaFaceGender(
        "../../test_data/test_models/face/gender_predictor.onnx");
    const auto im0 = modeldeploy::ImageData::imread("../../test_data/test_images/test_face_gender1.jpg");
    int gender_id = 0;
    if (!faceid_model.predict(im0, &gender_id)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    const std::string gender = gender_id == 0 ? "female" : "male";
    std::cout << "gender: " << gender << std::endl;
    im0.imshow("123");
    return 0;
}
