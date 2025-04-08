//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include "csrc/vision.h"

int main() {
    auto face_as_pipeline_model = modeldeploy::vision::face::SeetaFaceAsPipeline(
        "../../test_data/test_models/face/scrfd_2.5g_bnkps_shape640x640.onnx",
        "../../test_data/test_models/face/fas_first.onnx",
        "../../test_data/test_models/face/fas_second.onnx", 8);
    auto im0 = cv::imread("../../test_data/test_images/test_face_detection4.jpg");
    modeldeploy::vision::FaceAntiSpoofResult results;
    if (!face_as_pipeline_model.predict(im0, &results, 0.3)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    for (auto& anti_spoof : results.anti_spoofs) {
        switch (anti_spoof) {
        case modeldeploy::vision::FaceAntiSpoofType::FUZZY:
            std::cout << "FUZZY" << std::endl;
            break;
        case modeldeploy::vision::FaceAntiSpoofType::REAL:
            std::cout << "REAL" << std::endl;
            break;
        case modeldeploy::vision::FaceAntiSpoofType::SPOOF:
            std::cout << "SPOOF" << std::endl;
            break;
        }
    }
    return 0;
}
