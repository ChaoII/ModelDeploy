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
    auto im0 = cv::imread("../../test_data/test_images/test_face_detection.jpg");
    modeldeploy::vision::FaceAntiSpoofResult results;
    if (!face_as_pipeline_model.predict(im0, &results)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    for (auto& anti_spoof : results.anti_spoofs) {
        std::cout << static_cast<int>(anti_spoof) << std::endl;
    }
    return 0;
}
