//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include "csrc/vision/utils.h"
#include "csrc/vision.h"

int main() {
    auto faceid_model = modeldeploy::vision::face::SeetaFaceID(
        "../../test_data/test_models/face/face_recognizer_fp16.onnx");
    assert(table_model.Initialized());
    // auto im0 = cv::imread("../../test_data/test_images/test_face_id1.jpg");
    //    auto im0 = cv::imread("vis_result.jpg");
    auto im1 = cv::imread("../../test_data/test_images/test_face_id4.jpg");
    modeldeploy::vision::FaceRecognitionResult result0;
    modeldeploy::vision::FaceRecognitionResult result1;
    //    if (!faceid_model.predict(im0, &result0)) {
    //        std::cerr << "Failed to predict." << std::endl;
    //        return -1;
    //    }
    if (!faceid_model.predict(im1, &result1)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }

    // const auto similarity = modeldeploy::vision::utils::compute_similarity(result0.embedding, result1.embedding);

    // std::cout << result0.Str() << std::endl;
    result1.display();
    // std::cout << similarity << std::endl;
    return 0;
}
