//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include <csrc/vision/common/display/display.h>

#include "csrc/vision/utils.h"
#include "csrc/vision.h"

int main() {
    auto faceid_model = modeldeploy::vision::face::SeetaFaceID(
        "../../test_data/test_models/face/face_recognizer_fp16.onnx");
    assert(faceid_model.is_initialized());
    auto im0 = cv::imread("../../test_data/test_images/test_face_id1.jpg");
    auto im1 = cv::imread("../../test_data/test_images/test_face_id4.jpg");
    std::vector<modeldeploy::vision::FaceRecognitionResult> result0(1);
    std::vector<modeldeploy::vision::FaceRecognitionResult> result1(1);
    faceid_model.predict(im0, &result0[0]);
    faceid_model.predict(im1, &result1[0]);
    const auto similarity = modeldeploy::vision::utils::compute_similarity(result0[0].embedding, result1[0].embedding);
    modeldeploy::vision::dis_face_rec(result0);
    modeldeploy::vision::dis_face_rec(result1);
    std::cout << "similarity: " << similarity << std::endl;
    return 0;
}
