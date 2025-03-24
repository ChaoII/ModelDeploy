//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include "csrc/vision/faceid/seetaface.h"
#include "capi/utils/md_image_capi.h"

int main() {
    auto faceid_model = modeldeploy::vision::faceid::AdaFace(
        "../../test_data/test_models/face/face_recognition.onnx");
    assert(table_model.Initialized());
    auto im = cv::imread("../../test_data/test_images/test_face_id.jpg");
    auto im_bak = im.clone();
    modeldeploy::vision::FaceRecognitionResult result;
    if (!faceid_model.Predict(im, &result)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    std::cout << result.Str() << std::endl;
    return 0;
}
