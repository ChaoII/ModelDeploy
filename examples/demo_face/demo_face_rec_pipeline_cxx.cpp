//
// Created by AC on 2025-01-13.
//

#include <iostream>
#include <csrc/vision/face/face_rec_pipeline/face_rec_pipeline.h>

#include "../../csrc/vision.h"
#include "../../csrc/vision/common/visualize/visualize.h"

#ifdef WIN32
#include <windows.h>
#endif

int main() {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    auto model = modeldeploy::vision::face::FaceRecognizerPipeline(
        "../../test_data/test_models/face/scrfd_2.5g_bnkps_shape640x640.onnx",
        "../../test_data/test_models/face/face_recognizer.onnx");
    auto im = cv::imread("../../test_data/test_images/test_face_detection4.jpg");
    auto im_bak = im.clone();
    std::vector<modeldeploy::vision::FaceRecognitionResult> ress;
    if (!model.predict(im, &ress)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    std::cout << ress.size() << std::endl;
    for (auto& res : ress) {
        res.display();
    }
    return 0;
}
