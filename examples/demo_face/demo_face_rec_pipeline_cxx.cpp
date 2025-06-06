//
// Created by AC on 2025-01-13.
//

#include <iostream>
#include "csrc/vision/common/display/display.h"
#include "csrc/vision.h"

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
    const auto im = cv::imread("../../test_data/test_images/test_face_detection4.jpg");
    auto im_bak = im.clone();
    TimerArray timers;
    constexpr int loop_count = 50;
    std::vector<modeldeploy::vision::FaceRecognitionResult> results;
    for (int i = 0; i < loop_count; i++) {
        if (!model.predict(im, &results, &timers)) {
            std::cerr << "Failed to predict." << std::endl;
            return -1;
        }
    }
    timers.print_benchmark();
    dis_face_rec(results);
    return 0;
}
