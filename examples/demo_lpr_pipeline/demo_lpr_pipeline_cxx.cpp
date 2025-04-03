//
// Created by AC on 2025-01-13.
//

#include <iostream>
#include "csrc/vision.h"

#ifdef WIN32

#include <windows.h>

#endif

int main() {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    auto model = modeldeploy::vision::lpr::LprPipeline(
        "../../test_data/test_models/yolov5plate.onnx",
        "../../test_data/test_models/plate_recognition_color.onnx");
    auto im = cv::imread("../../test_data/test_images/test_lpr_pipeline2.jpg");
    auto im_bak = im.clone();

    std::vector<modeldeploy::vision::CarPlateRecognizerResult> res;
    if (!model.predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }


    return 0;
}
