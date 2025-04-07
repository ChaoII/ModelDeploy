//
// Created by AC on 2025-01-13.
//

#include <iostream>
#include "../../csrc/vision.h"

#ifdef WIN32

#include <windows.h>

#endif

int main() {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    auto model = modeldeploy::vision::lpr::LprRecognizer(
        "../../test_data/test_models/plate_recognition_color.onnx");
    auto im = cv::imread("../../test_data/test_images/test_lpr_recognizer.jpg");
    auto im_bak = im.clone();
    modeldeploy::vision::LprResult res;
    if (!model.predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    std::cout << "car plate color: " << res.car_plate_colors[0] << std::endl;
    std::cout << "car_plate_str: " << res.car_plate_strs[0] << std::endl;
    return 0;
}
