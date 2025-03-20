//
// Created by aichao on 2025/2/24.
//
#include "csrc/vision.h"
#ifdef WIN32

#include <windows.h>
#endif

int main() {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    modeldeploy::vision::ocr::PPOCRv4 ocr("../../test_data/test_models/ocr/det_infer.onnx",
                                          "../../test_data/test_models/ocr/cls_infer.onnx",
                                          "../../test_data/test_models/ocr/rec_infer1.onnx",
                                          "../../test_data/key.txt");
    auto img = cv::imread("../../test_data/test_images/test_ocr1.png");
    modeldeploy::vision::OCRResult result;
    ocr.predict(img, &result);
    std::cout << result.Str() << std::endl;
}
