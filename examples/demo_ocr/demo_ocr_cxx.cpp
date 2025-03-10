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

    modeldeploy::vision::ocr::PPOCRv4 ocr("../tests/test_models/ocr/det_infer.onnx",
                                          "../tests/test_models/ocr/cls_infer.onnx",
                                          "../tests/test_models/ocr/rec_infer.onnx",
                                          "../tests/key.txt");
    auto img = cv::imread("../tests/test_images/test_ocr1.png");
    modeldeploy::vision::OCRResult result;
    ocr.predict(img, &result);
    std::cout << result.Str() << std::endl;
}
