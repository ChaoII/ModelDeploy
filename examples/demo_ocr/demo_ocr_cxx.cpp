//
// Created by aichao on 2025/2/24.
//
#include "csrc/vision.h"

int main() {
    modeldeploy::vision::ocr::PPOCRv4 ocr("det_infer.onnx",
                                          "cls_infer.onnx",
                                          "rec_infer.onnx",
                                          "E:/CLionProjects/ModelDeploy/tests/key.txt");
    auto img = cv::imread("E:/CLionProjects/ModelDeploy/tests/test_images/test_ocr1.png");
    modeldeploy::vision::OCRResult result;
    ocr.predict(img, &result);
    std::cout << result.Str() << std::endl;
}
