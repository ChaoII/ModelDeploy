//
// Created by aichao on 2025/2/24.
//
#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"
#ifdef WIN32

#include <windows.h>
#endif

int main() {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    modeldeploy::vision::ocr::PPOCRv4 ocr("../../test_data/test_models/ocr/repsvtr_mobile/det_infer.onnx",
                                          "../../test_data/test_models/ocr/repsvtr_mobile/cls_infer.onnx",
                                          "../../test_data/test_models/ocr/repsvtr_mobile/rec_infer1.onnx",
                                          "../../test_data/key.txt");
    auto img = cv::imread("../../test_data/test_images/test_ocr1.png");
    modeldeploy::vision::OCRResult result;
    ocr.predict(img, &result);
    std::cout << result.str() << std::endl;
    auto vis_image = modeldeploy::vision::vis_ocr(img, result, "../../test_data/msyh.ttc", 14, 0.5, 0);
    cv::imshow("test", vis_image);
    cv::waitKey(0);
}
