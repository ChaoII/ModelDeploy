//
// Created by aichao on 2025/2/26.
//

#include <iostream>
#include <fstream>
#ifdef WIN32
#include <windows.h>
#endif
#include "capi/utils/md_image_capi.h"
#include "capi/vision/ocr/ocr_recognition_capi.h"

int main(int argc, char** argv) {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    MDStatusCode ret;
    //简单百宝箱
    MDModel model;
    if ((ret = md_create_ocr_recognition_model(&model,
                                               "../tests/test_models/ocr/rec_infer.onnx",
                                               "../tests/key.txt")) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    MDImage image = md_read_image("../tests/test_images/test_ocr_recognition1.png");
    MDOCRResult result;
    if ((ret = md_ocr_recognition_model_predict(&model, &image, &result)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    std::cout << "text: " << result.text << " score: " << result.score << std::endl;
    md_free_ocr_recognition_result(&result);
    md_free_ocr_recognition_model(&model);
}
