//
// Created by aichao on 2025/2/24.
//


#include "csrc/vision.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"
#ifdef WIN32
#include <windows.h>
#endif


int main() {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    modeldeploy::RuntimeOption option;
    option.use_ort_backend();
    modeldeploy::vision::ocr::Recognizer ocr_recognizer("../../test_data/test_models/onnx/ocr/ppocrv5_mobile/rec_infer1.onnx",
                                                        "../../test_data/dict.txt",
                                                        option);
    // ocr_recognizer.get_preprocessor().set_static_shape_infer(true);
    ocr_recognizer.get_preprocessor().set_rec_image_shape({3, 48, 320});
    auto img = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_ocr_recognition.jpg");


    std::vector<modeldeploy::vision::ImageData> images;
    for (int i = 0; i < 64; ++i) {
        images.emplace_back(img);
    }


    modeldeploy::vision::OCRResult result;
    constexpr int warming_up_count = 10;
    for (int i = 0; i < warming_up_count; ++i) {
        ocr_recognizer.batch_predict(images, &result);
    }
    TimerArray timers;
    constexpr int loop_count = 100;
    for (int i = 0; i < loop_count; ++i) {
        ocr_recognizer.batch_predict(images, &result, &timers);
    }
    timers.print_benchmark();
    modeldeploy::vision::dis_ocr(result);
    const auto vis_image = modeldeploy::vision::vis_ocr(img, result, "../../test_data/msyh.ttc");
    (void)vis_image.imwrite("ocr_rec_out.jpg");
}
