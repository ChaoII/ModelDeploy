//
// Created by aichao on 2025/2/24.
//


#include "csrc/vision.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"


int main() {
    modeldeploy::RuntimeOption option;
    option.use_gpu();
    option.enable_trt = true;
    option.enable_fp16 = true;
    modeldeploy::vision::ocr::Recognizer ocr_recognizer("../../test_data/test_models/ocr/ppocrv5_mobile/rec_infer.onnx",
                                                        "../../test_data/ppocrv5_dict.txt",
                                                        option);
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
    // modeldeploy::vision::dis_ocr(result);
    const auto vis_image = modeldeploy::vision::vis_ocr(img, result, "../../test_data/msyh.ttc");
    vis_image.imshow("ocr_db_detector");
}
