//
// Created by aichao on 2025/2/24.
//


#include "csrc/vision.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"


int main() {
    modeldeploy::RuntimeOption option;
    option.use_gpu();
    option.enable_trt = false;
    option.enable_fp16 = true;
    modeldeploy::vision::ocr::DBDetector db_detector("../../test_data/test_models/ocr/ppocrv5_mobile/det_infer2.onnx",
                                                     option);
    auto img = modeldeploy::vision::ImageData::imread("../../test_data/test_images/ocr2.jpg");
    // auto img = modeldeploy::vision::ImageData::imread("C:/Users/aichao/Desktop/stock/0003.jpg");
    db_detector.get_preprocessor().set_max_side_len(1280);
    db_detector.get_preprocessor().use_cuda_preproc();
    db_detector.get_postprocessor().set_det_db_thresh(0.3);
    db_detector.get_postprocessor().set_det_db_box_thresh(0.5);
    db_detector.get_postprocessor().set_det_db_unclip_ratio(1.5);
    // db_detector.get_postprocessor().set_use_dilation(true);

    std::vector<modeldeploy::vision::OCRResult> results;
    std::vector<modeldeploy::vision::ImageData> images;
    for (int i = 0; i < 8; i++) {
        images.push_back(img);
    }

    constexpr int warming_up_count = 10;
    for (int i = 0; i < warming_up_count; ++i) {
        db_detector.batch_predict(images, &results);
    }
    TimerArray timers;
    constexpr int loop_count = 100;
    for (int i = 0; i < loop_count; ++i) {
        db_detector.batch_predict(images, &results, &timers);
    }
    timers.print_benchmark();
    // modeldeploy::vision::dis_ocr(result);
    const auto vis_image = modeldeploy::vision::vis_ocr(img, results[0], "../../test_data/msyh.ttc");
    vis_image.imshow("ocr_db_detector");
}
