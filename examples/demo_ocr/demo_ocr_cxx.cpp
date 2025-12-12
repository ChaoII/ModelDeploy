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
    option.use_gpu();
    option.enable_trt = false;
    option.enable_fp16 = true;
    modeldeploy::vision::ocr::PaddleOCR ocr("C:/Users/aichao/Desktop/fsdownload/det_infer.onnx",
                                            "../../test_data/test_models/ocr/ppocrv4_mobile/cls_infer.onnx",
                                            "C:/Users/aichao/Desktop/fsdownload/rec_infer.onnx",
                                            "F:/unattended_laboratory/OCR/rec/dataset/dict.txt",
                                            option);
    // auto img = modeldeploy::ImageData::imread("../../test_data/test_images/ocr2.jpg");
    auto img = modeldeploy::ImageData::imread("C:/Users/aichao/Desktop/4324.png");
    ocr.set_rec_batch_size(16);
    ocr.get_detector()->get_preprocessor().set_max_side_len(1440);
    ocr.get_detector()->get_postprocessor().set_det_db_thresh(0.3);
    ocr.get_detector()->get_postprocessor().set_det_db_box_thresh(0.6);
    ocr.get_detector()->get_postprocessor().set_det_db_unclip_ratio(1.8);
    // ocr.get_detector()->get_postprocessor().set_use_dilation(true);
    modeldeploy::vision::OCRResult result;
    TimerArray timers;
    int loop_count = 50;
    for (int i = 0; i < loop_count; i++) {
        ocr.predict(img, &result, &timers);
    }
    timers.print_benchmark();
    modeldeploy::vision::dis_ocr(result);
    const auto vis_image = modeldeploy::vision::vis_ocr(img, result, "../../test_data/msyh.ttc");
    vis_image.imshow("ocr");
}
