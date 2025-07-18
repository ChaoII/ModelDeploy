//
// Created by aichao on 2025/2/24.
//

#include <capi/utils/md_utils_capi.h>
#include <capi/utils/internal/utils.h>

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
    option.enable_trt = true;
    option.enable_fp16 = true;
    modeldeploy::vision::ocr::PaddleOCR ocr("../../test_data/test_models/ocr/ppocrv5_mobile/det_infer.onnx",
                                            "../../test_data/test_models/ocr/ppocrv5_mobile/cls_infer.onnx",
                                            "../../test_data/test_models/ocr/ppocrv5_mobile/rec_infer.onnx",
                                            "../../test_data/ppocrv5_dict.txt",
                                            option);
    auto img = modeldeploy::ImageData::imread("../../test_data/test_images/test_ocr1.jpg");
    ocr.set_rec_batch_size(16);
    ocr.set_rec_batch_size(16);
    modeldeploy::vision::OCRResult result;
    TimerArray timers;
    int loop_count = 15;
    for (int i = 0; i < loop_count; i++) {
        ocr.predict(img, &result, &timers);
    }
    timers.print_benchmark();
    modeldeploy::vision::dis_ocr(result);
    const auto vis_image = modeldeploy::vision::vis_ocr(img, result, "../../test_data/msyh.ttc");
    vis_image.imshow("ocr");
}
