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
    modeldeploy::vision::ocr::DBDetector db_detector("../../test_data/test_models/ocr/repsvtr_mobile/det_infer.onnx",
                                                     option);
    auto img = modeldeploy::ImageData::imread("../../test_data/test_images/ocr2.jpg");
    db_detector.get_preprocessor().set_max_side_len(1440);
    // db_detector.get_preprocessor().set_det_image_shape({3, 1440, 1440});
    db_detector.get_postprocessor().set_det_db_thresh(0.3);
    // db_detector.get_postprocessor().set_det_db_box_thresh(0.5);
    db_detector.get_postprocessor().set_det_db_unclip_ratio(1.0);
    // db_detector.get_postprocessor().set_use_dilation(true);
    modeldeploy::vision::OCRResult result;
    db_detector.predict(img, &result);
    modeldeploy::vision::dis_ocr(result);
    const auto vis_image = modeldeploy::vision::vis_ocr(img, result, "../../test_data/msyh.ttc");
    vis_image.imshow("ocr_db_detector");
}
