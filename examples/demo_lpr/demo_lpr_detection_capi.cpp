//
// Created by AC on 2025-5-26.
//

#include "capi/utils/md_image_capi.h"
#include "capi/utils/md_utils_capi.h"
#include "capi/vision/lpr/lpr_det_capi.h"


int main() {
    MDModel model;
    const MDRuntimeOption option = md_create_default_runtime_option();
    md_create_lpr_det_model(&model, "../../test_data/test_models/yolov5plate.onnx", &option);
    MDImage image = md_read_image("../../test_data/test_images/test_lpr_pipeline2.jpg");
    MDKeyPointResults c_results;
    md_lpr_det_predict(&model, &image, &c_results);
    md_draw_lpr_det_result(&image, &c_results, "../../test_data/msyh.ttc", 14, 4, 0.3, 0);
    md_print_lpr_det_result(&c_results);
    md_show_image(&image);
    md_free_image(&image);
    md_free_lpr_det_result(&c_results);
    md_free_lpr_det_model(&model);
    return 0;
}
