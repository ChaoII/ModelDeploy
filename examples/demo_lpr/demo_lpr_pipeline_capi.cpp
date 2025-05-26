//
// Created by AC on 2025-05-26.
//

#include "capi/utils/md_image_capi.h"
#include "capi/vision/lpr/lpr_pipeline_capi.h"


int main() {
    MDModel model;
    md_create_lpr_pipeline_model(&model,
                                 "../../test_data/test_models/yolov5plate.onnx",
                                 "../../test_data/test_models/plate_recognition_color.onnx", 8);
    MDImage image = md_read_image("../../test_data/test_images/test_lpr_pipeline2.jpg");
    MDLPRResults c_results;
    md_lpr_pipeline_predict(&model, &image, &c_results);
    md_print_lpr_pipeline_result(&c_results);
    md_draw_lpr_pipeline_result(&image, &c_results, "../../test_data/msyh.ttc", 14, 4, 0.3, 0);
    md_show_image(&image);
    md_free_image(&image);
    md_free_lpr_pipeline_result(&c_results);
    md_free_lpr_pipeline_model(&model);
    return 0;
}
