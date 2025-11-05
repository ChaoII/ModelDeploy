//
// Created by AC on 2025-01-13.
//

#include "capi/utils/md_image_capi.h"
#include "capi/utils/md_utils_capi.h"
#include "capi/vision/face/face_det_capi.h"


int main() {
    MDModel model;
    const MDRuntimeOption option = md_create_default_runtime_option();
    md_create_face_det_model(&model, "../../test_data/test_models/face/scrfd_2.5g_bnkps_shape640x640.onnx", &option);
    MDImage image = md_read_image("../../test_data/test_images/test_face_detection3.jpg");
    MDKeyPointResults c_results;
    md_face_det_predict(&model, &image, &c_results);
    md_draw_face_det_result(&image, &c_results, "../../test_data/msyh.ttc", 14, 4, 0.3, 0);
    md_print_face_det_result(&c_results);
    md_show_image(&image);
    md_free_image(&image);
    md_free_face_det_result(&c_results);
    md_free_face_det_model(&model);
    return 0;
}
