//
// Created by AC on 2025-01-13.
//

#include <iostream>

#include "capi/utils/md_image_capi.h"
#include "capi/vision/face/face_rec_pipeline_capi.h"


int main() {
    MDModel model;
    md_create_face_rec_pipeline_model(&model,
                                      "../../test_data/test_models/face/scrfd_2.5g_bnkps_shape640x640.onnx",
                                      "../../test_data/test_models/face/face_recognizer.onnx", 8);
    MDImage image = md_read_image("../../test_data/test_images/test_face_detection4.jpg");
    MDFaceRecognizerResults c_results;
    md_face_rec_pipeline_predict(&model, &image, &c_results);

    md_print_face_rec_pipeline_result(&c_results);
    md_free_image(&image);
    md_free_face_rec_pipeline_result(&c_results);
    md_free_face_rec_pipeline_model(&model);
    return 0;
}
