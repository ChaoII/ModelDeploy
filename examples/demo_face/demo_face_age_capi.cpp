//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include "capi/utils/md_image_capi.h"
#include "capi/vision/face/face_age_capi.h"

int main() {
    MDModel model;
    md_create_face_age_model(&model, "../../test_data/test_models/face/age_predictor.onnx");
    MDImage image = md_read_image("../../test_data/test_images/test_face_id1.jpg");
    MDFaceAgeResult c_result;
    md_face_age_predict(&model, &image, &c_result);
    std::cout << "age is: " << c_result << std::endl;
    md_free_image(&image);
    md_free_face_age_result(&c_result);
    md_free_face_age_model(&model);
    return 0;
}
