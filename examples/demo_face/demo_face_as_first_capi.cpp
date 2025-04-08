//
// Created by AC on 2025-01-13.
//

#include <iostream>

#include "capi/utils/md_image_capi.h"
#include "capi/vision/face/face_as_first_capi.h"


int main() {
    MDModel model;
    md_create_face_as_first_model(&model, "../../test_data/test_models/face/fas_first.onnx");
    MDImage image = md_read_image("../../test_data/test_images/test_face_id3.jpg");
    float c_result;
    md_face_as_first_predict(&model, &image, &c_result);
    std::cout << "passive_result: " << c_result << std::endl;
    if (c_result > 0.8) {
        std::cout << "REAL" << std::endl;
    }
    else {
        std::cout << "SPOOF" << std::endl;
    }
    md_free_image(&image);
    md_free_face_as_first_model(&model);
    return 0;
}
