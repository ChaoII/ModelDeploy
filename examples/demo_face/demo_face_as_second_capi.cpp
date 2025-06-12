//
// Created by AC on 2025-01-13.
//

#include <iostream>

#include "capi/utils/md_image_capi.h"
#include "capi/utils/md_utils_capi.h"
#include "capi/vision/face/face_as_second_capi.h"


int main() {
    MDModel model;
    const MDRuntimeOption option = md_create_default_runtime_option();
    md_create_face_as_second_model(&model, "../../test_data/test_models/face/fas_second.onnx", &option);
    MDImage image = md_read_image("../../test_data/test_images/test_face_as_second2.jpg");
    MDFaceAsSecondResults c_results;
    md_face_as_second_predict(&model, &image, &c_results);
    if (c_results.size > 0) {
        std::cout << "SPOOF" << std::endl;
    }
    else {
        std::cout << "REAL" << std::endl;
    }
    md_free_image(&image);
    md_free_face_as_second_result(&c_results);
    md_free_face_as_second_model(&model);
    return 0;
}
