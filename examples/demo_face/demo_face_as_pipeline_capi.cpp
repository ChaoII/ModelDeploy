//
// Created by AC on 2025-01-13.
//

#include <iostream>

#include "capi/utils/md_image_capi.h"
#include "capi/utils/md_utils_capi.h"
#include "capi/vision/face/face_as_pipeline_capi.h"


int main() {
    MDModel model;
    const MDRuntimeOption option = md_create_default_runtime_option();
    md_create_face_as_pipeline_model(&model,
                                     "../../test_data/test_models/face/scrfd_2.5g_bnkps_shape640x640.onnx",
                                     "../../test_data/test_models/face/fas_first.onnx",
                                     "../../test_data/test_models/face/fas_second.onnx", &option);
    MDImage image = md_read_image("../../test_data/test_images/test_face_detection4.jpg");
    MDFaceAsResults c_results;
    md_face_as_pipeline_predict(&model, &image, &c_results);

    // REAL = 0, FUZZY = 1, SPOOF = 2
    for (int i = 0; i < c_results.size; i++) {
        switch (c_results.data[i]) {
        case 0:
            std::cout << "REAL" << std::endl;
            break;
        case 1:
            std::cout << "FUZZY" << std::endl;
            break;
        case 2:
            std::cout << "SPOOF" << std::endl;
            break;
        default:
            std::cout << "UNKNOWN" << std::endl;
            break;
        }
    }

    md_free_image(&image);
    md_free_face_as_pipeline_result(&c_results);
    md_free_face_as_pipeline_model(&model);
    return 0;
}
