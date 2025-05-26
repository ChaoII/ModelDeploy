//
// Created by AC on 2025-05-26.
//

#include <iostream>

#include "capi/vision/lpr/lpr_rec_capi.h"
#include "capi/utils/md_image_capi.h"


int main() {
    MDModel model;
    md_create_lpr_rec_model(&model, "../../test_data/test_models/plate_recognition_color.onnx");
    MDImage image = md_read_image("../../test_data/test_images/test_lpr_recognizer.jpg");
    MDLPRResults c_results;
    md_lpr_rec_predict(&model, &image, &c_results);
    std::cout << "car plate color: " << c_results.data[0].car_plate_color << std::endl;
    std::cout << "car_plate_str: " << c_results.data[0].car_plate_str << std::endl;
    md_free_image(&image);
    md_free_lpr_rec_result(&c_results);
    md_free_lpr_rec_model(&model);
    return 0;
}
