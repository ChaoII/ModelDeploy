//
// Created by AC on 2025-01-13.
//

#include "capi/vision/face/face_rec_capi.h"
#include "capi/utils/md_image_capi.h"


int main() {
    MDModel model;
    md_create_face_rec_model(&model, "../../test_data/test_models/face/face_recognizer.onnx");
    MDImage image = md_read_image("../../test_data/test_images/test_face_id4.jpg");
    MDFaceRecognizerResult c_results;
    md_face_rec_predict(&model, &image, &c_results);
    md_print_face_rec_result(&c_results);
    md_free_image(&image);
    md_free_face_rec_result(&c_results);
    md_free_face_rec_model(&model);
    return 0;
}
