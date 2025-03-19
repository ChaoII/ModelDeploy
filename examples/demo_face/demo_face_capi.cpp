//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include "capi/vision/face/face_capi.h"
#include "capi/utils/md_image_capi.h"
#include "capi/vision/detection/detection_capi.h"

int main() {

    MDStatusCode ret;
    MDModel model;
    ret = md_create_face_model(&model, "../test_data/test_models/seetaface", MD_MASK, 1);
    std::cout << "create model result: " << ret << std::endl;
    auto image = md_read_image("../test_data/test_images/test_face3.jpg");

    std::cout << "====================face detection==========================" << std::endl;
    MDDetectionResults r_face_detect;
    ret = md_face_detection(&model, &image, &r_face_detect);
    std::cout << "face detection " << (ret ? "failed" : "success") << std::endl;
    std::cout << "face size is: " << r_face_detect.size << std::endl;
    md_draw_detection_result(&image, &r_face_detect, "../test_data/msyh.ttc", 20, 0.5, 1);
//    md_show_image(&image);
    md_free_detection_result(&r_face_detect);

    std::cout << "====================feature extract==========================" << std::endl;
    MDFaceFeature feature;
    ret = md_face_feature_e2e(&model, &image, &feature);
    std::cout << "feature predict " << (ret ? "failed" : "success") << std::endl;
    std::cout << "feature size is: " << feature.size << std::endl;
    md_free_face_feature(&feature);

    std::cout << "==================anti spoofing predict======================" << std::endl;
    MDFaceAntiSpoofingResult as_result;
    ret = md_face_anti_spoofing(&model, &image, &as_result);
    std::cout << "anti spoofing predict " << (ret ? (std::string("failed ") + std::to_string(ret)) : "success")
              << std::endl;
    md_print_face_anti_spoofing_result(as_result);

    std::cout << "========================quality evaluate======================" << std::endl;
    MDFaceQualityEvaluateResult qs_result;
    ret = md_face_quality_evaluate(&model, &image, MDFaceQualityEvaluateType::BRIGHTNESS, &qs_result);
    std::cout << "brightness evaluate " << (ret ? "failed" : "success") << std::endl;
    std::cout << "brightness quality is: " << qs_result << std::endl;

    ret = md_face_quality_evaluate(&model, &image, MDFaceQualityEvaluateType::CLARITY, &qs_result);
    std::cout << "clarity evaluate " << (ret ? "failed" : "success") << std::endl;
    std::cout << "clarity quality is: " << qs_result << std::endl;

    ret = md_face_quality_evaluate(&model, &image, MDFaceQualityEvaluateType::CLARITY_EX, &qs_result);
    std::cout << "clarity ex evaluate " << (ret ? "failed" : "success") << std::endl;
    std::cout << "clarity ex quality is: " << qs_result << std::endl;

    ret = md_face_quality_evaluate(&model, &image, MDFaceQualityEvaluateType::INTEGRITY, &qs_result);
    std::cout << "integrity evaluate " << (ret ? "failed" : "success") << std::endl;
    std::cout << "integrity quality is: " << qs_result << std::endl;

    ret = md_face_quality_evaluate(&model, &image, MDFaceQualityEvaluateType::NO_MASK, &qs_result);
    std::cout << "no mask evaluate " << (ret ? "failed" : "success") << std::endl;
    std::cout << "no mask quality is: " << qs_result << std::endl;

    ret = md_face_quality_evaluate(&model, &image, MDFaceQualityEvaluateType::POSE, &qs_result);
    std::cout << "pose evaluate " << (ret ? "failed" : "success") << std::endl;
    std::cout << "pose quality is: " << qs_result << std::endl;

    ret = md_face_quality_evaluate(&model, &image, MDFaceQualityEvaluateType::RESOLUTION, &qs_result);
    std::cout << "resolution evaluate " << (ret ? "failed" : "success") << std::endl;
    std::cout << "resolution quality is: " << qs_result << std::endl;


    std::cout << "========================age predict===========================" << std::endl;
    int age = 0;
    ret = md_face_age_predict(&model, &image, &age);
    std::cout << "age predict " << (ret ? "failed" : "success") << std::endl;
    std::cout << "age is: " << age << std::endl;

    std::cout << "========================gender predict=========================" << std::endl;
    MDGenderResult r_gender;
    ret = md_face_gender_predict(&model, &image, &r_gender);
    std::cout << "gender predict " << (ret ? "failed" : "success") << std::endl;
    std::cout << "gender is: " << (r_gender ? "Female" : "male") << std::endl;

    std::cout << "========================eye state predict========================" << std::endl;
    MDEyeStateResult r_eye_state;
    ret = md_face_eye_state_predict(&model, &image, &r_eye_state);
    std::cout << "eye status predict " << (ret ? "failed" : "success") << std::endl;
    std::cout << "left eye status is: " << (r_eye_state.left_eye ? "Open" : "Close") << std::endl;
    std::cout << "left eye status is: " << (r_eye_state.right_eye ? "Open" : "Close") << std::endl;

    md_free_image(&image);

    md_free_face_model(&model);

    return 0;

}