#include <iostream>
#include "src/ocr_capi.h"
#include "src/detection_capi.h"
#include "src/utils.h"
#include <chrono>
#include "src/face/face_capi.h"

#ifdef _WIN32

#include <Windows.h>

#endif


int test_face() {
    MDStatusCode ret = MDStatusCode::Success;
    MDModel model;
    ret = md_create_face_model(&model, "../tests/models/seetaface", MD_MASK, 1);
    std::cout << "create model result: " << ret << std::endl;
    auto image = md_read_image("../tests/test_images/test_face2.jpg");
    MDDetectionResults r_face_detect;
    md_face_detection(&model, &image, &r_face_detect);
    md_draw_detection_result(&image, &r_face_detect, "../tests/msyh.ttc", 20, 0.5, 1);
    md_show_image(&image);
    md_free_detection_result(&r_face_detect);

    MDFaceFeature feature;
    md_face_feature_e2e(&model, &image, &feature);
    std::cout << feature.size << std::endl;
    md_free_face_feature(&feature);

    MDFaceAntiSpoofingResult result;
    md_face_anti_spoofing(&model, &image, &result);
    std::cout << result << std::endl;
    int age = 0;
    ret = md_face_age_predict(&model, &image, &age);
    std::cout << ret << std::endl;
    std::cout << age << std::endl;
    MDGenderResult r_gender;
    ret = md_face_gender_predict(&model, &image, &r_gender);
    std::cout << ret << std::endl;
    std::cout << r_gender << std::endl;
    MDEyeStateResult r_eye_state;
    ret = md_face_eye_state_predict(&model, &image, &r_eye_state);
    std::cout << ret << std::endl;
    std::cout << r_eye_state.left_eye << std::endl;
    std::cout << r_eye_state.right_eye << std::endl;
    return ret;
}

int test_detection() {
    MDStatusCode ret = MDStatusCode::Success;
    MDModel model;
    if ((ret = md_create_detection_model(&model, "../tests/models/best.onnx", 8)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    MDSize size = {1440, 1440};
    if ((ret = md_set_detection_input_size(&model, size)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    auto im = md_read_image("../tests/test_images/test_detection.png");
    MDDetectionResults result;
    if ((ret = md_detection_predict(&model, &im, &result)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    md_draw_detection_result(&im, &result, "../tests/msyh.ttc", 20, 0.5, 1);
    std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    std::cout << "cost: " << diff.count() << std::endl;
    md_show_image(&im);
    md_print_detection_result(&result);
    md_free_detection_result(&result);
    md_free_image(&im);
    md_free_detection_model(&model);
    return ret;
}

int test_ocr() {
    MDStatusCode ret = MDStatusCode::Success;
    //读取图片
    auto image = md_read_image("../tests/test_images/test_ocr.png");
    // 需要查找的文本
    const char *text = "暂停测试";
    // 创建模型句柄
    MDModel model;
    MDOCRModelParameters parameters = {
            "../tests/models/ocr",
            "../tests/key.txt",
            8,
            PaddlePaddle,
            960,
            0.3,
            0.6,
            1.5,
            "slow",
            0,
            8
    };
    if ((ret = md_create_ocr_model(&model, &parameters)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    // 获取文本目标位置
    MDOCRResults results;
    if ((ret = md_ocr_model_predict(&model, &image, &results)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    MDColor color = {255, 0, 255};
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    md_draw_ocr_result(&image, &results, "../tests/msyh.ttc", 15, &color, 0.5, 0);
    std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    std::cout << "cost: " << diff.count() << std::endl;
    md_print_ocr_result(&results);
    md_free_ocr_result(&results);
    auto rect = md_get_text_position(&model, &image, text);
    // 打印文本信息
    md_print_rect(&rect);
    // 先裁剪再绘制，不然image是指针传递，在绘制时会修改原始image
    auto roi = md_crop_image(&image, &rect);
    // 在原始画面上绘制文本结果

    md_draw_text(&image, &rect, text, "../tests/msyh.ttc", 15, &color, 0.5);
    // 判断按钮是否可用
    auto enable = md_get_button_enable_status(&roi, 50, 0.05);
    std::cout << "enable: " << enable << std::endl;
    // 显示原始画面
    md_show_image(&image);
    // 显示目标文本所在画面
    md_show_image(&roi);

    // 释放资源
    md_free_image(&roi);
    md_free_image(&image);
    md_free_ocr_model(&model);
    return ret;
}


int main() {
    SetConsoleOutputCP(CP_UTF8);

//    test_detection();
//    test_ocr();
    test_face();
}
