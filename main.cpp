#include <iostream>
#include "src/ocr/ocr_capi.h"
#include "src/detection/detection_capi.h"
#include "src/utils/utils_capi.h"
#include <chrono>
#include <string>


#ifdef _WIN32

#include <Windows.h>

#endif

#ifdef BUILD_FACE

#include "src/face/face_capi.h"

int test_face() {
    MDStatusCode ret;
    MDModel model;
    ret = md_create_face_model(&model, "../tests/models/seetaface", MD_MASK, 1);
    std::cout << "create model result: " << ret << std::endl;
    auto image = md_read_image("../tests/test_images/test_face3.jpg");

    std::cout << "====================face detection==========================" << std::endl;
    MDDetectionResults r_face_detect;
    ret = md_face_detection(&model, &image, &r_face_detect);
    std::cout << "face detection " << (ret ? "failed" : "success") << std::endl;
    std::cout << "face size is: " << r_face_detect.size << std::endl;
    md_draw_detection_result(&image, &r_face_detect, "../tests/msyh.ttc", 20, 0.5, 1);
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

#endif

#ifdef BUILD_ASR

#include "src/asr/asr_capi.h"
#include "src/asr/asr_2pass_capi.h"

int test_asr() {

    MDModel model;
    md_create_asr_model(&model,
                        "D:/funasr-runtime-resources/models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx",
                        "D:/funasr-runtime-resources/models/speech_fsmn_vad_zh-cn-16k-common-onnx",
                        "D:/funasr-runtime-resources/models/punc_ct-transformer_cn-en-common-vocab471067-large-onnx");

    MDASRResult result;
    md_asr_model_predict(&model, "D:/funasr-runtime-resources/vad_example.wav", &result);

//    std::cout << result.msg << std::endl;
//    std::cout << result.stamp << std::endl;
//    std::cout << result.stamp_sents << std::endl;
//    std::cout << result.tpass_msg << std::endl;
//    std::cout << result.snippet_time << std::endl;

    md_free_asr_result(&result);

    md_free_asr_model(&model);

    return 0;

}

int test_tpass() {

    MDModel model;
    md_create_two_pass_model(&model,
                             "D:/funasr-runtime-resources/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx",
                             "D:/funasr-runtime-resources/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx",
                             "D:/funasr-runtime-resources/models/speech_fsmn_vad_zh-cn-16k-common-onnx",
                             "D:/funasr-runtime-resources/models/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx",
                             "D:/funasr-runtime-resources/models/fst_itn_zh",
                             "D:/funasr-runtime-resources/models/speech_ngram_lm_zh-cn-ai-wesp-fst",
                             "",
                             ASRMode::TwoPass);

    MDASRResult result;
    md_two_pass_model_predict(&model, "D:/funasr-runtime-resources/vad_example.wav", &result);

//    std::cout << result.msg << std::endl;
//    std::cout << result.stamp << std::endl;
//    std::cout << result.stamp_sents << std::endl;
//    std::cout << result.tpass_msg << std::endl;
//    std::cout << result.snippet_time << std::endl;

//    md_free_asr_result(&result);

    md_free_two_pass_model(&model);

    return 0;

}


#endif

int test_detection() {
    MDStatusCode ret;
    MDModel model;
    if ((ret = md_create_detection_model(&model, "../tests/models/best.onnx", 4)) != 0) {
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
//    md_show_image(&im);
    md_print_detection_result(&result);
    md_free_detection_result(&result);

    md_free_image(&im);
    md_free_detection_model(&model);
    return ret;
}

int test_ocr() {
    MDStatusCode ret;
    //读取图片
    auto image = md_read_image("../tests/test_images/test_ocr1.png");
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
            4
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
    MDColor color = {0, 0, 255};
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    md_draw_ocr_result(&image, &results, "../tests/msyh.ttc", 15, &color, 0.5, 1);
    std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    std::cout << "cost: " << diff.count() << std::endl;
    md_print_ocr_result(&results);
    md_free_ocr_result(&results);
    auto rect = md_get_text_position(&model, &image, text);
    // 打印文本信息
    md_print_rect(&rect);
    // 先裁剪再绘制，不然image是指针传递，在绘制时会修改原始image
    if (rect.width > 0 && rect.height > 0) {
        auto roi = md_crop_image(&image, &rect);
        // 在原始画面上绘制文本结果
        md_draw_text(&image, &rect, text, "../tests/msyh.ttc", 15, &color, 0.5);
        auto enable = md_get_button_enable_status(&roi, 50, 0.05);
        // 判断按钮是否可用
        std::cout << "enable: " << enable << std::endl;
//        md_show_image(&roi);
        // 释放资源
        md_free_image(&roi);
    }

    // 显示原始画面
//    md_show_image(&image);
    // 显示目标文本所在画面
    md_free_image(&image);
    md_free_ocr_model(&model);
    return ret;
}


int main() {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
//    test_detection();
//    test_ocr();
#ifdef BUILD_FACE
//    test_face();
#endif
#ifdef BUILD_ASR
    test_tpass();
#endif

}
