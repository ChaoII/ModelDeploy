//#include <iostream>
//#include "src/ocr_capi.h"
//#include "src/detection_capi.h"
//#include "src/utils.h"
//#include <chrono>

#ifdef _WIN32

//#include <Windows.h>

#endif

int main() {
//    SetConsoleOutputCP(CP_UTF8);
    // StatusCode ret = StatusCode::Success;
//    WModel model;
//    if ((ret = create_detection_model(&model, "../tests/models/best.onnx", 8)) != 0) {
//        std::cout << ret << std::endl;
//        return 0;
//    }
//    if ((ret = set_detection_input_size(&model, {1440, 1440})) != 0) {
//        std::cout << ret << std::endl;
//        return 0;
//    }
//    auto im = read_image("../tests/test_images/test_detection.png");
//    WDetectionResults result;
//    if ((ret = detection_predict(&model, im, &result)) != 0) {
//        std::cout << ret << std::endl;
//        return 0;
//    }
//
//    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
//    draw_detection_result(im, &result, "../tests/msyh.ttc", 20, 0.5, 1);
//    std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
//    std::cout << "cost: " << diff.count() << std::endl;
//    show_image(im);
////    print_detection_result(&result);
//    free_detection_result(&result);
//    free_wimage(im);
//    free_detection_model(&model);


    //读取图片
//    auto image = read_image("../tests/test_images/test_ocr.png");
//    // 需要查找的文本
//    const char *text = "暂停测试";
//    // 创建模型句柄
//    WModel model;
//    OCRModelParameters parameters = {
//            "../tests/models/ocr",
//            "../tests/key.txt",
//            8,
//            PaddlePaddle,
//            960,
//            0.3,
//            0.6,
//            1.5,
//            "slow",
//            0,
//            8
//    };
//    if ((ret = create_ocr_model(&model, &parameters)) != 0) {
//        std::cout << ret << std::endl;
//        return 0;
//    }
//    // 获取文本目标位置
//    WOCRResults results;
//    if ((ret = ocr_model_predict(&model, image, &results)) != 0) {
//        std::cout << ret << std::endl;
//        return 0;
//    }
//    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
//    draw_ocr_result(image, &results, "../tests/msyh.ttc", 15, {255, 0, 255}, 0.5, 0);
//    std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
//    std::cout << "cost: " << diff.count() << std::endl;
//    print_ocr_result(&results);
//    free_ocr_result(&results);
//    auto rect = get_text_position(&model, image, text);
////    // 打印文本信息
//    print_rect(rect);
//    // 先裁剪再绘制，不然image是指针传递，在绘制时会修改原始image
//    WImage *roi = crop_image(image, rect);
//    // 在原始画面上绘制文本结果
//    draw_text(image, rect, text, "../tests/msyh.ttc", 15, {255, 0, 255}, 0.5);
//    // 判断按钮是否可用
//    auto enable = get_button_enable_status(roi, 50, 0.05);
//    std::cout << "enable: " << enable << std::endl;
//    // 显示原始画面
//    show_image(image);
//    // 显示目标文本所在画面
//    show_image(roi);
//
//    // 释放资源
//    free_wimage(roi);
//    free_wimage(image);
//    free_ocr_model(&model);
}
