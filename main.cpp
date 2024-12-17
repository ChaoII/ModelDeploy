#include <iostream>
#include "src/ocr_capi.h"
#include "src/detection_capi.h"
#include "src/utils.h"

#ifdef _WIN32

#include <Windows.h>

#endif

int main() {
    SetConsoleOutputCP(CP_UTF8);
    DetectionModelHandle model1 = create_detection_model("best.onnx", 8);
    set_detection_input_size(model1, {1440, 1440});
    auto im = read_image("003.png");
    auto result = detection_predict(model1, im, 1, {255, 255, 0}, 0.5, 1);
    print_detection_result(result);
    free_detection_result(result);
    free_wimage(im);
    free_ocr_model(model1);


//    // 读取图片
//    auto image = read_image("../test_images/shot_image.png");
//    // 需要查找的文本
//    const char *text = "暂停测试";
//    // 创建模型句柄
//    OCRModelHandle model = create_ocr_model("models", "key.txt", 6);
//    // 获取文本目标位置
//    auto rect = get_text_position(model, image, text);
//    // 打印文本信息
//    print_rect(rect);
//    // 先裁剪再绘制，不然image是指针传递，在绘制时会修改原始image
//    WImage *roi = crop_image(image, rect);
//    // 在原始画面上绘制文本结果
//    draw_text(image, rect, text, {255, 0, 255}, 0.5);
//    // 判断按钮是否可用
//    auto enable = get_button_enable_status(roi, 50, 0.05);
//    std::cout << "enable: " << enable << std::endl;
//    // 显示原始画面
//    show_image(image);
//    // 显示目标文本所在画面
////    show_image(roi);
//
//    //释放资源
//    free_wimage(roi);
//    free_wimage(image);
//    free_ocr_model(model);
}
