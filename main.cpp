#include <iostream>
#include "wrzs_capi.h"
#include "utils.h"

#ifdef _WIN32

#include <Windows.h>

#endif

int main() {
    SetConsoleOutputCP(CP_UTF8);
    auto image = read_image("../test_images/shot_image.png");
    auto image_bak = clone_image(image);
    const char *text = "暂停测试";
    OCRModelHandle model = create_ocr_model("models", "key.txt", 6);
    auto rect = get_text_position(model, image, text);
    print_rect(rect);
    draw_text(image, rect, text, {255, 0, 255}, 0.5);
    WImage *roi = crop_image(image_bak, rect);
    auto enable = get_button_enable_status(roi, 50, 0.05);

    std::cout << "enable: " << enable << std::endl;

    show_image(roi);
    free_ocr_model(model);
}
