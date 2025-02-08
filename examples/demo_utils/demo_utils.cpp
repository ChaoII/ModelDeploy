//
// Created by aichao on 2025/1/17.
//
#include "csrc/model_deploy.h"

int main() {
    // MDImage md_image = md_read_image_from_device(0, 1920, 1080, true);
    // md_show_image(&md_image);
    // md_free_image(&md_image);
    MDImage md_image = md_read_image("../tests/test_images/test_detection.png");

    auto rect = MDRect{0, 0, 500, 500};
    auto r = md_crop_image(&md_image, &rect);

    md_show_image(&r);
    md_free_image(&md_image);

    // md_crop_image(&md_image, 0, 0, 100, 100)
}
