//
// Created by aichao on 2025/1/17.
//
#include <iostream>
#include "src/utils/utils_capi.h"

int main() {
    MDImage md_image = md_read_image_from_device(0, 1920, 1080, true);
    md_free_image(&md_image);
}
