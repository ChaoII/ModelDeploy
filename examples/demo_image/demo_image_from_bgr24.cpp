//
// Created by aichao on 2025/3/18.
//

#include <fstream>
#include <iostream>
#include <filesystem>
#include "capi/utils/md_image_capi.h"

int main(int argc, char** argv) {
    //读取base64_image.txt文件

    auto file_path = "../../test_data/test_images/test_face_as_second.jpg";
    MDImage image = md_read_image(file_path);
    MDImage image1 = md_from_bgr24_data(image.data, image.width, image.height);
    md_save_image(&image1, "s.jpg");
    md_show_image(&image1);
    md_free_image(&image1);
    md_free_image(&image);
}
