#include <fstream>
#include <iterator>
#include <string>
#include <cstdio>

#include "../capi_common.h"

int main() {
    MDImageHandle img = nullptr;
    const char* file_path = "../../test_data/test_images/test_base64_image.txt";
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::fprintf(stderr, "Failed to open file: %s\n", file_path);
        return 1;
    }
    const std::string b64((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    die(md_image_from_base64(&img, b64.c_str()), "from_base64");
    int w = 0, h = 0;
    die(md_image_size(img, &w, &h), "size");
    die(md_image_save(img, "capi_base64_out.png"), "save");
    md_image_destroy(img);
    std::printf("decoded %dx%d OK -> capi_base64_out.png\n", w, h);
    return 0;
}
