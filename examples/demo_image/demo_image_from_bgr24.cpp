#include <vector>
#include <cstdio>

#include "../capi_common.h"

int main() {
    const int w = 64, h = 48;
    std::vector<unsigned char> bgr(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t i = (static_cast<size_t>(y) * w + x) * 3;
            bgr[i] = static_cast<unsigned char>(x * 255 / (w - 1));
            bgr[i + 1] = static_cast<unsigned char>(y * 255 / (h - 1));
            bgr[i + 2] = 128;
        }
    }

    MDImageHandle img = nullptr;
    die(md_image_from_bgr24(&img, bgr.data(), w, h), "from_bgr24");
    int ww = 0, hh = 0;
    die(md_image_size(img, &ww, &hh), "size");
    die(md_image_save(img, "capi_bgr24_out.png"), "save");
    md_image_destroy(img);
    std::printf("constructed %dx%d OK -> capi_bgr24_out.png\n", ww, hh);
    return 0;
}
