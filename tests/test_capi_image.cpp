#include <cstring>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cstdio>

#include "capi/utils/md_image_capi.h"
#include "csrc/utils/utils.h"
#include <catch2/catch_test_macros.hpp>

namespace fs = std::filesystem;

static fs::path test_data_dir() {
    const char* env = std::getenv("TEST_DATA_DIR");
    if (env && *env) return fs::path(env) / "test_data";
    return fs::current_path() / "test_data";
}

struct MDImageGuard {
    MDImage img{};
    bool has_heap_data = false;

    MDImageGuard() = default;
    explicit MDImageGuard(MDImage image) : img(image), has_heap_data(image.data != nullptr) {}
    ~MDImageGuard() {
        if (has_heap_data && img.data) {
            md_free_image(&img);
        }
    }
    MDImageGuard(const MDImageGuard&) = delete;
    MDImageGuard& operator=(const MDImageGuard&) = delete;
};

TEST_CASE("CAPI md_read_image roundtrip", "[capi_image]") {
    auto path = test_data_dir() / "test_images" / "test_person.jpg";
    REQUIRE(fs::exists(path));

    MDImageGuard guard(md_read_image(path.string().c_str()));
    REQUIRE(guard.img.data != nullptr);
    REQUIRE(guard.img.width == 900);
    REQUIRE(guard.img.height == 675);
}

TEST_CASE("CAPI md_clone_image", "[capi_image]") {
    auto path = test_data_dir() / "test_images" / "test_person.jpg";
    REQUIRE(fs::exists(path));

    MDImageGuard src(md_read_image(path.string().c_str()));
    REQUIRE(src.img.data != nullptr);

    MDImage cloned = md_clone_image(&src.img);
    REQUIRE(cloned.data != nullptr);
    REQUIRE(cloned.width == src.img.width);
    REQUIRE(cloned.height == src.img.height);

    cloned.data[0] = 0;
    REQUIRE(src.img.data[0] != 0);

    md_free_image(&cloned);
}

TEST_CASE("CAPI md_crop_image", "[capi_image]") {
    auto path = test_data_dir() / "test_images" / "test_person.jpg";
    REQUIRE(fs::exists(path));

    MDImageGuard src(md_read_image(path.string().c_str()));

    SECTION("crop full image") {
        MDRect full{0, 0, 900, 675};
        MDImage cropped = md_crop_image(&src.img, &full);
        REQUIRE(cropped.data != nullptr);
        REQUIRE(cropped.width == 900);
        REQUIRE(cropped.height == 675);
        md_free_image(&cropped);
    }

    SECTION("crop center region") {
        MDRect center{200, 100, 400, 300};
        MDImage cropped = md_crop_image(&src.img, &center);
        REQUIRE(cropped.data != nullptr);
        REQUIRE(cropped.width == 400);
        REQUIRE(cropped.height == 300);
        md_free_image(&cropped);
    }
}

TEST_CASE("CAPI md_from_bgr24_data", "[capi_image]") {
    int w = 16, h = 12;
    std::vector<unsigned char> bgr(w * h * 3);
    for (int i = 0; i < w * h * 3; ++i) bgr[i] = static_cast<unsigned char>((i * 7) % 256);

    MDImage img = md_from_bgr24_data(bgr.data(), w, h);
    REQUIRE(img.data != nullptr);
    REQUIRE(img.width == w);
    REQUIRE(img.height == h);

    REQUIRE(std::memcmp(img.data, bgr.data(), bgr.size()) == 0);
    md_free_image(&img);
}

TEST_CASE("CAPI md_from_rgb24_data_to_bgr24", "[capi_image]") {
    int w = 4, h = 4;
    std::vector<unsigned char> rgb(w * h * 3);
    for (int i = 0; i < w * h; ++i) {
        rgb[i * 3]     = 255;
        rgb[i * 3 + 1] = 0;
        rgb[i * 3 + 2] = 0;
    }

    MDImage img = md_from_rgb24_data_to_bgr24(rgb.data(), w, h);
    REQUIRE(img.data != nullptr);
    REQUIRE(img.data[0] == 0);
    REQUIRE(img.data[1] == 0);
    REQUIRE(img.data[2] == 255);
    md_free_image(&img);
}

TEST_CASE("CAPI md_from_nv12_data_to_bgr24", "[capi_image]") {
    int w = 4, h = 4;
    std::vector<unsigned char> nv12(w * h + w * h / 2, 128);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            nv12[y * w + x] = static_cast<unsigned char>((x + y) * 16);

    MDImage img = md_from_nv12_data_to_bgr24(nv12.data(), w, h);
    REQUIRE(img.data != nullptr);
    REQUIRE(img.width == w);
    REQUIRE(img.height == h);
    md_free_image(&img);
}

TEST_CASE("CAPI md_from_nv21_data_to_bgr24", "[capi_image]") {
    int w = 4, h = 4;
    std::vector<unsigned char> nv21(w * h + w * h / 2, 128);

    MDImage img = md_from_nv21_data_to_bgr24(nv21.data(), w, h);
    REQUIRE(img.data != nullptr);
    md_free_image(&img);
}

TEST_CASE("CAPI md_from_yuv420p_data_to_bgr24", "[capi_image]") {
    int w = 4, h = 4;
    std::vector<unsigned char> yuv420(w * h + w * h / 2, 128);

    MDImage img = md_from_yuv420p_data_to_bgr24(yuv420.data(), w, h);
    REQUIRE(img.data != nullptr);
    md_free_image(&img);
}

TEST_CASE("CAPI md_free_image null safety", "[capi_image]") {
    SECTION("free nullptr should not crash") {
        md_free_image(nullptr);
    }

    SECTION("free empty struct should not crash") {
        MDImage empty{};
        md_free_image(&empty);
    }
}

TEST_CASE("CAPI md_from_compressed_bytes", "[capi_image]") {
    auto path = test_data_dir() / "test_images" / "test_person.jpg";
    REQUIRE(fs::exists(path));

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    auto size = file.tellg();
    file.seekg(0);
    std::vector<unsigned char> bytes(size);
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    file.close();

    MDImage img = md_from_compressed_bytes(bytes.data(), static_cast<int>(bytes.size()));
    REQUIRE(img.data != nullptr);
    REQUIRE(img.width == 900);
    REQUIRE(img.height == 675);
    md_free_image(&img);
}

TEST_CASE("CAPI md_from_base64_str", "[capi_image]") {
    auto path = test_data_dir() / "test_images" / "test_base64_image.txt";
    REQUIRE(fs::exists(path));

    std::ifstream file(path);
    std::string b64((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    MDImage img = md_from_base64_str(b64.c_str());
    REQUIRE(img.data != nullptr);
    REQUIRE(img.width == 192);
    REQUIRE(img.height == 184);
    md_free_image(&img);
}

TEST_CASE("CAPI md_save_image", "[capi_image]") {
    auto path = test_data_dir() / "test_images" / "test_person.jpg";
    REQUIRE(fs::exists(path));

    MDImageGuard src(md_read_image(path.string().c_str()));

    auto tmp = fs::temp_directory_path() / "md_test_save.jpg";
    md_save_image(&src.img, tmp.string().c_str());
    REQUIRE(fs::exists(tmp));
    REQUIRE(fs::file_size(tmp) > 0);

    MDImageGuard loaded(md_read_image(tmp.string().c_str()));
    REQUIRE(loaded.img.data != nullptr);
    REQUIRE(loaded.img.width == src.img.width);
    REQUIRE(loaded.img.height == src.img.height);

    fs::remove(tmp);
}

TEST_CASE("CAPI chained operations", "[capi_image]") {
    auto path = test_data_dir() / "test_images" / "test_person.jpg";
    REQUIRE(fs::exists(path));

    MDImageGuard src(md_read_image(path.string().c_str()));
    REQUIRE(src.img.data != nullptr);

    MDImage cloned = md_clone_image(&src.img);
    REQUIRE(cloned.data != nullptr);

    MDRect crop_rect{100, 100, 400, 300};
    MDImage cropped = md_crop_image(&cloned, &crop_rect);
    REQUIRE(cropped.data != nullptr);
    REQUIRE(cropped.width == 400);
    REQUIRE(cropped.height == 300);

    md_free_image(&cropped);
    md_free_image(&cloned);
}

TEST_CASE("CAPI md_from_bgr24_data pixel verification", "[capi_image]") {
    int w = 16, h = 12;
    std::vector<unsigned char> expected(w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * 3;
            expected[idx]     = static_cast<unsigned char>(x * 17);
            expected[idx + 1] = static_cast<unsigned char>(y * 21);
            expected[idx + 2] = static_cast<unsigned char>(128);
        }

    MDImage img = md_from_bgr24_data(expected.data(), w, h);
    REQUIRE(img.data != nullptr);
    REQUIRE(std::memcmp(img.data, expected.data(), expected.size()) == 0);
    md_free_image(&img);
}
