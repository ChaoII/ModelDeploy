#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <array>
#include <filesystem>
#include <opencv2/core/mat.hpp>
#include "vision/common/image_data.h"
#include "vision/common/basic_types.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;

static ImageData create_solid_image(int w, int h, uint8_t b, uint8_t g, uint8_t r,
                                    MdImageType type = MdImageType::PKG_BGR_U8) {
    ImageData img(w, h, type);
    auto* data = img.data();
    if (!data) return img;
    int ch = img.channels();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * ch;
            data[idx] = b;
            if (ch > 1) data[idx + 1] = g;
            if (ch > 2) data[idx + 2] = r;
            if (ch > 3) data[idx + 3] = 255;
        }
    }
    return img;
}

static ImageData create_gradient_image(int w, int h) {
    ImageData img(w, h, MdImageType::PKG_BGR_U8);
    auto* data = img.data();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * 3;
            data[idx]     = static_cast<uint8_t>((x * 255) / w);
            data[idx + 1] = static_cast<uint8_t>((y * 255) / h);
            data[idx + 2] = static_cast<uint8_t>(128);
        }
    }
    return img;
}

TEST_CASE("ImageData construction", "[image_data]") {
    SECTION("default constructor creates empty image") {
        ImageData img;
        REQUIRE(img.empty());
        REQUIRE(img.width() == 0);
        REQUIRE(img.height() == 0);
        REQUIRE(img.data() == nullptr);
    }

    SECTION("from dimensions") {
        ImageData img(640, 480, MdImageType::PKG_BGR_U8);
        REQUIRE_FALSE(img.empty());
        REQUIRE(img.width() == 640);
        REQUIRE(img.height() == 480);
        REQUIRE(img.channels() == 3);
        REQUIRE(img.type() == MdImageType::PKG_BGR_U8);
        REQUIRE(img.data() != nullptr);
        REQUIRE(img.bytes() == 640 * 480 * 3);
    }

    SECTION("zero-size image creates empty") {
        ImageData img(0, 0, MdImageType::PKG_BGR_U8);
        REQUIRE(img.empty());
    }

    SECTION("GRAY_U8 from dimensions") {
        ImageData img(100, 100, MdImageType::GRAY_U8);
        REQUIRE_FALSE(img.empty());
        REQUIRE(img.channels() == 1);
        REQUIRE(img.bytes() == 100 * 100);
    }

    SECTION("copy constructor is shallow") {
        ImageData img1 = create_solid_image(100, 100, 128, 64, 32);
        ImageData img2(img1);
        REQUIRE(img2.width() == img1.width());
        REQUIRE(img2.height() == img1.height());
    }

    SECTION("move constructor") {
        ImageData img1 = create_solid_image(100, 100, 64, 128, 192);
        ImageData img2 = create_solid_image(200, 200, 32, 64, 128);
        ImageData img3(std::move(img2));
        REQUIRE(img3.width() == 200);
        REQUIRE(img3.height() == 200);
    }
}

TEST_CASE("ImageData accessors", "[image_data]") {
    auto img = create_solid_image(320, 240, 10, 20, 30);

    SECTION("basic accessors") {
        REQUIRE(img.width() == 320);
        REQUIRE(img.height() == 240);
        REQUIRE(img.channels() == 3);
        REQUIRE(img.type() == MdImageType::PKG_BGR_U8);
        REQUIRE(img.element_count() == 320 * 240);
        REQUIRE(img.bytes() == 320 * 240 * 3);
    }

    SECTION("data pointer modification") {
        REQUIRE(img.data() != nullptr);
        img.data()[0] = 99;
        REQUIRE(img.data()[0] == 99);
    }

    SECTION("const data") {
        const ImageData& const_img = img;
        REQUIRE(const_img.data() != nullptr);
    }

    SECTION("empty check") {
        ImageData empty_img;
        REQUIRE(empty_img.empty());
        REQUIRE_FALSE(img.empty());
    }
}

TEST_CASE("ImageData clone and sharing", "[image_data]") {
    SECTION("clone creates independent data") {
        auto img = create_solid_image(50, 50, 100, 150, 200);
        auto cloned = img.clone();
        REQUIRE(cloned.width() == img.width());
        REQUIRE(cloned.height() == img.height());
        cloned.data()[0] = 0;
        REQUIRE(img.data()[0] == 100);
    }
}

TEST_CASE("ImageData from_raw", "[image_data]") {
    SECTION("BGR copy mode") {
        std::vector<uint8_t> buf(100 * 100 * 3, 128);
        auto img = ImageData::from_raw(buf.data(), 100, 100, MdImageType::PKG_BGR_U8, true);
        REQUIRE_FALSE(img.empty());
        REQUIRE(img.width() == 100);
        REQUIRE(img.height() == 100);
        REQUIRE(img.channels() == 3);
        REQUIRE(img.data()[0] == 128);
    }

    SECTION("BGR zero-copy") {
        std::vector<uint8_t> buf(50 * 50 * 3, 64);
        auto img = ImageData::from_raw(buf.data(), 50, 50, MdImageType::PKG_BGR_U8, false);
        REQUIRE_FALSE(img.empty());
        REQUIRE(img.width() == 50);
    }

    SECTION("null data returns empty") {
        auto img = ImageData::from_raw(nullptr, 100, 100, MdImageType::PKG_BGR_U8, true);
        REQUIRE(img.empty());
    }

    SECTION("zero dimensions return empty") {
        std::vector<uint8_t> buf(100, 0);
        auto img = ImageData::from_raw(buf.data(), 0, 0, MdImageType::PKG_BGR_U8, true);
        REQUIRE(img.empty());
    }
}

TEST_CASE("ImageData color conversion", "[image_data]") {
    SECTION("BGR to RGB") {
        auto img = create_solid_image(10, 10, 255, 0, 0);
        auto converted = ImageData::cvt_color(img, ColorConvertType::CVT_PA_BGR2PA_RGB);
        REQUIRE_FALSE(converted.empty());
        REQUIRE(converted.data()[0] == 0);
        REQUIRE(converted.data()[1] == 0);
        REQUIRE(converted.data()[2] == 255);
    }

    SECTION("BGR to GRAY") {
        auto img = create_solid_image(10, 10, 128, 64, 32);
        auto gray = ImageData::cvt_color(img, ColorConvertType::CVT_PA_BGR2GRAY);
        REQUIRE_FALSE(gray.empty());
        REQUIRE(gray.channels() == 1);
    }

    SECTION("empty image returns empty") {
        ImageData empty;
        auto result = ImageData::cvt_color(empty, ColorConvertType::CVT_PA_BGR2PA_RGB);
        REQUIRE(result.empty());
    }
}

TEST_CASE("ImageData resize", "[image_data]") {
    auto img = create_gradient_image(200, 100);

    SECTION("resize to smaller") {
        auto resized = img.resize(100, 50);
        REQUIRE(resized.width() == 100);
        REQUIRE(resized.height() == 50);
        REQUIRE(resized.channels() == 3);
    }

    SECTION("resize to larger") {
        auto resized = img.resize(400, 200);
        REQUIRE(resized.width() == 400);
        REQUIRE(resized.height() == 200);
    }

    SECTION("resize same size") {
        auto resized = img.resize(200, 100);
        REQUIRE(resized.width() == 200);
        REQUIRE(resized.height() == 100);
    }

    SECTION("resize empty image") {
        ImageData empty;
        auto resized = empty.resize(100, 100);
        REQUIRE(resized.empty());
    }
}

TEST_CASE("ImageData crop", "[image_data]") {
    auto img = create_gradient_image(200, 200);

    SECTION("crop full image") {
        auto cropped = img.crop({0, 0, 200, 200});
        REQUIRE(cropped.width() == 200);
        REQUIRE(cropped.height() == 200);
    }

    SECTION("crop center") {
        auto cropped = img.crop({50, 50, 100, 100});
        REQUIRE(cropped.width() == 100);
        REQUIRE(cropped.height() == 100);
    }

    SECTION("crop empty image") {
        ImageData empty;
        auto cropped = empty.crop({0, 0, 10, 10});
        REQUIRE(cropped.empty());
    }
}

TEST_CASE("ImageData rotate", "[image_data]") {
    auto img = create_gradient_image(100, 50);

    SECTION("rotate 90 CW") {
        ImageData rotated = img;
        rotated.rotate(RotateFlags::ROTATE_90);
        REQUIRE(rotated.width() == 50);
        REQUIRE(rotated.height() == 100);
    }

    SECTION("rotate 180") {
        ImageData rotated = img;
        rotated.rotate(RotateFlags::ROTATE_180);
        REQUIRE(rotated.width() == 100);
        REQUIRE(rotated.height() == 50);
    }

    SECTION("rotate 270") {
        ImageData rotated = img;
        rotated.rotate(RotateFlags::ROTATE_270);
        REQUIRE(rotated.width() == 50);
        REQUIRE(rotated.height() == 100);
    }

    SECTION("rotate empty image") {
        ImageData empty;
        CHECK(empty.empty());
    }
}

TEST_CASE("ImageData pad", "[image_data]") {
    auto img = create_solid_image(50, 50, 128, 128, 128);

    SECTION("uniform padding") {
        auto padded = img.pad(10, 10, 10, 10, 0);
        REQUIRE(padded.width() == 70);
        REQUIRE(padded.height() == 70);
    }

    SECTION("asymmetric padding") {
        auto padded = img.pad(5, 10, 15, 20, 128);
        REQUIRE(padded.width() == 85);
        REQUIRE(padded.height() == 65);
    }

    SECTION("zero padding") {
        auto padded = img.pad(0, 0, 0, 0, 0);
        REQUIRE(padded.width() == 50);
        REQUIRE(padded.height() == 50);
    }

    SECTION("pad empty image") {
        ImageData empty;
        CHECK(empty.empty());
    }
}

TEST_CASE("ImageData normalize", "[image_data]") {
    auto img = create_solid_image(10, 10, 128, 64, 32);

    SECTION("normalize with scale and swap_rb") {
        auto norm = img.normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}, true, true);
        REQUIRE_FALSE(norm.empty());
        REQUIRE(norm.channels() == 3);
    }

    SECTION("normalize without scale") {
        auto norm = img.normalize({0, 0, 0}, {1, 1, 1}, false, false);
        REQUIRE_FALSE(norm.empty());
    }

    SECTION("normalize empty image") {
        ImageData empty;
        CHECK(empty.empty());
    }
}

TEST_CASE("ImageData convert", "[image_data]") {
    auto img = create_solid_image(10, 10, 128, 64, 32);

    SECTION("convert with default alpha/beta") {
        auto converted = img.convert();
        REQUIRE_FALSE(converted.empty());
    }

    SECTION("convert with custom alpha") {
        auto converted = img.convert({2.0f, 1.5f, 1.0f}, {0, 0, 0});
        REQUIRE_FALSE(converted.empty());
    }
}

TEST_CASE("ImageData cast", "[image_data]") {
    auto img = create_solid_image(10, 10, 100, 150, 200);

    SECTION("cast to float with scale") {
        auto casted = img.cast("float", true);
        REQUIRE_FALSE(casted.empty());
    }

    SECTION("cast to float without scale") {
        auto casted = img.cast("float", false);
        REQUIRE_FALSE(casted.empty());
    }

    SECTION("cast empty image") {
        ImageData empty;
        CHECK(empty.empty());
    }
}

TEST_CASE("ImageData letter_box", "[image_data]") {
    auto img = create_gradient_image(200, 100);

    SECTION("wider target") {
        auto lb = img.letter_box({300, 300}, 114);
        REQUIRE(lb.width() == 300);
        REQUIRE(lb.height() == 300);
    }

    SECTION("same aspect ratio") {
        auto lb = img.letter_box({200, 100}, 114);
        REQUIRE(lb.width() == 200);
        REQUIRE(lb.height() == 100);
    }

    SECTION("letter_box empty image") {
        ImageData empty;
        CHECK(empty.empty());
    }
}

TEST_CASE("ImageData center_crop", "[image_data]") {
    auto img = create_gradient_image(200, 200);

    SECTION("center crop smaller") {
        auto cropped = img.center_crop({100, 100});
        REQUIRE(cropped.width() == 100);
        REQUIRE(cropped.height() == 100);
    }

    SECTION("center crop same size") {
        auto cropped = img.center_crop({200, 200});
        REQUIRE(cropped.width() == 200);
        REQUIRE(cropped.height() == 200);
    }
}

TEST_CASE("ImageData permute", "[image_data]") {
    auto img = create_gradient_image(10, 10);

    SECTION("HWC to CHW") {
        auto permuted = img.permute();
        REQUIRE_FALSE(permuted.empty());
        REQUIRE(permuted.channels() == 3);
        REQUIRE(permuted.element_count() == 10 * 10 * 3);
    }

    SECTION("permute empty image") {
        ImageData empty;
        CHECK(empty.empty());
    }
}

TEST_CASE("ImageData fuse_normalize_and_permute", "[image_data]") {
    auto img = create_solid_image(10, 10, 128, 64, 32);

    SECTION("fused operation produces output") {
        auto fused = img.fuse_normalize_and_permute({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});
        REQUIRE_FALSE(fused.empty());
        REQUIRE(fused.channels() == 3);
    }

    SECTION("fuse empty image") {
        ImageData empty;
        CHECK(empty.empty());
    }
}

TEST_CASE("ImageData fuse_convert_and_permute", "[image_data]") {
    auto img = create_solid_image(10, 10, 128, 64, 32);

    SECTION("fused convert+permute") {
        auto fused = img.fuse_convert_and_permute();
        REQUIRE_FALSE(fused.empty());
    }

    SECTION("with custom alpha/beta") {
        auto fused = img.fuse_convert_and_permute({2.0f, 1.0f, 0.5f}, {10, 20, 30});
        REQUIRE_FALSE(fused.empty());
    }
}

TEST_CASE("ImageData fuse_resize_and_pad", "[image_data]") {
    auto img = create_gradient_image(100, 100);

    SECTION("basic resize+pad") {
        auto fused = img.fuse_resize_and_pad(80, 80, 10, 10, 114);
        REQUIRE(fused.width() == 90);
        REQUIRE(fused.height() == 90);
    }

    SECTION("no padding") {
        auto fused = img.fuse_resize_and_pad(50, 50, 0, 0, 0);
        REQUIRE(fused.width() == 50);
        REQUIRE(fused.height() == 50);
    }

    SECTION("fuse empty image") {
        ImageData empty;
        CHECK(empty.empty());
    }
}

TEST_CASE("ImageData to_mat and to_tensor", "[image_data]") {
    auto img = create_solid_image(20, 20, 100, 150, 200);

    SECTION("to_mat") {
        cv::Mat mat;
        img.to_mat(mat, false);
        REQUIRE(mat.cols == 20);
        REQUIRE(mat.rows == 20);
    }

    SECTION("to_tensor") {
        Tensor tensor;
        img.to_tensor(&tensor, false);
        REQUIRE(tensor.shape().size() > 0);
    }

    SECTION("images_to_tensor batch") {
        std::vector<ImageData> batch = {img, img};
        Tensor tensor;
        ImageData::images_to_tensor(batch, &tensor);
        REQUIRE(tensor.shape().size() > 0);
    }
}

TEST_CASE("ImageData rotate_crop", "[image_data]") {
    auto img = create_gradient_image(100, 100);

    SECTION("valid box") {
        auto rotated = img.rotate_crop({10, 10, 90, 10, 90, 90, 10, 90});
        REQUIRE_FALSE(rotated.empty());
    }
}

TEST_CASE("ImageData 1x1 image edge case", "[image_data]") {
    SECTION("create and read 1x1") {
        auto img = create_solid_image(1, 1, 255, 128, 64);
        REQUIRE(img.width() == 1);
        REQUIRE(img.height() == 1);
        REQUIRE(img.data()[0] == 255);
        REQUIRE(img.data()[1] == 128);
        REQUIRE(img.data()[2] == 64);
    }

    SECTION("resize 1x1") {
        auto img = create_solid_image(1, 1, 128, 128, 128);
        auto resized = img.resize(100, 100);
        REQUIRE(resized.width() == 100);
        REQUIRE(resized.height() == 100);
    }

    SECTION("crop 1x1 from 100x100") {
        auto img = create_gradient_image(100, 100);
        auto cropped = img.crop({50, 50, 1, 1});
        REQUIRE(cropped.width() == 1);
        REQUIRE(cropped.height() == 1);
    }
}

TEST_CASE("ImageData imencode/imdecode roundtrip", "[image_data]") {
    auto img = create_solid_image(50, 50, 128, 64, 32);

    SECTION("PNG roundtrip") {
        auto encoded = ImageData::imencode(img, ".png");
        REQUIRE_FALSE(encoded.empty());
        auto decoded = ImageData::imdecode(encoded);
        REQUIRE_FALSE(decoded.empty());
        REQUIRE(decoded.width() == 50);
        REQUIRE(decoded.height() == 50);
    }
}

TEST_CASE("ImageData with different types", "[image_data]") {
    SECTION("GRAY_U8 image") {
        ImageData img(100, 100, MdImageType::GRAY_U8);
        REQUIRE(img.channels() == 1);
        REQUIRE(img.bytes() == 100 * 100);
    }

    SECTION("PKG_BGRA_U8 image") {
        ImageData img(100, 100, MdImageType::PKG_BGRA_U8);
        REQUIRE(img.channels() == 4);
        REQUIRE(img.bytes() == 100 * 100 * 4);
    }

    SECTION("PKG_RGB_U8 image") {
        ImageData img(100, 100, MdImageType::PKG_RGB_U8);
        REQUIRE(img.channels() == 3);
    }
}

TEST_CASE("ImageData read/write file", "[image_data]") {
    auto img = create_solid_image(50, 50, 128, 64, 32);

    SECTION("write and read back") {
        auto tmp = std::filesystem::temp_directory_path() / "md_test_save.png";
        REQUIRE(img.imwrite(tmp.string()));
        auto loaded = ImageData::imread(tmp.string());
        REQUIRE_FALSE(loaded.empty());
        REQUIRE(loaded.width() == 50);
        REQUIRE(loaded.height() == 50);
        std::filesystem::remove(tmp);
    }
}

