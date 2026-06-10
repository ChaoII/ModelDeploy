#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <opencv2/core/mat.hpp>
#include "vision/common/image_data.h"
#include "vision/common/basic_types.h"
#include "vision/common/struct.h"
#include "core/tensor.h"

using namespace modeldeploy;

using namespace modeldeploy::vision;

// Resolutions under test
constexpr int SMALL_W = 320, SMALL_H = 240;
constexpr int MEDIUM_W = 640, MEDIUM_H = 480;
constexpr int LARGE_W = 1920, LARGE_H = 1080;
constexpr int YOLO_W = 640, YOLO_H = 640;
constexpr int YOLO_LARGE_W = 1280, YOLO_LARGE_H = 1280;

static ImageData create_test_image(int w, int h, MdImageType type = MdImageType::PKG_BGR_U8) {
    ImageData img(w, h, type);
    uint8_t* data = img.data();
    for (int i = 0; i < h * w * 3; ++i) {
        data[i] = static_cast<uint8_t>((i * 7 + 13) % 256);
    }
    return img;
}

// =====================================================================
// Group 1: clone
// =====================================================================
TEST_CASE("ImageData clone benchmark", "[benchmark][image_data]") {
    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("clone 320x240") {
            return img.clone();
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("clone 640x480") {
            return img.clone();
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("clone 1920x1080") {
            return img.clone();
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("clone 640x640") {
            return img.clone();
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("clone 1280x1280") {
            return img.clone();
        };
    }
}

// =====================================================================
// Group 2: resize
// =====================================================================
TEST_CASE("ImageData resize benchmark", "[benchmark][image_data]") {
    SECTION("320x240 to various") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("resize 320x240 -> 640x640") {
            return img.resize(640, 640);
        };
    }
    SECTION("640x480 to various") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("resize 640x480 -> 320x320") {
            return img.resize(320, 320);
        };
        BENCHMARK("resize 640x480 -> 640x640") {
            return img.resize(640, 640);
        };
    }
    SECTION("1920x1080 to various") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("resize 1920x1080 -> 640x640") {
            return img.resize(640, 640);
        };
        BENCHMARK("resize 1920x1080 -> 1280x1280") {
            return img.resize(1280, 1280);
        };
    }
    SECTION("640x640 to various") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("resize 640x640 -> 320x320") {
            return img.resize(320, 320);
        };
    }
    SECTION("1280x1280 to various") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("resize 1280x1280 -> 640x640") {
            return img.resize(640, 640);
        };
    }
}

// =====================================================================
// Group 3: cvt_color
// =====================================================================
TEST_CASE("ImageData cvt_color benchmark", "[benchmark][image_data]") {
    SECTION("PKG_BGR_U8 -> PLA_BGR_U8") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H, MdImageType::PKG_BGR_U8);
        BENCHMARK("cvt_color PKG_BGR_U8->PLA_BGR_U8 640x480") {
            return ImageData::cvt_color(img, ColorConvertType::CVT_PL_BGR2PA_BGR);
        };
    }
    {
        auto img = create_test_image(YOLO_W, YOLO_H, MdImageType::PKG_BGR_U8);
        BENCHMARK("cvt_color PKG_BGR_U8->PLA_BGR_U8 640x640") {
            return ImageData::cvt_color(img, ColorConvertType::CVT_PL_BGR2PA_BGR);
        };
    }
    {
        auto img = create_test_image(LARGE_W, LARGE_H, MdImageType::PKG_BGR_U8);
        BENCHMARK("cvt_color PKG_BGR_U8->PLA_BGR_U8 1920x1080") {
            return ImageData::cvt_color(img, ColorConvertType::CVT_PL_BGR2PA_BGR);
        };
    }

    SECTION("PLA_BGR_U8 -> PKG_BGR_U8") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H, MdImageType::PLA_BGR_U8);
        BENCHMARK("cvt_color PLA_BGR_U8->PKG_BGR_U8 640x480") {
            return ImageData::cvt_color(img, ColorConvertType::CVT_PA_BGR2PL_BGR);
        };
    }
    {
        auto img = create_test_image(YOLO_W, YOLO_H, MdImageType::PLA_BGR_U8);
        BENCHMARK("cvt_color PLA_BGR_U8->PKG_BGR_U8 640x640") {
            return ImageData::cvt_color(img, ColorConvertType::CVT_PA_BGR2PL_BGR);
        };
    }

    SECTION("PLA_BGR_U8 -> PLA_RGB_U8") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H, MdImageType::PLA_BGR_U8);
        BENCHMARK("cvt_color PLA_BGR_U8->PLA_RGB_U8 640x480") {
            return ImageData::cvt_color(img, ColorConvertType::CVT_PA_BGR2PA_RGB);
        };
    }
    {
        auto img = create_test_image(YOLO_W, YOLO_H, MdImageType::PLA_BGR_U8);
        BENCHMARK("cvt_color PLA_BGR_U8->PLA_RGB_U8 640x640") {
            return ImageData::cvt_color(img, ColorConvertType::CVT_PA_BGR2PA_RGB);
        };
    }
    {
        auto img = create_test_image(LARGE_W, LARGE_H, MdImageType::PLA_BGR_U8);
        BENCHMARK("cvt_color PLA_BGR_U8->PLA_RGB_U8 1920x1080") {
            return ImageData::cvt_color(img, ColorConvertType::CVT_PA_BGR2PA_RGB);
        };
    }
}

// =====================================================================
// Group 4: normalize
// =====================================================================
TEST_CASE("ImageData normalize benchmark", "[benchmark][image_data]") {
    const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> stddev = {0.229f, 0.224f, 0.225f};

    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("normalize 320x240") {
            return img.normalize(mean, stddev, true, true);
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("normalize 640x480") {
            return img.normalize(mean, stddev, true, true);
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("normalize 1920x1080") {
            return img.normalize(mean, stddev, true, true);
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("normalize 640x640") {
            return img.normalize(mean, stddev, true, true);
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("normalize 1280x1280") {
            return img.normalize(mean, stddev, true, true);
        };
    }
}

// =====================================================================
// Group 5: convert
// =====================================================================
TEST_CASE("ImageData convert benchmark", "[benchmark][image_data]") {
    const std::vector<float> alpha = {1/255.f, 1/255.f, 1/255.f};

    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("convert 320x240") {
            return img.convert(alpha);
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("convert 640x480") {
            return img.convert(alpha);
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("convert 1920x1080") {
            return img.convert(alpha);
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("convert 640x640") {
            return img.convert(alpha);
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("convert 1280x1280") {
            return img.convert(alpha);
        };
    }
}

// =====================================================================
// Group 6: cast
// =====================================================================
TEST_CASE("ImageData cast benchmark", "[benchmark][image_data]") {
    SECTION("PKG_BGR_U8 -> float") {
        SECTION("320x240") {
            auto img = create_test_image(SMALL_W, SMALL_H);
            BENCHMARK("cast u8->float 320x240") {
                return img.cast("float", true);
            };
        }
        SECTION("640x480") {
            auto img = create_test_image(MEDIUM_W, MEDIUM_H);
            BENCHMARK("cast u8->float 640x480") {
                return img.cast("float", true);
            };
        }
        SECTION("1920x1080") {
            auto img = create_test_image(LARGE_W, LARGE_H);
            BENCHMARK("cast u8->float 1920x1080") {
                return img.cast("float", true);
            };
        }
        SECTION("640x640") {
            auto img = create_test_image(YOLO_W, YOLO_H);
            BENCHMARK("cast u8->float 640x640") {
                return img.cast("float", true);
            };
        }
        SECTION("1280x1280") {
            auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
            BENCHMARK("cast u8->float 1280x1280") {
                return img.cast("float", true);
            };
        }
    }
}

// =====================================================================
// Group 7: permute (HWC -> CHW)
// =====================================================================
TEST_CASE("ImageData permute benchmark", "[benchmark][image_data]") {
    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("permute 320x240") {
            return img.permute();
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("permute 640x480") {
            return img.permute();
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("permute 1920x1080") {
            return img.permute();
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("permute 640x640") {
            return img.permute();
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("permute 1280x1280") {
            return img.permute();
        };
    }
}

// =====================================================================
// Group 8: fuse_normalize_and_permute
// =====================================================================
TEST_CASE("ImageData fuse_normalize_and_permute benchmark", "[benchmark][image_data]") {
    const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> stddev = {0.229f, 0.224f, 0.225f};

    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("fuse_normalize_and_permute 320x240") {
            return img.fuse_normalize_and_permute(mean, stddev, true);
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("fuse_normalize_and_permute 640x480") {
            return img.fuse_normalize_and_permute(mean, stddev, true);
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("fuse_normalize_and_permute 1920x1080") {
            return img.fuse_normalize_and_permute(mean, stddev, true);
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("fuse_normalize_and_permute 640x640") {
            return img.fuse_normalize_and_permute(mean, stddev, true);
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("fuse_normalize_and_permute 1280x1280") {
            return img.fuse_normalize_and_permute(mean, stddev, true);
        };
    }
}

// =====================================================================
// Group 9: fuse_convert_and_permute
// =====================================================================
TEST_CASE("ImageData fuse_convert_and_permute benchmark", "[benchmark][image_data]") {
    const std::vector<float> alpha = {1/255.f, 1/255.f, 1/255.f};

    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("fuse_convert_and_permute 320x240") {
            return img.fuse_convert_and_permute(alpha);
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("fuse_convert_and_permute 640x480") {
            return img.fuse_convert_and_permute(alpha);
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("fuse_convert_and_permute 1920x1080") {
            return img.fuse_convert_and_permute(alpha);
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("fuse_convert_and_permute 640x640") {
            return img.fuse_convert_and_permute(alpha);
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("fuse_convert_and_permute 1280x1280") {
            return img.fuse_convert_and_permute(alpha);
        };
    }
}

// =====================================================================
// Group 10: letter_box
// =====================================================================
TEST_CASE("ImageData letter_box benchmark", "[benchmark][image_data]") {
    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("letter_box 320x240 -> 640x640") {
            return img.letter_box({640, 640}, 114.0f);
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("letter_box 640x480 -> 640x640") {
            return img.letter_box({640, 640}, 114.0f);
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("letter_box 1920x1080 -> 640x640") {
            return img.letter_box({640, 640}, 114.0f);
        };
        BENCHMARK("letter_box 1920x1080 -> 1280x1280") {
            return img.letter_box({1280, 1280}, 114.0f);
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("letter_box 640x640 -> 640x640") {
            return img.letter_box({640, 640}, 114.0f);
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("letter_box 1280x1280 -> 640x640") {
            return img.letter_box({640, 640}, 114.0f);
        };
        BENCHMARK("letter_box 1280x1280 -> 1280x1280") {
            return img.letter_box({1280, 1280}, 114.0f);
        };
    }
}

// =====================================================================
// Group 11: center_crop
// =====================================================================
TEST_CASE("ImageData center_crop benchmark", "[benchmark][image_data]") {
    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("center_crop 320x240 -> 224x224") {
            return img.center_crop({224, 224});
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("center_crop 640x480 -> 224x224") {
            return img.center_crop({224, 224});
        };
        BENCHMARK("center_crop 640x480 -> 640x640") {
            return img.center_crop({640, 640});
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("center_crop 1920x1080 -> 224x224") {
            return img.center_crop({224, 224});
        };
        BENCHMARK("center_crop 1920x1080 -> 640x640") {
            return img.center_crop({640, 640});
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("center_crop 640x640 -> 224x224") {
            return img.center_crop({224, 224});
        };
        BENCHMARK("center_crop 640x640 -> 320x320") {
            return img.center_crop({320, 320});
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("center_crop 1280x1280 -> 224x224") {
            return img.center_crop({224, 224});
        };
        BENCHMARK("center_crop 1280x1280 -> 640x640") {
            return img.center_crop({640, 640});
        };
    }
}

// =====================================================================
// Group 12: pad
// =====================================================================
TEST_CASE("ImageData pad benchmark", "[benchmark][image_data]") {
    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("pad 320x240 +10") {
            return img.pad(10, 10, 10, 10, 0.0f);
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("pad 640x480 +10") {
            return img.pad(10, 10, 10, 10, 0.0f);
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("pad 1920x1080 +10") {
            return img.pad(10, 10, 10, 10, 0.0f);
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("pad 640x640 +10") {
            return img.pad(10, 10, 10, 10, 0.0f);
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("pad 1280x1280 +10") {
            return img.pad(10, 10, 10, 10, 0.0f);
        };
    }
}

// =====================================================================
// Group 13: crop
// =====================================================================
TEST_CASE("ImageData crop benchmark", "[benchmark][image_data]") {
    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("crop 320x240 -> 100x100") {
            return img.crop(Rect2f(10.0f, 10.0f, 100.0f, 100.0f));
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("crop 640x480 -> 100x100") {
            return img.crop(Rect2f(50.0f, 50.0f, 100.0f, 100.0f));
        };
        BENCHMARK("crop 640x480 -> 300x300") {
            return img.crop(Rect2f(50.0f, 50.0f, 300.0f, 300.0f));
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("crop 1920x1080 -> 100x100") {
            return img.crop(Rect2f(100.0f, 100.0f, 100.0f, 100.0f));
        };
        BENCHMARK("crop 1920x1080 -> 640x640") {
            return img.crop(Rect2f(100.0f, 100.0f, 640.0f, 640.0f));
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("crop 640x640 -> 100x100") {
            return img.crop(Rect2f(100.0f, 100.0f, 100.0f, 100.0f));
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("crop 1280x1280 -> 100x100") {
            return img.crop(Rect2f(200.0f, 200.0f, 100.0f, 100.0f));
        };
        BENCHMARK("crop 1280x1280 -> 640x640") {
            return img.crop(Rect2f(200.0f, 200.0f, 640.0f, 640.0f));
        };
    }
}

// =====================================================================
// Group 14: rotate (90, 180, 270)
// =====================================================================
TEST_CASE("ImageData rotate benchmark", "[benchmark][image_data]") {
    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("rotate 320x240 90deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_90);
            return cpy;
        };
        BENCHMARK("rotate 320x240 180deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_180);
            return cpy;
        };
        BENCHMARK("rotate 320x240 270deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_270);
            return cpy;
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("rotate 640x480 90deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_90);
            return cpy;
        };
        BENCHMARK("rotate 640x480 180deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_180);
            return cpy;
        };
        BENCHMARK("rotate 640x480 270deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_270);
            return cpy;
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("rotate 1920x1080 90deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_90);
            return cpy;
        };
        BENCHMARK("rotate 1920x1080 180deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_180);
            return cpy;
        };
        BENCHMARK("rotate 1920x1080 270deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_270);
            return cpy;
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("rotate 640x640 90deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_90);
            return cpy;
        };
        BENCHMARK("rotate 640x640 180deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_180);
            return cpy;
        };
        BENCHMARK("rotate 640x640 270deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_270);
            return cpy;
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("rotate 1280x1280 90deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_90);
            return cpy;
        };
        BENCHMARK("rotate 1280x1280 180deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_180);
            return cpy;
        };
        BENCHMARK("rotate 1280x1280 270deg") {
            auto cpy = img.clone();
            cpy.rotate(RotateFlags::ROTATE_270);
            return cpy;
        };
    }
}

// =====================================================================
// Group 15: to_mat / to_tensor
// =====================================================================
TEST_CASE("ImageData to_mat benchmark", "[benchmark][image_data]") {
    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("to_mat 320x240") {
            cv::Mat mat;
            img.to_mat(mat);
            return mat;
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("to_mat 640x480") {
            cv::Mat mat;
            img.to_mat(mat);
            return mat;
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("to_mat 1920x1080") {
            cv::Mat mat;
            img.to_mat(mat);
            return mat;
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("to_mat 640x640") {
            cv::Mat mat;
            img.to_mat(mat);
            return mat;
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("to_mat 1280x1280") {
            cv::Mat mat;
            img.to_mat(mat);
            return mat;
        };
    }
}

TEST_CASE("ImageData to_tensor benchmark", "[benchmark][image_data]") {
    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("to_tensor 320x240") {
            Tensor tensor;
            img.to_tensor(&tensor);
            return tensor;
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("to_tensor 640x480") {
            Tensor tensor;
            img.to_tensor(&tensor);
            return tensor;
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("to_tensor 1920x1080") {
            Tensor tensor;
            img.to_tensor(&tensor);
            return tensor;
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("to_tensor 640x640") {
            Tensor tensor;
            img.to_tensor(&tensor);
            return tensor;
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("to_tensor 1280x1280") {
            Tensor tensor;
            img.to_tensor(&tensor);
            return tensor;
        };
    }
}

// =====================================================================
// Group 16: fuse_resize_and_pad
// =====================================================================
TEST_CASE("ImageData fuse_resize_and_pad benchmark", "[benchmark][image_data]") {
    SECTION("320x240") {
        auto img = create_test_image(SMALL_W, SMALL_H);
        BENCHMARK("fuse_resize_and_pad 320x240 -> 640x640") {
            return img.fuse_resize_and_pad(640, 640, 0, 0, 114.0f);
        };
    }
    SECTION("640x480") {
        auto img = create_test_image(MEDIUM_W, MEDIUM_H);
        BENCHMARK("fuse_resize_and_pad 640x480 -> 640x640") {
            return img.fuse_resize_and_pad(640, 640, 0, 0, 114.0f);
        };
    }
    SECTION("1920x1080") {
        auto img = create_test_image(LARGE_W, LARGE_H);
        BENCHMARK("fuse_resize_and_pad 1920x1080 -> 640x640") {
            return img.fuse_resize_and_pad(640, 640, 0, 0, 114.0f);
        };
        BENCHMARK("fuse_resize_and_pad 1920x1080 -> 1280x1280") {
            return img.fuse_resize_and_pad(1280, 1280, 0, 0, 114.0f);
        };
    }
    SECTION("640x640") {
        auto img = create_test_image(YOLO_W, YOLO_H);
        BENCHMARK("fuse_resize_and_pad 640x640 -> 640x640") {
            return img.fuse_resize_and_pad(640, 640, 0, 0, 114.0f);
        };
    }
    SECTION("1280x1280") {
        auto img = create_test_image(YOLO_LARGE_W, YOLO_LARGE_H);
        BENCHMARK("fuse_resize_and_pad 1280x1280 -> 640x640") {
            return img.fuse_resize_and_pad(640, 640, 0, 0, 114.0f);
        };
        BENCHMARK("fuse_resize_and_pad 1280x1280 -> 1280x1280") {
            return img.fuse_resize_and_pad(1280, 1280, 0, 0, 114.0f);
        };
    }
}
