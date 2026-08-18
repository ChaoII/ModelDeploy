#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstring>
#include <filesystem>
#include <memory>
#include <opencv2/core/mat.hpp>
#include "vision/common/image_data.h"
#include "vision/common/basic_types.h"
#include "vision/processors/cpu/cpu_processor_backend.h"
#include "core/md_log.h"

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
        MD_SET_LOG_LEVEL(LogLevel::MD_LOG_F); // suppress expected error
        auto img = ImageData::from_raw(nullptr, 100, 100, MdImageType::PKG_BGR_U8, true);
        REQUIRE(img.empty());
        MD_SET_LOG_LEVEL(LogLevel::MD_LOG_I);
    }

    SECTION("zero dimensions return empty") {
        MD_SET_LOG_LEVEL(LogLevel::MD_LOG_F); // suppress expected error
        std::vector<uint8_t> buf(100, 0);
        auto img = ImageData::from_raw(buf.data(), 0, 0, MdImageType::PKG_BGR_U8, true);
        REQUIRE(img.empty());
        MD_SET_LOG_LEVEL(LogLevel::MD_LOG_I);
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

TEST_CASE("CPU NV12 device-style drawing", "[image_data]") {
    const int w = 320, h = 240;
    // 构造 host NV12 帧（Y 平面 + UV 交错平面），初始化为纯灰（Y=128, Cb=Cr=128）
    auto y_buf = std::make_unique<uint8_t[]>(static_cast<size_t>(w) * h);
    auto uv_buf = std::make_unique<uint8_t[]>(static_cast<size_t>(w) * (h / 2));
    std::memset(y_buf.get(), 128, static_cast<size_t>(w) * h);
    std::memset(uv_buf.get(), 128, static_cast<size_t>(w) * (h / 2));
    ImageData frame = ImageData::from_device_planes(y_buf.get(), uv_buf.get(), w, h, w, w, Device::CPU);

    CpuProcessorBackend backend;

    SECTION("draw_rect writes Y and UV planes") {
        REQUIRE(backend.draw_rect_nv12(frame, 20, 20, 100, 60, 255, 0, 0, 2));
        const uint8_t* y = frame.y();
        // 顶边中间像素（x=60,y=20）：画成红色 → 亮度低
        REQUIRE(y[20 * w + 60] != 128);
        // 边框外像素保持灰
        REQUIRE(y[20 * w + 5] == 128);
        // UV 平面也被写（矩形内某 UV 像素偏离灰平衡 128）
        const uint8_t* uv = frame.uv();
        bool uv_changed = false;
        for (int uy = 10; uy < 40; ++uy) {
            for (int ux = 10; ux < 60; ++ux) {
                if (uv[uy * w + ux * 2] != 128 || uv[uy * w + ux * 2 + 1] != 128) { uv_changed = true; break; }
            }
        }
        REQUIRE(uv_changed);
    }

    SECTION("draw_text writes label pixels") {
        REQUIRE(backend.draw_text_nv12(frame, 30, 100, "AB", 255, 255, 255, 1));
        const uint8_t* y = frame.y();
        bool any_changed = false;
        for (int py = 100; py < 116; ++py) {
            for (int px = 30; px < 30 + 16; ++px) {
                if (y[py * w + px] != 128) { any_changed = true; break; }
            }
        }
        REQUIRE(any_changed);
    }

    SECTION("draw_points writes") {
        std::vector<Point3f> pts = {Point3f(50, 50, 0)};
        REQUIRE(backend.draw_points_nv12(frame, pts, 0, 255, 0, 3));
        REQUIRE(frame.y()[50 * w + 50] != 128);
    }

    SECTION("draw_polygon draws closed loop") {
        std::vector<Point2f> pts = {Point2f(10, 10), Point2f(90, 10), Point2f(90, 70)};
        REQUIRE(backend.draw_polygon_nv12(frame, pts, 0, 0, 255, 2));
        const uint8_t* y = frame.y();
        // 顶点处应有像素
        REQUIRE(y[10 * w + 10] != 128);
    }
}

TEST_CASE("image_data: device frame self-describes (w/h not zeroed)", "[core]") {
    std::vector<unsigned char> y(128 * 96, 100);
    std::vector<unsigned char> uv(128 * 48, 100);
    auto img = modeldeploy::vision::ImageData::from_device_planes(
        y.data(), uv.data(), 128, 96, 128, 128, Device::CPU);
    REQUIRE(!img.empty());
    CHECK(img.width() == 128);
    CHECK(img.height() == 96);
    CHECK(img.format() == MdImageType::NV12);
    CHECK(img.device() == Device::CPU);
    CHECK(img.plane_count() == 2);
    CHECK(img.plane(0).data == y.data());
    CHECK(img.plane(1).data == uv.data());
    CHECK(img.plane(0).step == 128);
    CHECK(img.plane(1).step == 128);
}

TEST_CASE("image_data: from_bgr24 builds CPU single-plane", "[core]") {
    std::vector<unsigned char> bgr(10 * 8 * 3, 42);
    auto img = modeldeploy::vision::ImageData::from_bgr24(bgr.data(), 10, 8);
    REQUIRE(!img.empty());
    CHECK(img.width() == 10);
    CHECK(img.height() == 8);
    CHECK(img.format() == MdImageType::PKG_BGR_U8);
    CHECK(img.device() == Device::CPU);
    CHECK(img.plane_count() == 1);
    CHECK(img.plane(0).data == bgr.data());
}

TEST_CASE("image_data: asMat cpu borrow / toCpu device copy", "[core]") {
    std::vector<unsigned char> bgr(6 * 4 * 3, 7);
    auto cpu = modeldeploy::vision::ImageData::from_bgr24(bgr.data(), 6, 4);
    cv::Mat m;
    REQUIRE(cpu.asMat(&m));                 // CPU 借用
    CHECK(!m.empty());
    std::vector<unsigned char> y(8 * 4, 0), uv(8 * 2, 0);
    auto dev = modeldeploy::vision::ImageData::from_device_planes(y.data(), uv.data(), 8, 4, 8, 8, Device::CPU);
    modeldeploy::vision::ImageData cpu_copy;
    REQUIRE(dev.toCpu(&cpu_copy));          // 平面→CPU 深拷贝
    CHECK(cpu_copy.width() == 8);
    CHECK(cpu_copy.height() == 4);
    modeldeploy::vision::ImageData gpu;
    CHECK_FALSE(gpu.asMat(&m));             // 空图失败
    CHECK(modeldeploy::vision::ImageData::last_error() != nullptr);
}

TEST_CASE("image_data: refresh_meta preserves device dims", "[core]") {
    std::vector<unsigned char> y(8 * 4, 0), uv(8 * 2, 0);
    auto dev = modeldeploy::vision::ImageData::from_device_planes(y.data(), uv.data(), 8, 4, 8, 8, Device::CPU);
    // 内部 refresh_meta 不再清空设备帧宽高（对设备帧安全）
    CHECK(dev.width() == 8);
    CHECK(dev.height() == 4);
    // 走到 refresh_meta 的就地操作后设备帧宽高仍须保留（旧逻辑会在此清零 → 失败）
    ImageData rotated = dev; // 浅拷贝，共享 impl_
    rotated.rotate(RotateFlags::ROTATE_90);
    CHECK(rotated.width() == 8);
    CHECK(rotated.height() == 4);
}

TEST_CASE("image_data: from_bgr24 is usable by OpenCV members", "[core]") {
    std::vector<unsigned char> bgr(10 * 8 * 3, 0);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 10; ++x) {
            bgr[(y * 10 + x) * 3 + 0] = static_cast<unsigned char>(x);
            bgr[(y * 10 + x) * 3 + 1] = static_cast<unsigned char>(y);
            bgr[(y * 10 + x) * 3 + 2] = 128;
        }
    auto img = modeldeploy::vision::ImageData::from_bgr24(bgr.data(), 10, 8);
    REQUIRE(!img.empty());

    auto resized = img.resize(5, 4);
    CHECK(resized.width() == 5);
    CHECK(resized.height() == 4);

    ImageData rotated = img;
    rotated.rotate(RotateFlags::ROTATE_90);
    CHECK(rotated.width() == 8);
    CHECK(rotated.height() == 10);

    auto gray = modeldeploy::vision::ImageData::cvt_color(img, ColorConvertType::CVT_PA_BGR2GRAY);
    CHECK_FALSE(gray.empty());
    CHECK(gray.channels() == 1);

    auto encoded = modeldeploy::vision::ImageData::imencode(img, ".png");
    CHECK_FALSE(encoded.empty());
}

TEST_CASE("image_data: crop/rotate/cvt_color dispatch via CPU backend", "[core]") {
    std::vector<unsigned char> bgr(10 * 8 * 3, 0);
    for (int i = 0; i < 10 * 8; ++i) bgr[i * 3] = static_cast<unsigned char>(i % 256);
    auto img = modeldeploy::vision::ImageData::from_bgr24(bgr.data(), 10, 8);
    // crop
    auto c = img.crop({2, 2, 4, 4});
    REQUIRE(!c.empty());
    CHECK(c.width() == 4);
    CHECK(c.height() == 4);
    // rotate
    auto r = img.clone().rotate(ROTATE_90);
    CHECK(r.width() == 8);
    CHECK(r.height() == 10);
    // cvt_color BGR->GRAY
    auto g = modeldeploy::vision::ImageData::cvt_color(img, ColorConvertType::CVT_PA_BGR2GRAY);
    REQUIRE(!g.empty());
    CHECK(g.format() == MdImageType::GRAY_U8);
}

TEST_CASE("image_data: device-frame op not supported errors (no silent cpu)", "[core]") {
    std::vector<unsigned char> y(8 * 4, 0), uv(8 * 2, 0);
    auto dev = modeldeploy::vision::ImageData::from_device_planes(
        y.data(), uv.data(), 8, 4, 8, 8, Device::GPU);
    modeldeploy::vision::ImageData::last_error();           // 先清空
    auto c = dev.crop({0, 0, 2, 2});
    CHECK(c.empty());                                       // 设备帧未实现 → 空
    CHECK(modeldeploy::vision::ImageData::last_error() != nullptr);  // 且报错
}

