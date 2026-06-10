#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "vision/common/image_data.h"
#include "vision/common/processors/yolo_preproc.h"
#include "core/tensor.h"

#ifdef WITH_GPU
#include <cuda_runtime.h>
#include "vision/common/processors/yolo_preproc.cuh"
#endif

using namespace modeldeploy::vision;
using namespace modeldeploy;

namespace {
    const std::vector<float> kMean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> kStd = {0.229f, 0.224f, 0.225f};
    const std::vector<float> kScale255 = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    const std::vector<float> kZero = {0.0f, 0.0f, 0.0f};
    const float kPadVal = 114.0f;

    ImageData create_test_image(int w, int h) {
        ImageData img(w, h, MdImageType::PKG_BGR_U8);
        uint8_t* data = img.data();
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int idx = (y * w + x) * 3;
                data[idx] = static_cast<uint8_t>((x * 255) / w);
                data[idx + 1] = static_cast<uint8_t>((y * 255) / h);
                data[idx + 2] = static_cast<uint8_t>(128);
            }
        }
        return img;
    }

    ImageData create_nv12_image(int w, int h) {
        size_t y_size = static_cast<size_t>(w) * h;
        size_t uv_size = y_size / 2;
        std::vector<uint8_t> buf(y_size + uv_size);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                buf[y * w + x] = static_cast<uint8_t>((x + y) * 255 / (w + h));
            }
        }
        for (int y = 0; y < h / 2; ++y) {
            for (int x = 0; x < w / 2; ++x) {
                int uv_idx = static_cast<int>(y_size + y * w + x * 2);
                buf[uv_idx] = 128;
                buf[uv_idx + 1] = 128;
            }
        }
        return ImageData::from_raw(buf.data(), w, h, MdImageType::NV12, true);
    }

    std::vector<ImageData> make_batch(const ImageData& img, int n) {
        std::vector<ImageData> batch;
        batch.reserve(n);
        for (int i = 0; i < n; ++i) {
            batch.push_back(img.clone());
        }
        return batch;
    }
}

TEST_CASE("Individual operation benchmarks (640x640)", "[benchmark][yolo]") {
    const auto img = create_test_image(640, 640);

    BENCHMARK("resize 640x640 to 320x320") {
        auto r = img.resize(320, 320);
        return r.width();
    };

    BENCHMARK("resize 640x640 to 640x640") {
        auto r = img.resize(640, 640);
        return r.width();
    };

    BENCHMARK("resize 640x640 to 1280x1280") {
        auto r = img.resize(1280, 1280);
        return r.width();
    };

    BENCHMARK("convert (1/255 scale)") {
        auto r = img.convert(kScale255, kZero);
        return r.channels();
    };

    BENCHMARK("normalize (ImageNet mean/std)") {
        auto r = img.normalize(kMean, kStd, true, true);
        return r.channels();
    };

    BENCHMARK("permute (HWC to CHW)") {
        auto r = img.permute();
        return r.channels();
    };

    BENCHMARK("fuse_normalize_and_permute") {
        auto r = img.fuse_normalize_and_permute(kMean, kStd, true);
        return r.channels();
    };
}

TEST_CASE("Full pipeline benchmarks (CPU)", "[benchmark][yolo]") {
    const auto img = create_test_image(640, 640);

    BENCHMARK("Step-by-step: resize -> convert -> normalize -> permute") {
        auto r = img.resize(640, 640)
                     .convert(kScale255, kZero)
                     .normalize(kMean, kStd, false, true)
                     .permute();
        return r.channels();
    };

    BENCHMARK("Fused: resize -> fuse_normalize_and_permute") {
        auto r = img.resize(640, 640)
                     .fuse_normalize_and_permute(kMean, kStd, true);
        return r.channels();
    };

    BENCHMARK("LetterBox + normalize + permute") {
        auto r = img.letter_box({640, 640}, kPadVal)
                     .fuse_normalize_and_permute(kMean, kStd, true);
        return r.channels();
    };

    BENCHMARK("yolo_preprocess_cpu (letterbox + norm + CHW)") {
        Tensor output;
        LetterBoxRecord record;
        yolo_preprocess_cpu(img, &output, {640, 640}, kPadVal, &record);
        return output.size();
    };
}

TEST_CASE("Multi-resolution comparison", "[benchmark][yolo]") {
    const auto img_320 = create_test_image(320, 320);
    const auto img_640 = create_test_image(640, 640);
    const auto img_1280 = create_test_image(1280, 1280);

    BENCHMARK("Full pipeline at 320x320 (YOLO-nano)") {
        auto r = img_320.resize(320, 320)
                        .fuse_normalize_and_permute(kMean, kStd, true);
        return r.channels();
    };

    BENCHMARK("Full pipeline at 640x640 (YOLO-medium)") {
        auto r = img_640.resize(640, 640)
                        .fuse_normalize_and_permute(kMean, kStd, true);
        return r.channels();
    };

    BENCHMARK("Full pipeline at 1280x1280 (YOLO-large)") {
        auto r = img_1280.resize(1280, 1280)
                         .fuse_normalize_and_permute(kMean, kStd, true);
        return r.channels();
    };
}

TEST_CASE("Multi-batch benchmarks (NHWC)", "[benchmark][yolo]") {
    const auto img = create_test_image(640, 640);
    const auto batch1 = make_batch(img, 1);
    const auto batch4 = make_batch(img, 4);
    const auto batch8 = make_batch(img, 8);
    const auto batch16 = make_batch(img, 16);

    Tensor t1, t4, t8, t16;

    BENCHMARK("images_to_tensor NHWC batch 1") {
        ImageData::images_to_tensor(batch1, &t1);
        return t1.size();
    };

    BENCHMARK("images_to_tensor NHWC batch 4") {
        ImageData::images_to_tensor(batch4, &t4);
        return t4.size();
    };

    BENCHMARK("images_to_tensor NHWC batch 8") {
        ImageData::images_to_tensor(batch8, &t8);
        return t8.size();
    };

    BENCHMARK("images_to_tensor NHWC batch 16") {
        ImageData::images_to_tensor(batch16, &t16);
        return t16.size();
    };
}

TEST_CASE("Multi-batch benchmarks (NCHW)", "[benchmark][yolo]") {
    const auto pkg_img = create_test_image(640, 640);
    const auto pla_img = ImageData::cvt_color(pkg_img, ColorConvertType::CVT_PA_BGR2PL_BGR);
    const auto batch1 = make_batch(pla_img, 1);
    const auto batch4 = make_batch(pla_img, 4);
    const auto batch8 = make_batch(pla_img, 8);
    const auto batch16 = make_batch(pla_img, 16);

    Tensor t1, t4, t8, t16;

    BENCHMARK("images_to_tensor NCHW batch 1") {
        ImageData::images_to_tensor(batch1, &t1);
        return t1.size();
    };

    BENCHMARK("images_to_tensor NCHW batch 4") {
        ImageData::images_to_tensor(batch4, &t4);
        return t4.size();
    };

    BENCHMARK("images_to_tensor NCHW batch 8") {
        ImageData::images_to_tensor(batch8, &t8);
        return t8.size();
    };

    BENCHMARK("images_to_tensor NCHW batch 16") {
        ImageData::images_to_tensor(batch16, &t16);
        return t16.size();
    };
}

TEST_CASE("Color conversion benchmarks", "[benchmark][yolo]") {
    const auto bgr_img = create_test_image(640, 640);
    const auto nv12_img = create_nv12_image(640, 640);

    BENCHMARK("BGR to RGB (cvt_color)") {
        auto r = ImageData::cvt_color(bgr_img, ColorConvertType::CVT_PA_BGR2PA_RGB);
        return r.channels();
    };

    BENCHMARK("NV12 to BGR (cvt_color)") {
        auto r = ImageData::cvt_color(nv12_img, ColorConvertType::CVT_NV122PA_BGR);
        return r.channels();
    };
}

TEST_CASE("Cross-format benchmarks", "[benchmark][yolo]") {
    const auto pkg_img = create_test_image(640, 640);
    const auto pla_img = ImageData::cvt_color(pkg_img, ColorConvertType::CVT_PA_BGR2PL_BGR);
    const auto pkg_batch = make_batch(pkg_img, 4);
    const auto pla_batch = make_batch(pla_img, 4);

    Tensor tpkg, tpla;

    BENCHMARK("PKG_BGR_U8 to PLA_BGR_U8 conversion") {
        auto r = ImageData::cvt_color(pkg_img, ColorConvertType::CVT_PA_BGR2PL_BGR);
        return r.channels();
    };

    BENCHMARK("images_to_tensor on PKG_BGR_U8") {
        ImageData::images_to_tensor(pkg_batch, &tpkg);
        return tpkg.size();
    };

    BENCHMARK("images_to_tensor on PLA_BGR_U8") {
        ImageData::images_to_tensor(pla_batch, &tpla);
        return tpla.size();
    };

    BENCHMARK("resize on PKG_BGR_U8") {
        auto r = pkg_img.resize(320, 320);
        return r.width();
    };

    BENCHMARK("resize on PLA_BGR_U8") {
        auto r = pla_img.resize(320, 320);
        return r.width();
    };
}

#ifdef WITH_GPU
TEST_CASE("GPU preprocessing benchmarks", "[benchmark][yolo]") {
    const auto img = create_test_image(640, 640);

    BENCHMARK("yolo_preprocess_cuda (letterbox + norm + CHW)") {
        Tensor output;
        LetterBoxRecord record;
        yolo_preprocess_cuda(img, &output, {640, 640}, kPadVal, &record, nullptr);
        return output.size();
    };

    BENCHMARK("yolo_preprocess_bgr_cuda (raw BGR)") {
        Tensor output;
        LetterBoxRecord record;
        yolo_preprocess_bgr_cuda(img.data(), {img.width(), img.height()},
                                 &output, {640, 640}, kPadVal, &record, nullptr);
        return output.size();
    };

    BENCHMARK("yolo_preprocess_nv12_cuda (raw NV12)") {
        auto nv12 = create_nv12_image(640, 640);
        const auto* data = nv12.data();
        const int w = 640, h = 640;
        Tensor output;
        LetterBoxRecord record;
        yolo_preprocess_nv12_cuda(data, data + static_cast<size_t>(w) * h,
                                  {w, h}, w, w,
                                  &output, {w, h}, kPadVal, &record, nullptr);
        return output.size();
    };
}
#endif
