#include "catch2/catch_test_macros.hpp"
#include <opencv2/opencv.hpp>
#include <cstdlib>

#include "vision/common/image_data.h"
#include "vision/barcode/barcode.h"

using namespace modeldeploy::vision;
using namespace modeldeploy::vision::barcode;

namespace {
    // 读取 test_data/qr_sample.png（解码文本 https://example.com/MD），转 ImageData。
    // TEST_DATA_DIR 是 test 运行时注入的环境变量（=仓库根）。
    ImageData read_qr_image() {
        const char* dir = std::getenv("TEST_DATA_DIR");
        std::string path = (dir ? std::string(dir) : std::string(".")) + "/test_data/qr_sample.png";
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);   // BGR
        REQUIRE(!img.empty());
        return ImageData(img);   // ImageData 有 explicit ImageData(const cv::Mat&)，内部转 BGR
    }
}

TEST_CASE("BarcodeDetector decodes a sample QR", "[barcode]") {
    BarcodeDetector det;
    ImageData img = read_qr_image();
    auto res = det.detect(img);
    REQUIRE(!res.empty());
    bool found = false;
    for (auto& r : res) {
        if (r.is_qr && r.text == "https://example.com/MD") { found = true; break; }
    }
    REQUIRE(found);
}

TEST_CASE("BarcodeDetector honors format restriction", "[barcode]") {
    BarcodeDetector det;
    det.set_formats(FMT_CODE_128);   // 只允许 Code128
    ImageData img = read_qr_image(); // 内容是 QR
    auto res = det.detect(img);
    // 限制为 Code128 时不应解码出 QR
    for (auto& r : res) {
        REQUIRE_FALSE(r.is_qr);
    }
}
