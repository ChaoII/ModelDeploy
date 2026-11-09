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
    // 测试数据缺失时返回空 Mat，由调用方跳过（与套件内其它用例的约定一致）。
    cv::Mat read_qr_image() {
        const char* dir = std::getenv("TEST_DATA_DIR");
        std::string path = (dir ? std::string(dir) : std::string(".")) + "/test_data/qr_sample.png";
        return cv::imread(path, cv::IMREAD_COLOR);   // BGR
    }
}

TEST_CASE("BarcodeDetector decodes a sample QR", "[barcode]") {
    BarcodeDetector det;
    cv::Mat raw = read_qr_image();
    if (raw.empty()) {
        WARN("test_data/qr_sample.png not found; skipping barcode test.");
        return;
    }
    ImageData img(raw);
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
    cv::Mat raw = read_qr_image();   // 内容是 QR
    if (raw.empty()) {
        WARN("test_data/qr_sample.png not found; skipping barcode test.");
        return;
    }
    ImageData img(raw);
    auto res = det.detect(img);
    // 限制为 Code128 时不应解码出 QR
    for (auto& r : res) {
        REQUIRE_FALSE(r.is_qr);
    }
}
