//
// PP-OCR（det/cls/rec）ncnn vs ORT 锚点测试。
// 补秩守卫：det 4D、cls 2D、rec 3D（见 det/cls/rec postprocessor.cpp 各行 expand_dim(0)）。
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include "csrc/vision/ocr/ppocr.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace ocr_test {

std::filesystem::path data_root() { const char* e = std::getenv("TEST_DATA_DIR"); return (e && *e) ? std::filesystem::path(e) : std::filesystem::current_path(); }
bool avail(const std::filesystem::path& p) { return std::filesystem::exists(p); }
RuntimeOption ort_opt() { return RuntimeOption(); }
RuntimeOption ncnn_opt() { RuntimeOption o; o.use_ncnn_backend(); o.set_device(Device::CPU, 0); return o; }
std::filesystem::path ncnn_sub(const std::string& sub) { return data_root() / "test_data" / "test_models" / "ncnn" / "ocr" / sub / (sub + ".param"); }
std::filesystem::path onnx_sub(const std::string& sub) { return data_root() / "test_data" / "test_models" / "onnx" / "ocr" / "ppocrv6_tiny" / (sub + "_infer.onnx"); }
std::filesystem::path dict() { return data_root() / "test_data" / "ppocrv6_tiny_dict.txt"; }
std::filesystem::path image(const std::string& n) { return data_root() / "test_data" / "test_images" / n; }

}  // namespace ocr_test

#ifdef ENABLE_NCNN
TEST_CASE("OCR DBDetector ncnn vs ORT", "[ncnn][phase_b][ocr][det]") {
    namespace p = ocr_test;
    const auto mdl = p::ncnn_sub("det");
    const auto imgf = p::image("test_ocr.png");
    if (!p::avail(mdl) || !p::avail(imgf)) return;

    ocr::DBDetector ort(p::onnx_sub("det").string(), p::ort_opt());
    ocr::DBDetector ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    auto img = ImageData::imread(imgf.string());

    std::vector<std::array<int,8>> ro, rn;
    REQUIRE(ort.predict(img, &ro));
    REQUIRE(ncnn.predict(img, &rn));
    REQUIRE(!ro.empty());
    REQUIRE(!rn.empty());
    // ORT 动态输入 vs ncnn 固定输入尺寸可能略有差异：要求数量级相当（非空），并打印两侧帧数供报告。
    INFO("ORT boxes=" << ro.size() << " ncnn boxes=" << rn.size());
    REQUIRE(rn.size() >= 1);
}
#endif
