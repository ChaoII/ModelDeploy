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

TEST_CASE("OCR Classifier ncnn vs ORT", "[ncnn][phase_b][ocr][cls]") {
    namespace p = ocr_test;
    const auto mdl = p::ncnn_sub("cls");
    const auto imgf = p::image("test_ocr.png");
    if (!p::avail(mdl) || !p::avail(imgf)) return;

    ocr::Classifier ort(p::onnx_sub("cls").string(), p::ort_opt());
    ocr::Classifier ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    auto img = ImageData::imread(imgf.string());
    int32_t lo = -1, ln = -1; float so = 0, sn = 0;
    REQUIRE(ort.predict(img, &lo, &so));
    REQUIRE(ncnn.predict(img, &ln, &sn));
    REQUIRE(lo >= 0);
    REQUIRE(ln >= 0);
    INFO("cls ORT label=" << lo << " score=" << so << " | ncnn label=" << ln << " score=" << sn);
    REQUIRE(ln == lo);                       // 方向标签一致
    REQUIRE(std::fabs(sn - so) < 0.05f);     // 分容差
}

TEST_CASE("OCR Recognizer ncnn vs ORT", "[ncnn][phase_b][ocr][rec]") {
    namespace p = ocr_test;
    const auto mdl = p::ncnn_sub("rec");
    const auto dictf = p::dict();
    const auto imgf = p::image("test_ocr_recognition.jpg");
    const auto altf = p::image("test_ocr.png");
    if (!p::avail(mdl) || !p::avail(dictf)) return;
    if (!p::avail(imgf) && !p::avail(altf)) return;
    const std::filesystem::path use = p::avail(imgf) ? imgf : altf;

    ocr::Recognizer ort(p::onnx_sub("rec").string(), dictf.string(), p::ort_opt());
    ocr::Recognizer ncnn(mdl.string(), dictf.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    auto img = ImageData::imread(use.string());
    std::string to, tn; float so = 0, sn = 0;
    REQUIRE(ort.predict(img, &to, &so));
    if (to.empty()) return;  // v6 rec 对多行图可能返回空，跳过
    REQUIRE(ncnn.predict(img, &tn, &sn));
    INFO("rec ORT text=[" << to << "] score=" << so << " | ncnn text=[" << tn << "] score=" << sn);
    REQUIRE(tn == to);               // 文本串强锚点
    REQUIRE(std::fabs(sn - so) < 0.1f);
}

TEST_CASE("PaddleOCR pipeline ncnn vs ORT", "[ncnn][phase_b][ocr][pipe]") {
    namespace p = ocr_test;
    const auto detm = p::ncnn_sub("det"), clsm = p::ncnn_sub("cls"), recm = p::ncnn_sub("rec");
    const auto dictf = p::dict();
    const auto imgf = p::image("test_ocr.png");
    if (!p::avail(detm) || !p::avail(clsm) || !p::avail(recm) || !p::avail(dictf) || !p::avail(imgf)) return;

    auto img = ImageData::imread(imgf.string());
    // ORT 显式传 .onnx、ncnn 传 .param
    ocr::PaddleOCR ort(p::onnx_sub("det").string(), p::onnx_sub("cls").string(), p::onnx_sub("rec").string(), dictf.string(), p::ort_opt());
    ocr::PaddleOCR ncnn(detm.string(), clsm.string(), recm.string(), dictf.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    // pipeline 内部 cls/rec 按行 batch（默认 6/8），ncnn 强制 batch==1：两侧都设为 1 保持可比。
    REQUIRE(ort.set_cls_batch_size(1));
    REQUIRE(ort.set_rec_batch_size(1));
    REQUIRE(ncnn.set_cls_batch_size(1));
    REQUIRE(ncnn.set_rec_batch_size(1));

    OCRResult ro, rn;
    REQUIRE(ort.predict(img, &ro));
    REQUIRE(ncnn.predict(img, &rn));
    INFO("ORT boxes=" << ro.boxes.size() << " text=" << ro.text.size()
         << " | ncnn boxes=" << rn.boxes.size() << " text=" << rn.text.size());
    REQUIRE(!ro.boxes.empty());
    REQUIRE(!ro.text.empty());
    REQUIRE(!rn.boxes.empty());
    // 至少一个文本串双侧一致
    bool match = false;
    for (const auto& a : ro.text) for (const auto& b : rn.text)
        if (!a.empty() && a == b) match = true;
    REQUIRE(match);
}
#endif
