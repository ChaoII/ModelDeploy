#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include "csrc/vision/face/insightface/face_analysis.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace phase_b_test {

std::filesystem::path data_root() {
    const char* env = std::getenv("TEST_DATA_DIR");
    if (env && *env) return std::filesystem::path(env);
    return std::filesystem::current_path();
}
bool avail(const std::filesystem::path& p) { return std::filesystem::exists(p); }
RuntimeOption ort_opt() { return RuntimeOption(); }  // 默认 ORT
RuntimeOption ncnn_opt() {
    RuntimeOption o;
    o.use_ncnn_backend();
    o.set_device(Device::CPU, 0);
    return o;
}
std::filesystem::path iface_model(const std::string& sub) {
    return data_root() / "test_data" / "test_models" / "ncnn" / "insightface" / sub / (sub + ".param");
}
std::filesystem::path iface_onnx(const std::string& sub) {
    return data_root() / "test_data" / "test_models" / "onnx" / "insightface" / "buffalo_l" / (sub + ".onnx");
}
std::filesystem::path image(const std::string& name) {
    return data_root() / "test_data" / "test_images" / name;
}

// ORT 与 ncnn 双侧都跑 det，取两侧各自 top 检测框并核对（IoU 最大匹配，容忍排序差异）。
bool box_match(const std::array<float,4>& a, const std::array<float,4>& b) {
    const float x1 = std::max(a[0], b[0]), y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]), y2 = std::min(a[3], b[3]);
    const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    const float ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter;
    return inter / std::max(1e-6f, ua) > 0.5f;
}

}  // namespace phase_b_test

#ifdef ENABLE_NCNN
TEST_CASE("InsightFace det_10g ncnn vs ORT", "[ncnn][phase_b][insightface][det]") {
    namespace p = phase_b_test;
    const auto mdl = p::iface_model("det_10g");
    const auto onx = p::iface_onnx("det_10g");
    const auto imgf = p::image("test_person.jpg");
    if (!p::avail(mdl) || !p::avail(imgf)) return;

    face::InsightFaceDet ort(onx.string(), p::ort_opt());
    face::InsightFaceDet ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    auto img = ImageData::imread(imgf.string());

    std::vector<face::InsightFaceBox> boxes_ort, boxes_ncnn;
    REQUIRE(ort.predict(img, &boxes_ort));
    REQUIRE(ncnn.predict(img, &boxes_ncnn));
    REQUIRE(!boxes_ort.empty());
    REQUIRE(!boxes_ncnn.empty());

    // 两侧 top 检测框应 IoU 匹配（ncnn 输出可能是 2D 同秩，shape()[0] 语义一致）
    REQUIRE(p::box_match(boxes_ort[0].bbox, boxes_ncnn[0].bbox));
    REQUIRE(std::fabs(boxes_ort[0].score - boxes_ncnn[0].score) < 0.05f);
    REQUIRE(boxes_ncnn[0].kps.size() == 5);
}
#endif
