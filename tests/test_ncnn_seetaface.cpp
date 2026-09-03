#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include "csrc/vision/face/face_det/scrfd.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace seetaface_test {

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
std::filesystem::path ncnn_sub(const std::string& sub) {
    return data_root() / "test_data" / "test_models" / "ncnn" / "seetaface" / sub / (sub + ".param");
}
std::filesystem::path onnx_sub(const std::string& sub) {
    return data_root() / "test_data" / "test_models" / "onnx" / "seetaface" /
           (sub == "scrfd" ? "scrfd_2.5g_bnkps_shape640x640.onnx" : sub + ".onnx");
}
std::filesystem::path image(const std::string& n) {
    return data_root() / "test_data" / "test_images" / n;
}

// 两侧各自 top 检测框核对（IoU 最大匹配，容忍排序差异）
bool box_match(const Rect2f& a, const Rect2f& b) {
    const float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    const float ua = a.width * a.height + b.width * b.height - inter;
    return inter / std::max(1e-6f, ua) > 0.5f;
}

}  // namespace seetaface_test

#ifdef ENABLE_NCNN
TEST_CASE("Scrfd face detection ncnn vs ORT", "[ncnn][phase_b][seetaface][det]") {
    namespace p = seetaface_test;
    const auto mdl = p::ncnn_sub("scrfd");
    const auto imgf = p::image("test_face_detection.jpg");
    if (!p::avail(mdl) || !p::avail(imgf)) return;
    face::Scrfd ort(p::onnx_sub("scrfd").string(), p::ort_opt());
    face::Scrfd ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    auto img = ImageData::imread(imgf.string());
    std::vector<KeyPointsResult> ro, rn;
    REQUIRE(ort.predict(img, &ro));
    REQUIRE(ncnn.predict(img, &rn));
    INFO("ORT faces=" << ro.size() << " ncnn faces=" << rn.size());
    REQUIRE(ro.size() == rn.size());
    REQUIRE(!ro.empty());
    REQUIRE(!rn.empty());
    const KeyPointsResult* best_o = &ro[0];
    const KeyPointsResult* best_n = &rn[0];
    for (const auto& r : ro) if (r.score > best_o->score) best_o = &r;
    for (const auto& r : rn) if (r.score > best_n->score) best_n = &r;
    REQUIRE(p::box_match(best_o->box, best_n->box));
}
#endif
