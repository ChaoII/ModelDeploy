#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include "csrc/vision/face/face_det/scrfd.h"
#include "csrc/vision/face/face_rec/seetaface.h"
#include "csrc/vision/face/face_rec_pipeline/face_rec_pipeline.h"

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

TEST_CASE("SeetaFaceID recognition ncnn vs ORT", "[ncnn][phase_b][seetaface][rec]") {
    namespace p = seetaface_test;
    const auto mdl = p::ncnn_sub("face_recognizer");
    const auto imgf = p::image("test_face_detection.jpg");
    if (!p::avail(mdl) || !p::avail(imgf)) return;
    face::SeetaFaceID ort(p::onnx_sub("face_recognizer").string(), p::ort_opt());
    face::SeetaFaceID ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    auto img = ImageData::imread(imgf.string());
    FaceRecognitionResult ro, rn;
    REQUIRE(ort.predict(img, &ro));
    REQUIRE(ncnn.predict(img, &rn));
    REQUIRE(!ro.embedding.empty());
    REQUIRE(!rn.embedding.empty());
    REQUIRE(ro.embedding.size() == rn.embedding.size());
    float dot = 0, an = 0, bn = 0;
    for (size_t i = 0; i < ro.embedding.size(); ++i) {
        dot += ro.embedding[i] * rn.embedding[i];
        an += ro.embedding[i] * ro.embedding[i];
        bn += rn.embedding[i] * rn.embedding[i];
    }
    float cos = dot / (std::sqrt(an) * std::sqrt(bn) + 1e-9f);
    INFO("rec embedding dim=" << rn.embedding.size() << " cos=" << cos);
    REQUIRE(cos > 0.99f);
}

TEST_CASE("FaceRecognizerPipeline predict_max_face ncnn vs ORT", "[ncnn][phase_b][seetaface][pipe]") {
    namespace p = seetaface_test;
    const auto detm = p::ncnn_sub("scrfd"), recm = p::ncnn_sub("face_recognizer");
    const auto imgf = p::image("test_face_detection.jpg");
    if (!p::avail(detm) || !p::avail(recm) || !p::avail(imgf)) return;
    auto img = ImageData::imread(imgf.string());
    face::FaceRecognizerPipeline ort(p::onnx_sub("scrfd").string(),
                                     p::onnx_sub("face_recognizer").string(), p::ort_opt());
    face::FaceRecognizerPipeline ncnn(detm.string(), recm.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    FaceRecognitionResult ro, rn;
    REQUIRE(ort.predict_max_face(img, &ro));
    REQUIRE(ncnn.predict_max_face(img, &rn));
    REQUIRE(!ro.embedding.empty());
    REQUIRE(!rn.embedding.empty());
    REQUIRE(ro.embedding.size() == rn.embedding.size());
    float dot = 0, an = 0, bn = 0;
    for (size_t i = 0; i < ro.embedding.size(); ++i) {
        dot += ro.embedding[i] * rn.embedding[i];
        an += ro.embedding[i] * ro.embedding[i];
        bn += rn.embedding[i] * rn.embedding[i];
    }
    float cos = dot / (std::sqrt(an) * std::sqrt(bn) + 1e-9f);
    INFO("pipe embedding dim=" << rn.embedding.size() << " cos=" << cos);
    REQUIRE(cos > 0.99f);
}
#endif
