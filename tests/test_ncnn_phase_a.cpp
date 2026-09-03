#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <filesystem>
#include <vector>
#include "runtime/runtime_option.h"
#include "vision/classification/classification.h"
#include "vision/obb/ultralytics_obb.h"
#include "vision/pose/ultralytics_pose.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace phase_a_test {

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
std::filesystem::path ncnn_model(const std::string& name) {
    return data_root() / "test_data" / "test_models" / "ncnn" / name / (name + ".param");
}
std::filesystem::path onnx_model(const std::string& rel) {
    return data_root() / "test_data" / "test_models" / rel;
}
std::filesystem::path image(const std::string& name) {
    return data_root() / "test_data" / "test_images" / name;
}

}  // namespace phase_a_test

#ifdef ENABLE_NCNN
TEST_CASE("Classification ncnn vs ORT", "[ncnn][phase_a][cls]") {
    namespace p = phase_a_test;
    using namespace modeldeploy::vision::classification;
    const auto mdl = p::ncnn_model("yolo26n-cls");
    const auto imgf = p::image("test_person.jpg");
    const auto onx = p::onnx_model("yolo26n-cls.onnx");
    if (!p::avail(mdl) || !p::avail(imgf) || !p::avail(onx)) return;

    Classification ort(onx.string(), p::ort_opt());
    Classification ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    auto img = ImageData::imread(imgf.string());
    ClassifyResult r_ort, r_ncnn;
    REQUIRE(ort.predict(img, &r_ort));
    REQUIRE(ncnn.predict(img, &r_ncnn));
    REQUIRE(!r_ort.label_ids.empty());
    REQUIRE(!r_ncnn.label_ids.empty());
    // 基线锚点: top label 一致
    REQUIRE(r_ncnn.label_ids[0] == r_ort.label_ids[0]);
}

TEST_CASE("UltralyticsObb ncnn vs ORT", "[ncnn][phase_a][obb]") {
    namespace p = phase_a_test;
    const auto mdl = p::ncnn_model("yolo26n-obb");
    const auto imgf = p::image("test_obb.jpg");
    const auto onx = p::onnx_model("yolo26n-obb.onnx");
    if (!p::avail(mdl) || !p::avail(imgf) || !p::avail(onx)) return;
    modeldeploy::vision::detection::UltralyticsObb ort(onx.string(), p::ort_opt());
    modeldeploy::vision::detection::UltralyticsObb ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE((ort.is_initialized() && ncnn.is_initialized()));
    ort.get_preprocessor().set_size({1024, 1024});
    ncnn.get_preprocessor().set_size({1024, 1024});
    auto img = ImageData::imread(imgf.string());
    std::vector<modeldeploy::vision::ObbResult> r_ort, r_ncnn;
    REQUIRE(ort.predict(img, &r_ort, nullptr));
    REQUIRE(ncnn.predict(img, &r_ncnn, nullptr));
    REQUIRE((!r_ort.empty() && !r_ncnn.empty()));
    REQUIRE(r_ncnn[0].label_id == r_ort[0].label_id);
    REQUIRE(r_ncnn[0].score > 0.5f);
}

TEST_CASE("UltralyticsPose ncnn vs ORT", "[ncnn][phase_a][pose]") {
    namespace p = phase_a_test;
    const auto mdl = p::ncnn_model("yolo26n-pose");
    const auto imgf = p::image("test_person.jpg");
    const auto onx = p::onnx_model("yolo26n-pose.onnx");
    if (!p::avail(mdl) || !p::avail(imgf) || !p::avail(onx)) return;
    modeldeploy::vision::detection::UltralyticsPose ort(onx.string(), p::ort_opt());
    modeldeploy::vision::detection::UltralyticsPose ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE((ort.is_initialized() && ncnn.is_initialized()));
    auto img = ImageData::imread(imgf.string());
    std::vector<modeldeploy::vision::KeyPointsResult> r_ort, r_ncnn;
    REQUIRE(ort.predict(img, &r_ort, nullptr));
    REQUIRE(ncnn.predict(img, &r_ncnn, nullptr));
    REQUIRE((!r_ort.empty() && !r_ncnn.empty()));
    REQUIRE(r_ncnn[0].label_id == r_ort[0].label_id);
    REQUIRE(r_ncnn[0].keypoints.size() == 17);
}
#endif
