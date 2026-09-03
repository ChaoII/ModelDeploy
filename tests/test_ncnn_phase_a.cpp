#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <filesystem>
#include <vector>
#include "runtime/runtime_option.h"
#include "vision/classification/classification.h"

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
#endif
