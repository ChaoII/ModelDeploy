#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>
#include "runtime/runtime_option.h"
#include "vision/classification/classification.h"
#include "vision/obb/ultralytics_obb.h"
#include "vision/pose/ultralytics_pose.h"
#include "vision/iseg/ultralytics_seg.h"
#include "vision/sem/ultralytics_sem.h"
#include "vision/depth/ultralytics_depth.h"

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

// Spearman 秩相关：分别对 a、b 求平均秩后计算两者秩的 Pearson 相关。
double spearman(const float* a, const float* b, size_t n) {
    auto ranks = [n](const float* d) {
        std::vector<size_t> idx(n);
        for (size_t i = 0; i < n; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [d](size_t x, size_t y) { return d[x] < d[y]; });
        std::vector<double> out(n);
        size_t i = 0;
        while (i < n) {
            size_t j = i;
            while (j + 1 < n && d[idx[j + 1]] == d[idx[i]]) ++j;
            const double avg = static_cast<double>(i + j + 2) * 0.5;  // 1-based 秩，并列取平均
            for (size_t k = i; k <= j; ++k) out[idx[k]] = avg;
            i = j + 1;
        }
        return out;
    };
    const auto ra = ranks(a);
    const auto rb = ranks(b);
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { ma += ra[i]; mb += rb[i]; }
    ma /= static_cast<double>(n);
    mb /= static_cast<double>(n);
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < n; ++i) {
        num += (ra[i] - ma) * (rb[i] - mb);
        da += (ra[i] - ma) * (ra[i] - ma);
        db += (rb[i] - mb) * (rb[i] - mb);
    }
    return num / std::sqrt(da * db);
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

TEST_CASE("UltralyticsSeg ncnn vs ORT", "[ncnn][phase_a][seg]") {
    namespace p = phase_a_test;
    const auto mdl = p::ncnn_model("yolo26n-seg");
    const auto imgf = p::image("test_detection0.jpg");
    const auto onx = p::onnx_model("yolo26n-seg.onnx");
    if (!p::avail(mdl) || !p::avail(imgf) || !p::avail(onx)) return;
    modeldeploy::vision::detection::UltralyticsSeg ort(onx.string(), p::ort_opt());
    modeldeploy::vision::detection::UltralyticsSeg ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE((ort.is_initialized() && ncnn.is_initialized()));
    auto img = ImageData::imread(imgf.string());
    std::vector<modeldeploy::vision::InstanceSegResult> r_ort, r_ncnn;
    REQUIRE(ort.predict(img, &r_ort, nullptr));
    REQUIRE(ncnn.predict(img, &r_ncnn, nullptr));
    REQUIRE((!r_ort.empty() && !r_ncnn.empty()));
    REQUIRE(r_ncnn[0].label_id == r_ort[0].label_id);
    REQUIRE(r_ncnn[0].score > 0.5f);
}

TEST_CASE("UltralyticsSem ncnn vs ORT", "[ncnn][phase_a][sem]") {
    namespace p = phase_a_test;
    const auto mdl = p::ncnn_model("yolo26n-sem");
    const auto imgf = p::image("test_detection0.jpg");
    const auto onx = p::onnx_model("yolo26n-sem.onnx");
    if (!p::avail(mdl) || !p::avail(imgf) || !p::avail(onx)) return;
    modeldeploy::vision::detection::UltralyticsSem ort(onx.string(), p::ort_opt());
    modeldeploy::vision::detection::UltralyticsSem ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE((ort.is_initialized() && ncnn.is_initialized()));
    auto img = ImageData::imread(imgf.string());
    modeldeploy::vision::SemSegResult r_ort, r_ncnn;
    REQUIRE(ort.predict(img, &r_ort, nullptr));
    REQUIRE(ncnn.predict(img, &r_ncnn, nullptr));
    REQUIRE(r_ncnn.shape == r_ort.shape);
    const size_t total = static_cast<size_t>(r_ort.shape[0]) * static_cast<size_t>(r_ort.shape[1]);
    REQUIRE(total > 0);
    size_t same = 0;
    for (size_t i = 0; i < total; ++i) {
        if (r_ncnn.labels[i] == r_ort.labels[i]) ++same;
    }
    REQUIRE(static_cast<double>(same) / static_cast<double>(total) > 0.90);
}

TEST_CASE("UltralyticsDepth ncnn vs ORT", "[ncnn][phase_a][depth]") {
    namespace p = phase_a_test;
    const auto mdl = p::ncnn_model("yolo26n-depth");
    const auto imgf = p::image("test_detection0.jpg");
    const auto onx = p::onnx_model("yolo26n-depth.onnx");
    if (!p::avail(mdl) || !p::avail(imgf) || !p::avail(onx)) return;
    modeldeploy::vision::detection::UltralyticsDepth ort(onx.string(), p::ort_opt());
    modeldeploy::vision::detection::UltralyticsDepth ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE((ort.is_initialized() && ncnn.is_initialized()));
    auto img = ImageData::imread(imgf.string());
    modeldeploy::vision::DepthResult r_ort, r_ncnn;
    REQUIRE(ort.predict(img, &r_ort, nullptr));
    REQUIRE(ncnn.predict(img, &r_ncnn, nullptr));
    REQUIRE(r_ncnn.shape == r_ort.shape);
    const size_t n = static_cast<size_t>(r_ort.shape[0]) * static_cast<size_t>(r_ort.shape[1]);
    REQUIRE(n > 0);
    REQUIRE(phase_a_test::spearman(r_ort.depth.data(), r_ncnn.depth.data(), n) > 0.9);
}
#endif
