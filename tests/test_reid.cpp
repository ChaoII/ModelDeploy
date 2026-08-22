#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vision/common/image_data.h"
#include "runtime/runtime_option.h"
#include "vision/reid/reid.h"
#include "vision/reid/gallery.h"
#include "vision/utils.h"

namespace fs = std::filesystem;
using namespace modeldeploy::vision;

namespace {
    fs::path get_test_data() {
        const char* env = std::getenv("TEST_DATA_DIR");
        if (env && *env) return fs::path(env) / "test_data";
        return fs::current_path() / "test_data";
    }
}

// 确定性测试，不依赖模型权重，始终运行。
TEST_CASE("ReIdGallery top-k cosine match", "[reid]") {
    reid::ReIdGallery g;
    std::vector<float> a(512, 0.0f), b(512, 0.0f), q(512, 0.0f);
    a[0] = 1.0f; b[1] = 1.0f; q[0] = 0.9f; q[1] = 0.1f;   // q 更接近 a
    a = utils::l2_normalize(a);
    b = utils::l2_normalize(b);
    q = utils::l2_normalize(q);
    g.enroll("A", a);
    g.enroll("B", b);
    REQUIRE(g.size() == 2);

    auto top = g.match(q, 1);
    REQUIRE(top.size() == 1);
    REQUIRE(top[0].first == "A");

    // 覆盖同 label：size 不增长，仍为 2
    g.enroll("A", b);
    REQUIRE(g.size() == 2);

    // remove
    REQUIRE(g.remove("A").at(0));
    REQUIRE_FALSE(g.remove("A").at(0));
    REQUIRE(g.size() == 1);

    g.clear();
    REQUIRE(g.size() == 0);
}

// 模型依赖测试：OSNet ONNX 外链（modelscope）。权重缺失时静默跳过，不硬失败。
TEST_CASE("ReID predict produces 512-d L2 embedding", "[reid]") {
    auto path = get_test_data() / "test_models" / "onnx" / "osnet_x1_0.onnx";
    if (!fs::exists(path)) {
        WARN("OSNet model not found; skipping predict test. "
             "Download test_data from modelscope: "
             "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps");
        return;
    }

    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();

    reid::ReID model(path.string(), opt);

    cv::Mat img(256, 128, CV_8UC3, cv::Scalar::all(128));
    ImageData data(img);

    std::vector<ReIdResult> results;
    REQUIRE(model.predict(data, &results));
    REQUIRE_FALSE(results.empty());

    const auto& emb = results[0].embedding;
    REQUIRE(emb.size() == 512);

    float norm = 0.0f;
    for (float v : emb) norm += v * v;
    REQUIRE(std::sqrt(norm) == Catch::Approx(1.0f).margin(1e-3));
}
