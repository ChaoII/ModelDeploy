#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <vector>
#include "vision/sam/fastsam.h"
#include "runtime/runtime_option.h"
#include "test_gpu_utils.h"

namespace fs = std::filesystem;
using namespace modeldeploy::vision;

namespace {
fs::path test_data_path() {
    const char* env = std::getenv("TEST_DATA_DIR");
    if (env && *env) return fs::path(env) / "test_data";
    return fs::current_path() / "test_data";
}
} // namespace

TEST_CASE("FastSAM predict produces InstanceSegResult", "[seg]") {
    const fs::path model = test_data_path() / "test_models" / "onnx" / "FastSAM-s.onnx";
    if (!fs::exists(model)) {
        SKIP("fastsam-s.onnx 测试模型缺失，跳过 FastSAM 运行时验证");
    }
    const fs::path img_path = test_data_path() / "test_images" / "test_detection0.jpg";
    if (!fs::exists(img_path)) {
        SKIP("测试图片缺失，跳过");
    }

    seg::FastSam sam(model.string());
    REQUIRE(sam.is_initialized());

    auto img = ImageData::imread(img_path.string());
    REQUIRE_FALSE(img.empty());

    std::vector<InstanceSegResult> res;
    REQUIRE(sam.predict(img, &res));
    REQUIRE_FALSE(res.empty());
    REQUIRE(res[0].mask.data() != nullptr);
    REQUIRE_FALSE(res[0].mask.shape.empty());
}

TEST_CASE("FastSAM predict_with_prompts filters by bbox and point", "[seg]") {
    const fs::path model = test_data_path() / "test_models" / "onnx" / "FastSAM-s.onnx";
    const fs::path img_path = test_data_path() / "test_images" / "test_detection0.jpg";
    if (!fs::exists(model) || !fs::exists(img_path)) { SKIP("FastSAM onnx or test image missing"); }
    seg::FastSam sam(model.string());
    REQUIRE(sam.is_initialized());
    auto img = ImageData::imread(img_path.string());
    REQUIRE_FALSE(img.empty());

    std::vector<InstanceSegResult> all;
    REQUIRE(sam.predict(img, &all));
    REQUIRE_FALSE(all.empty());
    const auto anchor = all[0];

    seg::FastSamPrompts pb; pb.bboxes.push_back(anchor.box);
    std::vector<InstanceSegResult> bybox;
    REQUIRE(sam.predict_with_prompts(img, pb, &bybox));
    REQUIRE_FALSE(bybox.empty());
    CHECK(bybox.size() <= all.size());

    Point2f pc(anchor.box.x + anchor.box.width * 0.5f, anchor.box.y + anchor.box.height * 0.5f);
    seg::FastSamPrompts pp; pp.points.push_back(pc); pp.point_labels.push_back(1);
    std::vector<InstanceSegResult> bypt;
    REQUIRE(sam.predict_with_prompts(img, pp, &bypt));
    REQUIRE_FALSE(bypt.empty());
    for (const auto& r : bypt) {
        const int h = r.mask.shape.size() >= 2 ? (int)r.mask.shape[0] : 0;
        const int w = r.mask.shape.size() >= 2 ? (int)r.mask.shape[1] : 0;
        int mx = (int)(pc.x - r.box.x), my = (int)(pc.y - r.box.y);
        const bool in_box = (mx >= 0) && (my >= 0) && (mx < w) && (my < h);
        CHECK(in_box);
    }

    seg::FastSamPrompts empty;
    std::vector<InstanceSegResult> byall;
    REQUIRE(sam.predict_with_prompts(img, empty, &byall));
    CHECK(byall.size() == all.size());
}

// ==================== FastSAM GPU 预测（[gpu]） ====================
// 现有 [seg] 用例均为默认 CPU RuntimeOption（纯 CPU 路径）。本用例补 ORT CUDA EP 的
// GPU 推理覆盖：无 CUDA 设备时 WARN 跳过（沿用 test_gpu_utils.h 约定），不进 FAIL。
#ifdef WITH_GPU
TEST_CASE("FastSAM predict on ORT CUDA GPU device", "[gpu][seg]") {
    MD_TEST_GPU_OR_SKIP();
    const fs::path model = test_data_path() / "test_models" / "onnx" / "FastSAM-s.onnx";
    if (!fs::exists(model)) {
        SKIP("fastsam-s.onnx 测试模型缺失，跳过 FastSAM GPU 验证");
    }
    const fs::path img_path = test_data_path() / "test_images" / "test_detection0.jpg";
    if (!fs::exists(img_path)) {
        SKIP("测试图片缺失，跳过");
    }

    modeldeploy::RuntimeOption opt;
    opt.use_gpu(0);
    seg::FastSam sam(model.string(), opt);
    REQUIRE(sam.is_initialized());

    auto img = ImageData::imread(img_path.string());
    REQUIRE_FALSE(img.empty());

    std::vector<InstanceSegResult> res;
    REQUIRE(sam.predict(img, &res));
    REQUIRE_FALSE(res.empty());
    REQUIRE(res[0].mask.data() != nullptr);
    REQUIRE_FALSE(res[0].mask.shape.empty());
}
#endif  // WITH_GPU
