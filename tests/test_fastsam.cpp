#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <vector>
#include "vision/sam/fastsam.h"

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
    const fs::path model = test_data_path() / "test_models" / "onnx" / "fastsam" / "fastsam-s.onnx";
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
