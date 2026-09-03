#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <vector>
#include "csrc/vision.h"

// ==================== ncnn 后端 vision 集成验证 ====================
// 需要真实 test_data（ncnn/yolo11n/yolo11n.param+.bin + test_images），无数据时优雅跳过
// （沿用 test_mnn_device.cpp 的 exists 守卫）。[ncnn][cpu] 标签可在 CPU-only CI 运行。
#ifdef ENABLE_NCNN
using namespace modeldeploy::vision;

namespace {

std::filesystem::path ncnn_test_data() {
    const char* env = std::getenv("TEST_DATA_DIR");
    if (env && *env) return std::filesystem::path(env) / "test_data";
    return std::filesystem::current_path() / "test_data";
}

std::filesystem::path ncnn_model(const std::string& rel) {
    return ncnn_test_data() / "test_models" / "ncnn" / rel;
}

std::filesystem::path ncnn_image(const std::string& name) {
    return ncnn_test_data() / "test_images" / name;
}

}  // namespace

TEST_CASE("UltralyticsDet ncnn CPU detection", "[ncnn][vision][cpu]") {
    auto model = ncnn_model("yolo11n/yolo11n.param");
    auto imgf = ncnn_image("test_detection0.jpg");
    if (!std::filesystem::exists(model) || !std::filesystem::exists(imgf)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_ncnn_backend();
    opt.set_device(modeldeploy::Device::CPU, 0);

    detection::UltralyticsDet det(model.string(), opt);
    REQUIRE(det.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<DetectionResult> results;
    REQUIRE(det.predict(img, &results, nullptr));
    REQUIRE_FALSE(results.empty());
    INFO("ncnn CPU detections=" << results.size());
    for (auto& r : results) {
        REQUIRE(r.box.width > 0);
        REQUIRE(r.box.height >= 0);
        REQUIRE(r.label_id >= 0);
        REQUIRE(r.score > 0);
    }
}
#endif  // ENABLE_NCNN
