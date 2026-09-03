#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <vector>
#include "csrc/vision.h"
#include "csrc/vision/utils.h"

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

std::filesystem::path onnx_yolo11n() {
    return ncnn_test_data() / "test_models" / "onnx" / "yolo11n" / "yolo11n.onnx";
}

const DetectionResult* best_det(const std::vector<DetectionResult>& results) {
    const DetectionResult* best = &results[0];
    for (auto& r : results) {
        if (r.score > best->score) best = &r;
    }
    return best;
}

}  // namespace

TEST_CASE("UltralyticsDet ncnn CPU detection", "[ncnn][vision][cpu]") {
    auto model = ncnn_model("yolo11n/yolo11n.param");
    auto ort_model = onnx_yolo11n();
    auto imgf = ncnn_image("test_detection0.jpg");
    if (!std::filesystem::exists(model) || !std::filesystem::exists(ort_model) || !std::filesystem::exists(imgf)) return;

    // ORT 基线：同图同尺寸，默认 backend（CPU）先走一次，作为 ncnn 结果的对齐锚点。
    modeldeploy::RuntimeOption ort_opt;
    ort_opt.use_cpu();
    detection::UltralyticsDet ort_det(ort_model.string(), ort_opt);
    REQUIRE(ort_det.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());
    std::vector<DetectionResult> ort_results;
    REQUIRE(ort_det.predict(img, &ort_results, nullptr));
    REQUIRE_FALSE(ort_results.empty());

    modeldeploy::RuntimeOption opt;
    opt.use_ncnn_backend();
    opt.set_device(modeldeploy::Device::CPU, 0);

    detection::UltralyticsDet det(model.string(), opt);
    REQUIRE(det.is_initialized());

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

    // 与 ORT 基线锚点对比：两者最高置信检测的 label_id 一致，且 ncnn 最高置信框与 ORT 的 IoU>=0.5。
    const DetectionResult* ort_best = best_det(ort_results);
    const DetectionResult* ncnn_best = best_det(results);
    INFO("ORT best: label=" << ort_best->label_id << " score=" << ort_best->score
                            << " box=" << ort_best->box.to_string());
    INFO("ncnn best: label=" << ncnn_best->label_id << " score=" << ncnn_best->score
                             << " box=" << ncnn_best->box.to_string());
    REQUIRE(ncnn_best->label_id == ort_best->label_id);
    REQUIRE(modeldeploy::vision::utils::iou_rects(ncnn_best->box, ort_best->box) >= 0.5f);
}
#endif  // ENABLE_NCNN
