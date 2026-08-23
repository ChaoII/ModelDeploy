#include "catch2/catch_test_macros.hpp"
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "vision/landmark/vehicle_keypoint.h"

namespace fs = std::filesystem;
using namespace modeldeploy::vision;

namespace {
    // 权重外链（modelscope），不在仓库内。存在则真跑；缺失则优雅跳过。
    fs::path vehicle_model_path() {
        const char* dir = std::getenv("TEST_DATA_DIR");
        const fs::path base = (dir && *dir) ? fs::path(dir) / "test_data" : fs::current_path() / "test_data";
        return base / "test_models" / "onnx" / "vehicle_keypoint.onnx";
    }
}

TEST_CASE("VehicleKeypoint defaults to 4 keypoints", "[landmark]") {
    auto modelfile = vehicle_model_path();
    if (!fs::exists(modelfile)) {
        WARN("vehicle_keypoint.onnx 权重缺失（外链 modelscope），跳过默认 4 点断言");
        return;
    }
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    landmark::VehicleKeypoint model(modelfile.string(), opt);
    REQUIRE(model.get_postprocessor().get_keypoints_num() == 4);
    REQUIRE(model.is_initialized());
}

TEST_CASE("VehicleKeypoint set_keypoints_num override", "[landmark]") {
    auto modelfile = vehicle_model_path();
    if (!fs::exists(modelfile)) {
        WARN("vehicle_keypoint.onnx 权重缺失，跳过 set_keypoints_num 断言");
        return;
    }
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    landmark::VehicleKeypoint model(modelfile.string(), opt);
    model.get_postprocessor().set_keypoints_num(8);
    REQUIRE(model.get_postprocessor().get_keypoints_num() == 8);
}

TEST_CASE("VehicleKeypoint construction without weights", "[landmark]") {
    landmark::VehicleKeypoint model("nonexistent_vehicle_keypoint.onnx");
    REQUIRE_FALSE(model.is_initialized());
}
