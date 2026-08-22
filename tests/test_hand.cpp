#include "catch2/catch_test_macros.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

#include "vision/hand/hand.h"

namespace fs = std::filesystem;
using namespace modeldeploy::vision;

namespace {
    // 权重外链（modelscope），不在仓库内。存在则真跑；缺失则优雅跳过。
    fs::path hand_model_path() {
        const char* dir = std::getenv("TEST_DATA_DIR");
        const fs::path base = (dir && *dir) ? fs::path(dir) / "test_data" : fs::current_path() / "test_data";
        return base / "test_models" / "onnx" / "hand_pose.onnx";
    }
}

TEST_CASE("HandKeypoint postprocessor defaults to 21 keypoints", "[hand]") {
    auto modelfile = hand_model_path();
    if (!fs::exists(modelfile)) {
        WARN("hand_pose.onnx 权重缺失（外链 modelscope），跳过后处理 21 点断言");
        return;
    }
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    hand::HandKeypoint model(modelfile.string(), opt);
    const auto& pp = model.get_postprocessor();
    REQUIRE(pp.get_keypoints_num() == 21);
}

TEST_CASE("HandKeypoint predict produces 21 keypoints", "[hand]") {
    auto modelfile = hand_model_path();
    if (!fs::exists(modelfile)) {
        WARN("hand_pose.onnx 权重缺失（外链 modelscope），跳过 predict 主路径");
        return;
    }
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    hand::HandKeypoint model(modelfile.string(), opt);

    // 构造 640x640 BGR 画布做冒烟：真实权重下应产出 21 个关键点
    cv::Mat canvas(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    ImageData img(canvas);

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    REQUIRE_FALSE(results.empty());
    const auto& front = results.front();
    REQUIRE(front.keypoints.size() == 21);
    REQUIRE(front.box.width > 0);
    REQUIRE(front.score >= 0.0f);
}
