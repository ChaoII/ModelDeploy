#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include "inference_engine.hpp"

// LPR 车牌 pipeline：det + rec 端到端推理，结果落到 InferResult.boxes
TEST_CASE("InferenceEngine loads LPR pipeline", "[agent][pipeline]") {
    const std::string det = "test_data/test_models/onnx/yolov5plate.onnx";
    const std::string rec = "test_data/test_models/onnx/plate_recognition_color.onnx";
    const std::string img_path = "test_data/test_images/test_lpr_detection.jpg";
    if (!std::ifstream(det).good() || !std::ifstream(rec).good() || !std::ifstream(img_path).good())
        SKIP("LPR test data absent");

    ModelConfig cfg;
    cfg.name = "lpr";
    cfg.type = "lpr";
    cfg.backend = "ort";
    cfg.device = "cpu";
    cfg.path = det;       // 车牌检测模型
    cfg.rec_path = rec;   // 车牌识别模型

    InferenceEngine eng;
    REQUIRE(eng.load(cfg));

    auto img = modeldeploy::vision::ImageData::imread(img_path);
    REQUIRE(!img.empty());

    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "lpr");
    REQUIRE(!r.boxes.empty());
    REQUIRE(!r.boxes[0].text.empty());

    // 字段映射断言：车牌无类别（label_id=-1），检测得分写入 score/text_score；
    // 识别阶段必须保留检测框几何信息（回归：rec 后处理曾把 box/score 清零）
    const auto& box = r.boxes[0];
    REQUIRE(box.label_id == -1);
    REQUIRE(box.text_score > 0.f);
    REQUIRE(box.score > 0.f);
    REQUIRE(box.w > 0.f);
    REQUIRE(box.h > 0.f);
}
