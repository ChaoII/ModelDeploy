#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include "inference_engine.hpp"

// OCR pipeline：det + cls + rec + dict 端到端推理，结果落到 InferResult.boxes
TEST_CASE("InferenceEngine loads OCR pipeline", "[agent][pipeline]") {
    const std::string det = "test_data/test_models/onnx/ocr/ppocrv6_tiny/det_infer.onnx";
    const std::string cls = "test_data/test_models/onnx/ocr/ppocrv6_tiny/cls_infer.onnx";
    const std::string rec = "test_data/test_models/onnx/ocr/ppocrv6_tiny/rec_infer.onnx";
    const std::string dict = "test_data/ppocrv6_tiny_dict.txt";
    if (!std::ifstream(det).good() || !std::ifstream(cls).good() ||
        !std::ifstream(rec).good() || !std::ifstream(dict).good())
        SKIP("OCR test data absent");

    ModelConfig cfg;
    cfg.name = "ocr";
    cfg.type = "ocr";
    cfg.backend = "ort";
    cfg.device = "cpu";
    cfg.path = det;       // det 主模型路径
    cfg.cls_path = cls;   // 方向分类
    cfg.rec_path = rec;   // 文字识别
    cfg.dict_path = dict;

    InferenceEngine eng;
    REQUIRE(eng.load(cfg));

    auto img = modeldeploy::vision::ImageData::imread("test_data/test_images/ocr2.jpg");
    REQUIRE(!img.empty());

    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "ocr");
    REQUIRE(!r.boxes.empty());
    REQUIRE(!r.boxes[0].text.empty());

    // 字段映射断言：OCR 无类别（label_id=-1），识别得分已写入，且包围盒为图像内合理矩形
    const auto& box = r.boxes[0];
    REQUIRE(box.label_id == -1);
    REQUIRE(box.score >= 0.f);
    REQUIRE(box.text_score > 0.f);   // 非空文字对应正的识别置信度
    REQUIRE(box.w > 0.f);
    REQUIRE(box.h > 0.f);
    REQUIRE(box.x >= 0.f);
    REQUIRE(box.y >= 0.f);
    REQUIRE(box.x + box.w <= static_cast<float>(img.width()));
    REQUIRE(box.y + box.h <= static_cast<float>(img.height()));
}
