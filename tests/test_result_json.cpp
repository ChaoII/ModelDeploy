#include <catch2/catch_test_macros.hpp>

#include <string>

#include "vision/common/result_json.h"

using modeldeploy::vision::AttributeResult;
using modeldeploy::vision::ClassifyResult;
using modeldeploy::vision::DetectionResult;
using modeldeploy::vision::FaceRecognitionResult;
using modeldeploy::vision::InstanceSegResult;
using modeldeploy::vision::KeyPointsResult;
using modeldeploy::vision::LprResult;
using modeldeploy::vision::Mask;
using modeldeploy::vision::ObbResult;
using modeldeploy::vision::OCRResult;
using modeldeploy::vision::ReIdResult;
using modeldeploy::vision::Rect2f;
using modeldeploy::vision::RotatedRect;
using modeldeploy::vision::SemSegResult;

TEST_CASE("result_json: 基础几何类型", "[result_json]") {
    auto jr = modeldeploy::vision::to_json(Rect2f(1.5f, 2.5f, 30.f, 40.f));
    REQUIRE(jr["x"] == 1.5f);
    REQUIRE(jr["y"] == 2.5f);
    REQUIRE(jr["width"] == 30.f);
    REQUIRE(jr["height"] == 40.f);

    auto jp = modeldeploy::vision::to_json(modeldeploy::vision::Point3f(1.f, 2.f, 3.f));
    REQUIRE(jp["z"] == 3.f);

    auto jr2 = modeldeploy::vision::to_json(RotatedRect(10.f, 20.f, 5.f, 6.f, 0.5f));
    REQUIRE(jr2["xc"] == 10.f);
    REQUIRE(jr2["angle"] == 0.5f);
}

TEST_CASE("result_json: DetectionResult 及 vector", "[result_json]") {
    DetectionResult d;
    d.box = Rect2f(1.f, 2.f, 100.f, 200.f);
    d.label_id = 0;
    d.score = 0.95f;

    auto j = modeldeploy::vision::to_json(d);
    REQUIRE(j["label_id"] == 0);
    REQUIRE(j["score"] == 0.95f);
    REQUIRE(j["box"]["x"] == 1.f);
    REQUIRE(j["box"]["width"] == 100.f);

    std::vector<DetectionResult> v{d, d};
    auto jv = modeldeploy::vision::to_json(v);
    REQUIRE(jv.is_array());
    REQUIRE(jv.size() == 2);
    REQUIRE(jv[1]["score"] == 0.95f);
}

TEST_CASE("result_json: OCRResult", "[result_json]") {
    OCRResult r;
    r.boxes = {{0, 0, 10, 0, 10, 20, 0, 20}};
    r.text = {"hello"};
    r.rec_scores = {0.9f};
    r.cls_labels = {0};
    r.table_html = "<table></table>";

    auto j = modeldeploy::vision::to_json(r);
    REQUIRE(j["text"].size() == 1);
    REQUIRE(j["text"][0] == "hello");
    REQUIRE(j["boxes"][0].size() == 8);
    REQUIRE(j["rec_scores"][0] == 0.9f);
    REQUIRE(j["cls_labels"][0] == 0);
    REQUIRE(j["table_html"] == "<table></table>");
}

TEST_CASE("result_json: ObbResult / KeyPoints / Lpr", "[result_json]") {
    ObbResult o;
    o.rotated_box = RotatedRect(1, 2, 3, 4, 0.1f);
    o.score = 0.8f;
    auto jo = modeldeploy::vision::to_json(o);
    REQUIRE(jo["score"] == 0.8f);
    REQUIRE(jo["rotated_box"]["angle"] == 0.1f);

    KeyPointsResult k;
    k.box = Rect2f(0, 0, 10, 10);
    k.keypoints = {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
    auto jk = modeldeploy::vision::to_json(k);
    REQUIRE(jk["keypoints"].size() == 2);
    REQUIRE(jk["keypoints"][1]["z"] == 6.f);

    LprResult l;
    l.car_plate_str = "粤A12345";
    l.box = Rect2f(0, 0, 50, 20);
    auto jl = modeldeploy::vision::to_json(l);
    REQUIRE(jl["car_plate_str"] == "粤A12345");
}

TEST_CASE("result_json: 序列化 roundtrip (dump/parse 保真)", "[result_json]") {
    ClassifyResult c;
    c.label_ids = {1, 2};
    c.scores = {0.9f, 0.1f};
    c.feature = {0.1f, 0.2f, 0.3f};
    auto jc = modeldeploy::vision::to_json(c);
    auto dumped = jc.dump();
    auto parsed = nlohmann::json::parse(dumped);
    REQUIRE(parsed["label_ids"] == nlohmann::json::array({1, 2}));
    REQUIRE(parsed["scores"][0] == 0.9f);

    InstanceSegResult s;
    s.box = Rect2f(1, 2, 3, 4);
    s.mask.shape = {2, 2};
    s.mask.buffer = {0, 1, 1, 0};
    auto js = modeldeploy::vision::to_json(s);
    REQUIRE(js["mask"]["shape"] == nlohmann::json::array({2, 2}));
    REQUIRE(js["mask"]["data"].size() == 4);

    SemSegResult sem;
    sem.shape = {2, 2};
    sem.num_classes = 19;
    sem.labels = {0, 1, 1, 2};
    auto jsem = modeldeploy::vision::to_json(sem);
    REQUIRE(jsem["num_classes"] == 19);
    REQUIRE(jsem["labels"][3] == 2);

    FaceRecognitionResult f;
    f.embedding = {0.5f, 0.5f};
    REQUIRE(modeldeploy::vision::to_json(f)["embedding"][1] == 0.5f);

    ReIdResult rid;
    rid.embedding = {1.f};
    REQUIRE(modeldeploy::vision::to_json(rid)["embedding"][0] == 1.f);

    AttributeResult a;
    a.box_score = 0.7f;
    a.attr_scores = {0.1f, 0.9f};
    REQUIRE(modeldeploy::vision::to_json(a)["box_score"] == 0.7f);
}
