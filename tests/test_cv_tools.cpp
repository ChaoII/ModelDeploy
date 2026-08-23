#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "vision/tools/detections.h"

using namespace modeldeploy::vision;
using namespace modeldeploy::vision::tool;

TEST_CASE("Detections iou", "[cv_tools]") {
    REQUIRE(iou(Rect2f(0,0,10,10), Rect2f(0,0,10,10)) == Catch::Approx(1.0f).margin(1e-5f));
    REQUIRE(iou(Rect2f(0,0,10,10), Rect2f(20,20,10,10)) == Catch::Approx(0.0f).margin(1e-6f));
    float half = iou(Rect2f(0,0,10,10), Rect2f(5,0,10,10));
    REQUIRE((half > 0.30f && half < 0.36f)); // 并=150 交=50
}

TEST_CASE("Detections nms keeps top by confidence", "[cv_tools]") {
    Detections d;
    d.boxes   = {Rect2f(0,0,10,10), Rect2f(1,1,10,10), Rect2f(100,100,10,10)};
    d.class_id = {0, 0, 1};
    d.confidence = {0.5f, 0.9f, 0.7f};
    nms(d, 0.4f);   // 框0/框1 IoU 高 → 只留框1；框2 分离保留
    REQUIRE(d.size() == 2);
    REQUIRE(d.confidence[0] == Catch::Approx(0.9f));
}

TEST_CASE("Detections filter_by_class", "[cv_tools]") {
    Detections d;
    d.class_id = {0, 1, 2, 0};
    d.boxes.resize(4);
    filter_by_class(d, {0, 2});
    REQUIRE(d.size() == 3);
}

TEST_CASE("Detections from_track maps tracker_id", "[cv_tools]") {
    std::vector<tracking::TrackResult> t(2);
    t[0].track_id = 7; t[0].box = Rect2f(1,1,5,5); t[0].label_id = 2; t[0].score = 0.8f;
    t[1].track_id = 3; t[1].box = Rect2f(9,9,5,5);
    auto d = from_track(t);
    REQUIRE(d.size() == 2);
    REQUIRE(d.tracker_id[0] == 7);
    REQUIRE(d.class_id[0] == 2);
    REQUIRE(d.confidence[0] == Catch::Approx(0.8f));
    REQUIRE(d.boxes[1].x == Catch::Approx(9.0f));
}
