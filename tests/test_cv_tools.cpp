#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "vision/tools/detections.h"
#include "vision/tools/zone.h"
#include "vision/tools/annotator.h"
#include "vision/tools/metrics.h"
#include "vision/tools/slicer.h"

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

TEST_CASE("LineZone counts crossing once", "[cv_tools]") {
    LineZone z(Point2f(5, 0), Point2f(5, 10));
    REQUIRE(z.trigger_count() == 0);
    REQUIRE(z.trigger(Point2f(0, 5)) == false); // out 侧
    REQUIRE(z.trigger(Point2f(8, 5)) == true);  // 跨到 in → 计数
    REQUIRE(z.trigger_count() == 1);
    REQUIRE(z.trigger(Point2f(9, 5)) == false); // 仍在 in
    z.reset();
    REQUIRE(z.trigger_count() == 0);
}

TEST_CASE("PolygonZone contains + current_count", "[cv_tools]") {
    PolygonZone z({Point2f(0,0), Point2f(10,0), Point2f(10,10), Point2f(0,10)});
    REQUIRE(z.contains(Point2f(5,5)));
    REQUIRE_FALSE(z.contains(Point2f(20,20)));
    z.update({Point2f(2,2), Point2f(50,50)});
    REQUIRE(z.current_count() == 1);
}

static ImageData make_canvas(int w, int h) {
    std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 3, 0);
    return ImageData::from_raw(pixels.data(), w, h, MdImageType::PKG_BGR_U8, true);
}

TEST_CASE("Annotator draws rectangle on canvas", "[cv_tools]") {
    ImageData frame = make_canvas(20, 20);
    Annotator ann(&frame);
    ann.rectangle(Rect2f(2, 2, 10, 10), cv::Scalar(0, 0, 255), 2);
    ann.text("obj", Point2f(2, 2), cv::Scalar(255, 255, 255), 0.5);
    cv::Mat m;
    REQUIRE(frame.asMat(&m));
    REQUIRE(m.at<cv::Vec3b>(3, 3)[2] == 255); // 边框上红通道
}

TEST_CASE("draw_box_labels draws boxes", "[cv_tools]") {
    ImageData frame = make_canvas(30, 30);
    Detections d;
    d.boxes = {Rect2f(1, 1, 5, 5)}; d.class_id = {0}; d.confidence = {0.9f};
    draw_box_labels(d, &frame, {{0, "person"}});
    cv::Mat m;
    REQUIRE(frame.asMat(&m));
    REQUIRE(m.at<cv::Vec3b>(2, 2)[2] == 255);
}

TEST_CASE("Metrics counts and scores on tiny sample", "[cv_tools]") {
    std::vector<Rect2f> preds = {Rect2f(0,0,10,10), Rect2f(50,50,10,10)};
    std::vector<float> scores = {0.9f, 0.3f};
    std::vector<Rect2f> gt = {Rect2f(0,0,10,10)};
    auto mc = count_tp_fp_fn(preds, scores, gt, 0.5);
    REQUIRE(mc.tp == 1); REQUIRE(mc.fp == 1); REQUIRE(mc.fn == 0);
    auto s = evaluate_metrics(preds, scores, gt, 0.5);
    REQUIRE(s.precision == Catch::Approx(0.5));
    REQUIRE(s.recall == Catch::Approx(1.0));
    REQUIRE(s.f1 == Catch::Approx(2.0 * 0.5 * 1.0 / 1.5).margin(1e-6));
    REQUIRE(s.map50 > 0.0);
}

TEST_CASE("Slicer tiles a large image", "[cv_tools]") {
    std::vector<uint8_t> pixels(100 * 100 * 3, 0);
    ImageData img = ImageData::from_raw(pixels.data(), 100, 100, MdImageType::PKG_BGR_U8, true);
    InferenceSlicer slicer(60, 60, 10);
    auto tiles = slicer.slice(img);
    REQUIRE(tiles.size() >= 4);
    REQUIRE(tiles[0].tile.width() == 60);
    REQUIRE(tiles[0].offset.x == 0.0f);
}

TEST_CASE("Slicer reassembles mapped boxes", "[cv_tools]") {
    InferenceSlicer slicer(50, 50, 0);
    std::vector<uint8_t> pixels(100 * 100 * 3, 0);
    ImageData img = ImageData::from_raw(pixels.data(), 100, 100, MdImageType::PKG_BGR_U8, true);
    auto tiles = slicer.slice(img);
    Detections per; per.boxes = {Rect2f(0,0,10,10)}; per.confidence={0.9f}; per.class_id={0};
    std::vector<Detections> per_slice(tiles.size());
    per_slice[3] = per;  // 右下 tile 的局部框
    ImageData out; std::vector<Rect2f> mapped;
    reassemble(tiles, per_slice, &out, &mapped);
    REQUIRE(out.width() == 100);
    REQUIRE(mapped.size() == 1);
    REQUIRE(mapped[0].x == Catch::Approx(50.0f));  // tile offset (50,50)
    REQUIRE(mapped[0].y == Catch::Approx(50.0f));
}
