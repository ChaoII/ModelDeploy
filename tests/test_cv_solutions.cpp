#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <opencv2/opencv.hpp>
#include "vision/solutions/object_counter.h"
#include "vision/solutions/heatmap.h"
#include "vision/solutions/speed_estimator.h"
#include "vision/solutions/distance_estimator.h"
#include "vision/solutions/object_cropper.h"
#include "vision/solutions/object_blur.h"
#include "vision/solutions/workout_monitor.h"
#include "vision/solutions/parking_manager.h"
#include "vision/solutions/vision_eye.h"
#include "vision/solutions/region_counter.h"
#include "vision/solutions/queue_manager.h"
#include "vision/solutions/track_zone.h"

using namespace modeldeploy::vision;
using namespace modeldeploy::vision::solution;
using namespace modeldeploy::vision::tracking;

TEST_CASE("ObjectCounter counts line crossing once per track", "[cv_solution]") {
    ObjectCounter c;
    c.set_line(Point2f(5, 0), Point2f(5, 10)); // 竖线 x=5
    std::vector<TrackResult> t1(1); t1[0].track_id = 1; t1[0].box = Rect2f(0,4,2,2); t1[0].label_id = 0;
    c.update(t1);
    REQUIRE(c.stats().line_in == 0);
    std::vector<TrackResult> t2(1); t2[0].track_id = 1; t2[0].box = Rect2f(8,4,2,2); t2[0].label_id = 0;
    c.update(t2);
    REQUIRE(c.stats().line_in == 1); // 进 in 侧
    std::vector<TrackResult> t3(1); t3[0].track_id = 1; t3[0].box = Rect2f(9,4,2,2); t3[0].label_id = 0;
    c.update(t3);
    REQUIRE(c.stats().line_in == 1); // 仍在 in 侧不重复
}

TEST_CASE("ObjectCounter counts region + class dimension", "[cv_solution]") {
    ObjectCounter c;
    c.set_region({Point2f(0,0), Point2f(10,0), Point2f(10,10), Point2f(0,10)});
    std::vector<TrackResult> t(2);
    t[0].track_id = 1; t[0].box = Rect2f(2,2,2,2); t[0].label_id = 0;
    t[1].track_id = 2; t[1].box = Rect2f(3,3,2,2); t[1].label_id = 1;
    c.update(t);
    REQUIRE(c.region_count() == 2);
    REQUIRE(c.stats().class_count[0] == 1);
    REQUIRE(c.stats().class_count[1] == 1);
}

TEST_CASE("Heatmap accumulates at centroids", "[cv_solution]") {
    Heatmap hm;
    hm.set_size(10, 10);
    std::vector<TrackResult> t(1);
    t[0].track_id = 1; t[0].box = Rect2f(3, 3, 2, 2); // 质心 (4,4)
    hm.update(t, 10, 10);
    hm.update(t, 10, 10);
    auto p = hm.peak();
    REQUIRE(p.first == 4);
    REQUIRE(p.second == 4);
    REQUIRE(hm.heat_at(4, 4) == Catch::Approx(2.0f).margin(1e-5f));
    REQUIRE(hm.heat_at(0, 0) == Catch::Approx(0.0f));
}

TEST_CASE("SpeedEstimator displacement over timestamp", "[cv_solution]") {
    SpeedEstimator se;
    se.set_meter_per_pixel(0.01f);
    std::vector<TrackResult> t(1); t[0].track_id = 1; t[0].box = Rect2f(0,0,10,10);
    se.update(t, 0.0);
    t[0].box = Rect2f(10,0,10,10); // 质心移动 10px
    se.update(t, 1000.0);
    REQUIRE(se.speeds_px_per_s()[1] == Catch::Approx(10.0f).margin(1e-3f));
    REQUIRE(se.speeds_m_s()[1] == Catch::Approx(0.1f).margin(1e-3f));
}

TEST_CASE("DistanceEstimator pair distances", "[cv_solution]") {
    DistanceEstimator de;
    de.set_meter_per_pixel(0.5f);
    std::vector<TrackResult> t(2);
    t[0].track_id = 1; t[0].box = Rect2f(0,0,1,1);   // 质心 (0.5,0.5)
    t[1].track_id = 2; t[1].box = Rect2f(5,0,1,1);   // 质心 (5.5,0.5)
    auto d = de.pair_distances_px(t);
    REQUIRE(d.size() == 1);
    REQUIRE(d[0].second == Catch::Approx(5.0f).margin(1e-3f));
    auto dm = de.pair_distances_m(t);
    REQUIRE(dm[0].second == Catch::Approx(2.5f).margin(1e-3f));
}

TEST_CASE("ObjectCropper extracts ROI", "[cv_solution]") {
    std::vector<uint8_t> pixels(20 * 20 * 3, 0);
    ImageData img = ImageData::from_raw(pixels.data(), 20, 20, MdImageType::PKG_BGR_U8, true);
    ObjectCropper cr;
    ImageData out;
    cr.crop(img, Rect2f(2, 2, 10, 8), &out);
    REQUIRE(out.width() == 10);
    REQUIRE(out.height() == 8);
}

TEST_CASE("ObjectBlur blurs region only", "[cv_solution]") {
    cv::Mat m(20, 20, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(m, cv::Rect(5, 5, 10, 10), cv::Scalar(255, 255, 255), cv::FILLED);
    ImageData img(m);
    ObjectBlur bl(11);
    ImageData out;
    bl.blur(img, Rect2f(5, 5, 10, 10), &out);
    cv::Mat om;
    REQUIRE(out.asMat(&om));
    REQUIRE(om.at<cv::Vec3b>(10, 10)[0] < 255);   // 框内中值被模糊
    REQUIRE(om.at<cv::Vec3b>(1, 1)[0] == 0);      // 框外角不受影响
}

TEST_CASE("WorkoutMonitor angles and rep count", "[cv_solution]") {
    WorkoutMonitor wm(70.0f, 120.0f);
    REQUIRE(WorkoutMonitor::angle(Point3f(0,0,0), Point3f(1,1,0), Point3f(2,0,0)) == Catch::Approx(90.0f).margin(1e-2f));
    wm.update(135.0f); REQUIRE(wm.reps() == 0); // 高于 max 且未先下蹲
    wm.update(50.0f);                           // 下蹲：angle < min_
    wm.update(130.0f); REQUIRE(wm.reps() == 1); // 伸直：> max 且下蹲过 → 记一次
}

TEST_CASE("ParkingManager occupancy per slot", "[cv_solution]") {
    ParkingManager pm;
    pm.set_slots({
        {Point2f(0,0), Point2f(10,0), Point2f(10,10), Point2f(0,10)},
        {Point2f(20,0), Point2f(30,0), Point2f(30,10), Point2f(20,10)}
    });
    std::vector<TrackResult> t(1); t[0].track_id = 1; t[0].box = Rect2f(2,2,2,2);
    pm.update(t);
    auto occ = pm.occupancy();
    REQUIRE(occ.size() == 2);
    REQUIRE(occ[0] == true);
    REQUIRE(occ[1] == false);
}

TEST_CASE("VisionEye maps centroid to eye point", "[cv_solution]") {
    VisionEye ve(100.0f);
    auto e = ve.map_to_eye(Point2f(50, 200), 100.0f);
    REQUIRE(e.x == Catch::Approx(50.0f));
    REQUIRE(e.y == Catch::Approx(100.0f));
    ve.add(Point2f(50, 200));
    REQUIRE(ve.eyes().size() == 1);
    REQUIRE(ve.eyes()[0].x == Catch::Approx(50.0f));
}

TEST_CASE("RegionCounter counts tracks per named region", "[cv_solution]") {
    RegionCounter rc;
    rc.add_region("A", {Point2f(0,0), Point2f(10,0), Point2f(10,10), Point2f(0,10)});
    rc.add_region("B", {Point2f(20,0), Point2f(30,0), Point2f(30,10), Point2f(20,10)});
    std::vector<TrackResult> t(3);
    t[0].track_id=1; t[0].box=Rect2f(2,2,2,2);
    t[1].track_id=2; t[1].box=Rect2f(22,2,2,2);
    t[2].track_id=3; t[2].box=Rect2f(100,100,2,2);
    rc.update(t);
    auto c = rc.region_counts();
    REQUIRE(c["A"] == 1);
    REQUIRE(c["B"] == 1);
    REQUIRE(rc.total_regions() == 2);
    rc.update({t[0]});
    REQUIRE(rc.region_counts()["A"] == 1);
    REQUIRE(rc.region_counts()["B"] == 0);
}

TEST_CASE("RegionCounter class filter", "[cv_solution]") {
    RegionCounter rc;
    rc.add_region("A", {Point2f(0,0), Point2f(10,0), Point2f(10,10), Point2f(0,10)});
    rc.set_classes({2});
    std::vector<TrackResult> t(2);
    t[0].track_id=1; t[0].box=Rect2f(2,2,2,2); t[0].label_id=2;
    t[1].track_id=2; t[1].box=Rect2f(3,3,2,2); t[1].label_id=0;
    rc.update(t);
    REQUIRE(rc.region_counts()["A"] == 1);
}

TEST_CASE("QueueManager counts tracks inside queue region per frame", "[cv_solution]") {
    QueueManager qm;
    qm.set_region({Point2f(0,0), Point2f(10,0), Point2f(10,10), Point2f(0,10)});
    std::vector<TrackResult> t(3);
    t[0].track_id=1; t[0].box=Rect2f(1,1,2,2);
    t[1].track_id=2; t[1].box=Rect2f(4,4,2,2);
    t[2].track_id=3; t[2].box=Rect2f(50,50,2,2);
    qm.update(t);
    REQUIRE(qm.queue_count() == 2);
    qm.update({t[2]});
    REQUIRE(qm.queue_count() == 0);
}

TEST_CASE("TrackZone keeps only tracks inside region", "[cv_solution]") {
    TrackZone tz;
    tz.set_region({Point2f(0,0), Point2f(10,0), Point2f(10,10), Point2f(0,10)});
    std::vector<TrackResult> t(3);
    t[0].track_id=1; t[0].box=Rect2f(1,1,2,2);
    t[1].track_id=2; t[1].box=Rect2f(50,50,2,2);
    t[2].track_id=3; t[2].box=Rect2f(4,4,2,2);
    tz.update(t);
    REQUIRE(tz.inside_count() == 2);
    bool ids_ok = false;
    for (const auto& r : tz.inside_tracks()) if (r.track_id == 2) ids_ok = true;
    REQUIRE_FALSE(ids_ok);
}
