#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "vision/solutions/object_counter.h"
#include "vision/solutions/heatmap.h"

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
