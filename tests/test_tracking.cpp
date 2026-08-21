#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <vector>
#include "vision/tracking/base_tracker.h"
#include "vision/tracking/matching/iou_matching.h"
#include "vision/tracking/matching/hungarian.h"
#include "vision/tracking/matching/kalman_filter.h"
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::tracking;
using namespace Catch;

TEST_CASE("BaseTracker: empty input yields empty output", "[tracking]") {
    auto r = empty_update();
    REQUIRE(r.empty());
}

TEST_CASE("Matching: IoU overlap", "[tracking]") {
    Rect2f a{0,0,10,10}, b{0,0,10,10};
    REQUIRE(iou(a,b) == Approx(1.0f));
}

TEST_CASE("Matching: IoU no overlap", "[tracking]") {
    Rect2f a{0,0,10,10}, b{100,100,10,10};
    REQUIRE(iou(a,b) == Approx(0.0f));
}

TEST_CASE("Matching: IoU distance", "[tracking]") {
    Rect2f a{0,0,10,10}, b{0,0,10,10};
    auto d = iou_distance({a}, {b});
    REQUIRE(d.size() == 1);
    REQUIRE(d[0].size() == 1);
    REQUIRE(d[0][0] == Approx(0.0f));
}

TEST_CASE("Matching: hungarian assignment", "[tracking]") {
    std::vector<std::vector<float>> cost{{1,2},{2,1}};
    auto res = linear_sum_assignment(cost);
    REQUIRE(res.size() == 2);
    REQUIRE(res[0].first == 0);
    REQUIRE(res[0].second == 0);
    REQUIRE(res[1].first == 1);
    REQUIRE(res[1].second == 1);
    float total = cost[res[0].first][res[0].second] + cost[res[1].first][res[1].second];
    REQUIRE(total == Approx(2.0f));
}

TEST_CASE("Matching: hungarian rectangular", "[tracking]") {
    std::vector<std::vector<float>> cost{{1,3,1},{2,1,4}};
    auto res = linear_sum_assignment(cost);
    REQUIRE(res.size() == 2);
    std::vector<int> used_cols;
    float total = 0.0f;
    for (size_t r = 0; r < res.size(); ++r) {
        REQUIRE(res[r].first == static_cast<int>(r));
        REQUIRE(std::find(used_cols.begin(), used_cols.end(), res[r].second) == used_cols.end());
        used_cols.push_back(res[r].second);
        total += cost[res[r].first][res[r].second];
    }
    REQUIRE(total == Approx(2.0f));
}

TEST_CASE("KalmanFilter: init sets nonzero position/velocity covariance diag", "[tracking]") {
    KalmanFilter kf;
    Rect2f box{10, 10, 20, 40};
    kf.init(box);
    auto diag = kf.get_covariance_diag();
    REQUIRE(diag.size() == 8);
    for (size_t i = 0; i < diag.size(); ++i) {
        INFO("state dim " << i);
        REQUIRE(diag[i] > 0.0);
    }
}

TEST_CASE("KalmanFilter: static box stays put", "[tracking]") {
    KalmanFilter kf;
    Rect2f box{10,10,20,40};
    kf.init(box);
    kf.predict();
    auto s1 = kf.get_state();
    REQUIRE(s1.x == Approx(10.0f).margin(0.1f));
    REQUIRE(s1.y == Approx(10.0f).margin(0.1f));
    REQUIRE(s1.width == Approx(20.0f).margin(0.1f));
    REQUIRE(s1.height == Approx(40.0f).margin(0.1f));
    kf.update(box);
    auto s2 = kf.get_state();
    REQUIRE(s2.x == Approx(10.0f).margin(0.1f));
    REQUIRE(s2.y == Approx(10.0f).margin(0.1f));
    REQUIRE(s2.width == Approx(20.0f).margin(0.1f));
    REQUIRE(s2.height == Approx(40.0f).margin(0.1f));
}

TEST_CASE("KalmanFilter: converges toward measurement", "[tracking]") {
    KalmanFilter kf;
    Rect2f start{0,0,20,40};
    Rect2f target{100,50,30,60};
    kf.init(start);
    Rect2f prev = target;
    for (int i = 0; i < 20; ++i) {
        kf.predict();
        kf.update(target);
        prev = kf.get_state();
    }
    REQUIRE(prev.x == Approx(100.0f).margin(2.0f));
    REQUIRE(prev.y == Approx(50.0f).margin(2.0f));
    REQUIRE(prev.width == Approx(30.0f).margin(2.0f));
    REQUIRE(prev.height == Approx(60.0f).margin(2.0f));
}
