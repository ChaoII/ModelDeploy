#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <cmath>
#include "perf_stats.hpp"

static bool approx_eq(double a, double b, double eps = 0.05) {
    return std::fabs(a - b) < eps;
}

TEST_CASE("PerfStats basic recording", "[perf_stats]") {
    PerfStats ps;
    ps.start();
    ps.record_frame(1000, 5000, 3000, 2000);
    REQUIRE(ps.frame_count == 1);
    REQUIRE(approx_eq(ps.avg_decode_ms(), 1.0));
    REQUIRE(approx_eq(ps.avg_infer_ms(), 5.0));
    REQUIRE(approx_eq(ps.avg_draw_ms(), 3.0));
    REQUIRE(approx_eq(ps.avg_encode_ms(), 2.0));
    REQUIRE(approx_eq(ps.avg_total_ms(), 11.0));
}

TEST_CASE("PerfStats multiple frames", "[perf_stats]") {
    PerfStats ps;
    ps.start();
    for (int i = 0; i < 10; ++i)
        ps.record_frame(1000, 2000, 1500, 500);
    REQUIRE(ps.frame_count == 10);
    REQUIRE(approx_eq(ps.avg_decode_ms(), 1.0));
    REQUIRE(approx_eq(ps.avg_infer_ms(), 2.0));
    REQUIRE(approx_eq(ps.avg_total_ms(), 5.0));
}

TEST_CASE("PerfStats empty stats", "[perf_stats]") {
    PerfStats ps;
    ps.start();
    REQUIRE(ps.frame_count == 0);
    REQUIRE(ps.avg_decode_ms() == 0.0);
    REQUIRE(ps.fps() == 0.0);
}

TEST_CASE("PerfStats reset", "[perf_stats]") {
    PerfStats ps;
    ps.start();
    ps.record_frame(1000, 2000, 3000, 4000);
    REQUIRE(ps.frame_count == 1);
    ps.reset();
    REQUIRE(ps.frame_count == 0);
    REQUIRE(ps.avg_decode_ms() == 0.0);
}

TEST_CASE("PerfStats thread safety", "[perf_stats]") {
    PerfStats ps;
    ps.start();
    std::thread t1([&]() { for (int i = 0; i < 100; ++i) ps.record_frame(1000, 2000, 3000, 4000); });
    std::thread t2([&]() { for (int i = 0; i < 100; ++i) ps.record_frame(4000, 3000, 2000, 1000); });
    t1.join(); t2.join();
    REQUIRE(ps.frame_count == 200);
    REQUIRE(approx_eq(ps.avg_total_ms(), 10.0));
}

TEST_CASE("PerfStats to_json", "[perf_stats]") {
    PerfStats ps;
    ps.start();
    ps.record_frame(1000, 2000, 3000, 4000);
    auto j = ps.to_json();
    REQUIRE(j.find("\"frames\":1") != std::string::npos);
}
