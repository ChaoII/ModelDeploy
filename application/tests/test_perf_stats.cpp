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

TEST_CASE("PerfStats ingest_sdk", "[perf_stats]") {
    PerfStats ps;
    ps.start();
    ps.ingest_sdk(120, 118, 2, 3.5, 1, 7.25);
    REQUIRE(ps.sdk_frames_in == 120);
    REQUIRE(ps.sdk_frames_out == 118);
    REQUIRE(ps.sdk_dropped == 2);
    REQUIRE(ps.sdk_reconnect_count == 1);
    REQUIRE(approx_eq(ps.sdk_avg_decode_ms, 3.5));
    REQUIRE(approx_eq(ps.sdk_avg_encode_ms, 7.25));
    auto j = ps.to_json();
    // 既有字段仍保留（向后兼容），SDK 字段为增量
    REQUIRE(j.find("\"frames\":0") != std::string::npos);
    REQUIRE(j.find("\"avg_decode_ms\"") != std::string::npos);
    REQUIRE(j.find("\"sdk_frames_in\":120") != std::string::npos);
    REQUIRE(j.find("\"sdk_frames_out\":118") != std::string::npos);
    REQUIRE(j.find("\"sdk_dropped\":2") != std::string::npos);
    REQUIRE(j.find("\"sdk_reconnect_count\":1") != std::string::npos);
    REQUIRE(j.find("\"sdk_avg_decode_ms\":3.5") != std::string::npos);
    REQUIRE(j.find("\"sdk_avg_encode_ms\":7.25") != std::string::npos);
    // reset 清零 SDK 统计
    ps.reset();
    REQUIRE(ps.sdk_frames_in == 0);
    REQUIRE(ps.sdk_dropped == 0);
    REQUIRE(ps.sdk_avg_decode_ms == 0.0);
}
