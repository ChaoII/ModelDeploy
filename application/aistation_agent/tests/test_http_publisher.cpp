#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include "event_publisher.hpp"
#include "httplib.h"
#include "nlohmann/json.hpp"

using nlohmann::json;

static int free_port() {
    static std::mt19937 rng(std::random_device{}());
    return 21000 + static_cast<int>(rng() % 10000);
}

static DetectionEvent sample_event() {
    DetectionEvent e;
    e.event_id = "11111111-1111-4111-8111-111111111111";
    e.edge_code = "edge-01"; e.camera_id = 7; e.task_id = 123;
    e.algorithm_type = "INTRUSION"; e.ts = "2026-09-12T08:00:00.123Z"; e.latency_ms = 12.3;
    EventDetection d; d.label = "person"; d.label_id = 0; d.confidence = 0.91f;
    d.x = 0.1f; d.y = 0.2f; d.w = 0.15f; d.h = 0.3f;
    e.detections.push_back(d);
    return e;
}

TEST_CASE("HttpPublisher posts event with bearer token", "[agent][publisher]") {
    int port = free_port();
    httplib::Server srv;
    std::atomic<int> hits{0};
    std::string got_auth, got_body;
    srv.Post("/cb", [&](const httplib::Request& req, httplib::Response& res) {
        ++hits;
        got_auth = req.get_header_value("Authorization");
        got_body = req.body;
        res.status = 200;
    });
    std::thread t([&]() { srv.listen("127.0.0.1", port); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    HttpPublisher pub("http://127.0.0.1:" + std::to_string(port) + "/cb", "tk");
    REQUIRE(pub.publish(sample_event()));
    REQUIRE(hits.load() == 1);
    REQUIRE(got_auth == "Bearer tk");
    auto j = json::parse(got_body);
    REQUIRE(j["event_id"] == "11111111-1111-4111-8111-111111111111");
    REQUIRE(j["camera_id"] == 7);
    REQUIRE(j["detections"][0]["label"] == "person");
    REQUIRE(j["detections"][0]["bbox"]["width"] == Catch::Approx(0.15f));

    srv.stop();
    t.join();
}

TEST_CASE("HttpPublisher returns false when unreachable", "[agent][publisher]") {
    HttpPublisher pub("http://127.0.0.1:1/cb", "tk");
    REQUIRE_FALSE(pub.publish(sample_event()));
}
