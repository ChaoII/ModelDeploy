#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <random>
#include <stdexcept>
#include <thread>
#include "capability.hpp"
#include "heartbeat.hpp"
#include "httplib.h"
#include "nlohmann/json.hpp"

using nlohmann::json;

static int free_port() {
    static std::mt19937 rng(std::random_device{}());
    return 23000 + static_cast<int>(rng() % 5000);
}

TEST_CASE("detect_capabilities has required fields", "[agent][capability]") {
    auto c = detect_capabilities(8);
    REQUIRE(c.contains("hardware"));
    REQUIRE(c["hardware"].contains("platform"));
    REQUIRE(c.contains("backends"));
    REQUIRE(c["backends"].is_array());
    REQUIRE(c.contains("model_families"));
    REQUIRE(c["max_channels"] == 8);
    REQUIRE(c.contains("codecs"));
    REQUIRE(c["codecs"].contains("decode"));
    REQUIRE(c["codecs"].contains("encode"));
}

TEST_CASE("Heartbeat payload and POST", "[agent][heartbeat]") {
    int port = free_port();
    httplib::Server srv;
    std::atomic<int> hits{0};
    std::string got_body;
    srv.Post("/api/v1/video/edge/heartbeat", [&](const httplib::Request& req, httplib::Response& res) {
        ++hits; got_body = req.body; res.status = 200;
    });
    std::thread t([&]() { srv.listen("127.0.0.1", port); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    Heartbeat hb("http://127.0.0.1:" + std::to_string(port), "edge-01", "tok",
                 []() { return json{{"running_channels", 2}, {"event_queue_len", 0}, {"event_dropped_total", 0}}; },
                 30, 8);
    auto payload = hb.make_payload();
    REQUIRE(payload["edge_code"] == "edge-01");
    REQUIRE(payload["token"] == "tok");
    REQUIRE(payload["capabilities"].contains("backends"));
    REQUIRE(payload["metrics"]["running_channels"] == 2);
    REQUIRE(payload.contains("version"));

    REQUIRE(hb.send_once());
    REQUIRE(hits.load() == 1);
    auto posted = json::parse(got_body);
    REQUIRE(posted["edge_code"] == "edge-01");

    srv.stop();
    t.join();
}

TEST_CASE("Heartbeat disabled when cloud_url empty", "[agent][heartbeat]") {
    Heartbeat hb("", "edge-01", "tok", []() { return json::object(); });
    REQUIRE_FALSE(hb.send_once());
}

TEST_CASE("Heartbeat tolerates throwing metrics provider", "[agent][heartbeat]") {
    Heartbeat hb("", "edge-01", "tok", []() -> json { throw std::runtime_error("boom"); });
    json payload;
    REQUIRE_NOTHROW(payload = hb.make_payload());
    REQUIRE(payload["metrics"].is_object());
    REQUIRE(payload["metrics"].empty());
    REQUIRE(payload["edge_code"] == "edge-01");
}

TEST_CASE("Heartbeat start respects interval_sec", "[agent][heartbeat]") {
    int port = free_port();
    httplib::Server srv;
    std::atomic<int> hits{0};
    srv.Post("/api/v1/video/edge/heartbeat", [&](const httplib::Request&, httplib::Response& res) {
        ++hits; res.status = 200;
    });
    std::thread t([&]() { srv.listen("127.0.0.1", port); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    Heartbeat hb("http://127.0.0.1:" + std::to_string(port), "edge-01", "tok",
                 []() { return json::object(); }, 2, 8);
    hb.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    REQUIRE(hits.load() == 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    const int after = hits.load();
    REQUIRE(after >= 2);
    REQUIRE(after <= 3);
    hb.stop();

    srv.stop();
    t.join();
}
