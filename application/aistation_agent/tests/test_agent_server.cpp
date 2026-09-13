#include <catch2/catch_test_macros.hpp>
#include <random>
#include <thread>
#include <chrono>
#include "agent_server.hpp"
#include "pipeline_manager.hpp"
#include "httplib.h"
#include "nlohmann/json.hpp"

using nlohmann::json;

static int free_port() {
    static std::mt19937 rng(std::random_device{}());
    return 20000 + static_cast<int>(rng() % 10000);
}

static json task_body() {
    return json::parse(R"({
      "task_id": 321,
      "camera": {"id": 7, "name": "北门", "url": "test_data/test_video.mp4", "transport": "tcp"},
      "models": [
        {"name": "det", "type": "det", "backend": "ort", "device": "cpu",
         "path": "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx",
         "labels": ["person"], "input_size": [640,640], "confidence_threshold": 0.4}
      ],
      "preview": {"enabled": false},
      "events": {"transport": "http", "http": {"url": "http://127.0.0.1:1/cb"}, "buffer": {"dir": "./events_buffer", "max_mb": 8}}
    })");
}

TEST_CASE("AgentServer health and readyz", "[agent][server]") {
    PipelineManager mgr;
    ConfigAdapter adapter;
    int port = free_port();
    AgentServer srv(mgr, adapter, "127.0.0.1", port);
    REQUIRE(srv.start());
    httplib::Client cli("127.0.0.1", port);
    auto h = cli.Get("/health");
    REQUIRE(h);
    REQUIRE(h->status == 200);
    auto r = cli.Get("/readyz");
    REQUIRE(r);
    REQUIRE(r->status == 200);
    srv.stop();
}

TEST_CASE("AgentServer task lifecycle", "[agent][server]") {
    PipelineManager mgr;
    ConfigAdapter adapter;
    int port = free_port();
    AgentServer srv(mgr, adapter, "127.0.0.1", port);
    REQUIRE(srv.start());
    httplib::Client cli("127.0.0.1", port);

    auto created = cli.Post("/api/v1/tasks", task_body().dump(), "application/json");
    REQUIRE(created);
    REQUIRE(created->status == 200);
    REQUIRE(json::parse(created->body)["task_id"] == "321");

    auto list = cli.Get("/api/v1/tasks");
    REQUIRE(list);
    REQUIRE(list->status == 200);
    auto lj = json::parse(list->body);
    REQUIRE(lj["items"].size() == 1);
    REQUIRE(lj["items"][0]["task_id"] == "321");

    auto one = cli.Get("/api/v1/tasks/321");
    REQUIRE(one);
    REQUIRE(one->status == 200);

    auto start = cli.Post("/api/v1/tasks/321/start");
    REQUIRE(start);
    REQUIRE(start->status == 200);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    auto stats = cli.Get("/api/v1/tasks/321/stats");
    REQUIRE(stats);
    REQUIRE(stats->status == 200);
    REQUIRE(json::parse(stats->body).contains("stats"));

    auto stop = cli.Post("/api/v1/tasks/321/stop");
    REQUIRE(stop);
    REQUIRE(stop->status == 200);

    auto del = cli.Delete("/api/v1/tasks/321");
    REQUIRE(del);
    REQUIRE(del->status == 200);
    REQUIRE(mgr.task_count() == 0);
    srv.stop();
}

TEST_CASE("AgentServer bearer auth", "[agent][server]") {
    PipelineManager mgr;
    ConfigAdapter adapter;
    int port = free_port();
    AgentServer srv(mgr, adapter, "127.0.0.1", port);
    srv.set_api_key("sk-agent");
    REQUIRE(srv.start());
    httplib::Client cli("127.0.0.1", port);
    auto r1 = cli.Get("/api/v1/tasks");
    REQUIRE(r1);
    REQUIRE(r1->status == 401);
    REQUIRE(json::parse(r1->body)["error"]["code"] == "UNAUTHORIZED");
    auto r2 = cli.Get("/api/v1/tasks", httplib::Headers{{"Authorization", "Bearer sk-agent"}});
    REQUIRE(r2);
    REQUIRE(r2->status == 200);
    srv.stop();
}

TEST_CASE("AgentServer bad config returns 400", "[agent][server]") {
    PipelineManager mgr;
    ConfigAdapter adapter;
    int port = free_port();
    AgentServer srv(mgr, adapter, "127.0.0.1", port);
    REQUIRE(srv.start());
    httplib::Client cli("127.0.0.1", port);
    auto bad = cli.Post("/api/v1/tasks", "{}", "application/json");
    REQUIRE(bad);
    REQUIRE(bad->status == 400);
    REQUIRE(json::parse(bad->body)["error"]["code"] == "BAD_REQUEST");
    auto missing = cli.Get("/api/v1/tasks/does-not-exist");
    REQUIRE(missing);
    REQUIRE(missing->status == 404);
    srv.stop();
}
