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

// 模型热更新请求体（单模型 ModelConfig JSON）；默认指向真实可加载模型。
static json model_update_body(const std::string& name, float conf,
                              const std::string& path =
                                  "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx") {
    return json{{"name", name}, {"type", "det"}, {"backend", "ort"}, {"device", "cpu"},
                {"path", path}, {"labels", json::array({"person"})},
                {"input_size", json::array({640, 640})}, {"confidence_threshold", conf}};
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
    auto snap = cli.Get("/api/v1/tasks/does-not-exist/snapshot.jpg");
    REQUIRE(snap);
    REQUIRE(snap->status == 404);
    REQUIRE(json::parse(snap->body)["error"]["code"] == "NOT_FOUND");
    srv.stop();
}

// SP6-c Task 2：模型热更新控制面（COW 热交换）
// 约定：模型名必须已存在（本接口只更新既有模型，不隐式追加）；
//       引擎未就绪/加载失败一律 fail-safe（ok:false 且 generation 不递增，旧模型继续服务）。
TEST_CASE("AgentServer model hot update", "[agent][server]") {
    PipelineManager mgr;
    ConfigAdapter adapter;
    int port = free_port();
    AgentServer srv(mgr, adapter, "127.0.0.1", port);
    REQUIRE(srv.start());
    httplib::Client cli("127.0.0.1", port);

    auto created = cli.Post("/api/v1/tasks", task_body().dump(), "application/json");
    REQUIRE(created);
    REQUIRE(created->status == 200);

    // 1) 任务不存在 → 404
    auto no_task = cli.Post("/api/v1/tasks/does-not-exist/models/det/update",
                            model_update_body("det", 0.3f).dump(), "application/json");
    REQUIRE(no_task);
    REQUIRE(no_task->status == 404);
    REQUIRE(json::parse(no_task->body)["ok"] == false);
    REQUIRE(srv.generation() == 0);

    // 2) 模型名不存在 → 404（约定：不隐式追加）
    auto no_model = cli.Post("/api/v1/tasks/321/models/ghost/update",
                             model_update_body("ghost", 0.3f).dump(), "application/json");
    REQUIRE(no_model);
    REQUIRE(no_model->status == 404);
    REQUIRE(json::parse(no_model->body)["ok"] == false);
    REQUIRE(srv.generation() == 0);

    // 3) 非法 ModelConfig → 400，generation 不递增（旧模型保留）
    auto bad = cli.Post("/api/v1/tasks/321/models/det/update", "{}", "application/json");
    REQUIRE(bad);
    REQUIRE(bad->status == 400);
    REQUIRE(json::parse(bad->body)["ok"] == false);
    REQUIRE(srv.generation() == 0);
    // 控制面仍可见旧模型（det 未被移除）
    auto one = cli.Get("/api/v1/tasks/321");
    REQUIRE(one);
    REQUIRE(json::parse(one->body)["task"]["models"][0]["name"] == "det");

    // 4) 引擎未就绪（路径不存在 → create_engine 返回非空但 is_loaded()==false）→
    //    ok:false，generation 不递增，旧模型保留（fail-safe）
    auto unready = cli.Post("/api/v1/tasks/321/models/det/update",
                            model_update_body("det", 0.9f, "/nonexistent/det.onnx").dump(),
                            "application/json");
    REQUIRE(unready);
    REQUIRE(unready->status == 200);
    REQUIRE(json::parse(unready->body)["ok"] == false);
    REQUIRE(srv.generation() == 0);

    // 5) 成功热更新 → ok:true 且 generation 递增到 1
    auto ok1 = cli.Post("/api/v1/tasks/321/models/det/update",
                        model_update_body("det", 0.3f).dump(), "application/json");
    REQUIRE(ok1);
    REQUIRE(ok1->status == 200);
    auto j1 = json::parse(ok1->body);
    REQUIRE(j1["ok"] == true);
    REQUIRE(j1["generation"] == 1);
    REQUIRE(srv.generation() == 1);

    // 6) 再次成功 → generation 递增到 2
    auto ok2 = cli.Post("/api/v1/tasks/321/models/det/update",
                        model_update_body("det", 0.35f).dump(), "application/json");
    REQUIRE(ok2);
    REQUIRE(ok2->status == 200);
    REQUIRE(json::parse(ok2->body)["generation"] == 2);
    REQUIRE(srv.generation() == 2);
    srv.stop();
}

TEST_CASE("AgentServer start is idempotent and restartable", "[agent][server]") {
    PipelineManager mgr;
    ConfigAdapter adapter;
    int port = free_port();
    AgentServer srv(mgr, adapter, "127.0.0.1", port);
    REQUIRE(srv.start());
    REQUIRE(srv.is_running());
    REQUIRE(srv.start());                 // 已在运行时再次 start：不重绑、不重复注册路由
    REQUIRE(srv.start());
    httplib::Client cli("127.0.0.1", port);
    for (int i = 0; i < 3; ++i) {
        auto h = cli.Get("/health");
        REQUIRE(h);
        REQUIRE(h->status == 200);
    }
    srv.stop();
    REQUIRE_FALSE(srv.is_running());
    REQUIRE(srv.start());                 // stop() 后重启（路由仍幂等）
    REQUIRE(srv.start());
    httplib::Client cli2("127.0.0.1", port);
    for (int i = 0; i < 3; ++i) {
        auto h2 = cli2.Get("/health");
        REQUIRE(h2);
        REQUIRE(h2->status == 200);
    }
    srv.stop();
}

TEST_CASE("AgentServer start fails fast on unbindable host", "[agent][server]") {
    PipelineManager mgr;
    ConfigAdapter adapter;
    // TEST-NET-1 地址，非本机接口，bind 立即失败（不依赖 sleep 判活）。
    AgentServer srv(mgr, adapter, "192.0.2.1", 39117);
    REQUIRE_FALSE(srv.start());
    REQUIRE_FALSE(srv.is_running());
}
