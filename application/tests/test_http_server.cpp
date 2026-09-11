#include <catch2/catch_test_macros.hpp>
#include "http_server.hpp"
#include "httplib.h"
#include "nlohmann/json.hpp"

TEST_CASE("HttpServer construct/destroy", "[http]") {
    PipelineManager mgr;
    HttpServer srv(mgr, "127.0.0.1", 18080);
    REQUIRE_FALSE(srv.is_running());
}

TEST_CASE("HttpServer start/stop", "[http]") {
    PipelineManager mgr;
    HttpServer srv(mgr, "127.0.0.1", 18081);
    REQUIRE(srv.start());
    REQUIRE(srv.is_running());
    srv.stop();
    REQUIRE_FALSE(srv.is_running());
}

TEST_CASE("HttpServer double start", "[http]") {
    PipelineManager mgr;
    HttpServer srv(mgr, "127.0.0.1", 18082);
    REQUIRE(srv.start());
    REQUIRE(srv.start());
    srv.stop();
}

TEST_CASE("HttpServer stop without start", "[http]") {
    PipelineManager mgr;
    HttpServer srv(mgr, "127.0.0.1", 18083);
    REQUIRE_NOTHROW(srv.stop());
}

TEST_CASE("HttpServer bearer auth", "[http]") {
    PipelineManager mgr;
    HttpServer srv(mgr, "127.0.0.1", 18084);
    srv.set_api_keys({"sk-test"});
    REQUIRE(srv.start());
    httplib::Client cli("127.0.0.1", 18084);

    // 无 token → 401 统一错误体
    auto r1 = cli.Get("/api/v1/models");
    REQUIRE(r1);
    REQUIRE(r1->status == 401);
    auto j = nlohmann::json::parse(r1->body);
    REQUIRE(j["error"]["code"] == "UNAUTHORIZED");

    // 正确 token → 200
    auto r2 = cli.Get("/api/v1/models", httplib::Headers{{"Authorization", "Bearer sk-test"}});
    REQUIRE(r2);
    REQUIRE(r2->status == 200);

    // /health 放行（无需 token）
    auto r3 = cli.Get("/health");
    REQUIRE(r3);
    REQUIRE(r3->status == 200);

    srv.stop();
}

TEST_CASE("HttpServer cors + options preflight", "[http]") {
    PipelineManager mgr;
    HttpServer srv(mgr, "127.0.0.1", 18085);
    REQUIRE(srv.start());
    httplib::Client cli("127.0.0.1", 18085);

    auto opt = cli.Options("/api/v1/tasks", httplib::Headers{{"Origin", "http://x"}});
    REQUIRE(opt);
    REQUIRE(opt->status == 204);
    REQUIRE(opt->get_header_value("Access-Control-Allow-Origin") == "*");

    auto g = cli.Get("/api/v1/models", httplib::Headers{{"Origin", "http://x"}});
    REQUIRE(g);
    REQUIRE(g->get_header_value("Access-Control-Allow-Origin") == "*");

    srv.stop();
}
