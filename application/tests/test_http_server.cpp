#include <catch2/catch_test_macros.hpp>
#include "http_server.hpp"

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
