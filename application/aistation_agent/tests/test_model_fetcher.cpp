#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>
#include "model_fetcher.hpp"
#include "httplib.h"

namespace fs = std::filesystem;

static int free_port() {
    static std::mt19937 rng(std::random_device{}());
    return 24000 + static_cast<int>(rng() % 3000);
}

TEST_CASE("ModelFetcher local path", "[agent][fetch]") {
    auto dir = fs::temp_directory_path() / "md_fetch_local";
    fs::remove_all(dir); fs::create_directories(dir);
    auto file = dir / "model.onnx";
    { std::ofstream f(file); f << "dummy"; }
    ModelFetcher fetcher(dir.string());
    std::string out, err;
    REQUIRE(fetcher.fetch(file.string(), &out, &err));
    REQUIRE(fs::exists(out));
    REQUIRE_FALSE(fetcher.fetch((dir / "missing.onnx").string(), &out, &err));
    REQUIRE_FALSE(err.empty());
    fs::remove_all(dir);
}

TEST_CASE("ModelFetcher http download", "[agent][fetch]") {
    int port = free_port();
    httplib::Server srv;
    srv.Get("/yolo.onnx", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("fake-onnx-bytes", "application/octet-stream");
    });
    std::thread t([&]() { srv.listen("127.0.0.1", port); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto dir = fs::temp_directory_path() / "md_fetch_http";
    fs::remove_all(dir);
    ModelFetcher fetcher(dir.string());
    std::string out, err;
    REQUIRE(fetcher.fetch("http://127.0.0.1:" + std::to_string(port) + "/yolo.onnx", &out, &err));
    REQUIRE(fs::exists(out));
    REQUIRE(fs::file_size(out) > 0);
    // 二次拉取命中缓存
    REQUIRE(fetcher.fetch("http://127.0.0.1:" + std::to_string(port) + "/yolo.onnx", &out, &err));
    srv.stop(); t.join();
    fs::remove_all(dir);
}

TEST_CASE("ModelFetcher s3 requires endpoint", "[agent][fetch]") {
    auto dir = fs::temp_directory_path() / "md_fetch_s3";
    fs::remove_all(dir);
    ModelFetcher fetcher(dir.string());
    std::string out, err;
    REQUIRE_FALSE(fetcher.fetch("s3://bucket/model.onnx", &out, &err));
    REQUIRE(err.find("s3") != std::string::npos);
    fs::remove_all(dir);
}
