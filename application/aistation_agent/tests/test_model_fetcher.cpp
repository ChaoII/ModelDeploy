#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <thread>
#include <vector>
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

TEST_CASE("ModelFetcher concurrent downloads share one dest safely", "[agent][fetch]") {
    int port = free_port();
    httplib::Server srv;
    // 全部线程请求同一 URL（同一 dest）：强制并发命中同一 temp_path_for(dest)，
    // 检验同名临时文件不会互相覆盖/污染。
    const std::string payload = "shared-model-bytes";
    srv.Get("/model.onnx", [&](const httplib::Request&, httplib::Response& res) {
        // 略作延迟，确保所有线程在任一请求完成前都已进入下载路径，放大并发窗口。
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        res.set_content(payload, "application/octet-stream");
    });
    std::thread t([&]() { srv.listen("127.0.0.1", port); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto dir = fs::temp_directory_path() / "md_fetch_concurrent";
    fs::remove_all(dir);
    const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/model.onnx";
    constexpr int kThreads = 8;
    std::atomic<int> ok{0};
    std::atomic<int> empty{0};
    std::vector<std::thread> workers;
    for (int i = 0; i < kThreads; ++i) {
        workers.emplace_back([&]() {
            ModelFetcher fetcher(dir.string());
            std::string out, err;
            if (!fetcher.fetch(url, &out, &err)) return;
            // 并发同名 dest：所有线程共享同一临时写入路径。fetch 返回即成功。
            // 注意不要在 worker 内读取 dest（会与其它线程的原子替换争用共享句柄，
            // 属测试自身的竞态）；内容正确性在全部 join 后确定性校验。
            if (out.empty()) empty.fetch_add(1);
            ok.fetch_add(1);
        });
    }
    for (auto& w : workers) w.join();
    REQUIRE(ok.load() == kThreads);
    REQUIRE(empty.load() == 0);

    // 同一 URL 只产出一个最终文件，且不残留任何 .tmp- 临时文件。
    int files = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
        ++files;
        REQUIRE(e.path().filename().string().find(".tmp-") == std::string::npos);
    }
    REQUIRE(files == 1);
    REQUIRE(fs::file_size(dir / "model.onnx") == payload.size());
    // 并发写同名临时文件 + 原子 rename 到同一 dest 后，最终内容必须完整无损。
    {
        std::ifstream f(dir / "model.onnx", std::ios::binary);
        std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        REQUIRE(body == payload);
    }

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

TEST_CASE("ModelFetcher rejects url without filename", "[agent][fetch]") {
    auto dir = fs::temp_directory_path() / "md_fetch_noname";
    fs::remove_all(dir);
    ModelFetcher fetcher(dir.string());
    std::string out = "sentinel", err;
    REQUIRE_FALSE(fetcher.fetch("http://127.0.0.1:1/", &out, &err));
    REQUIRE(out == "sentinel");
    REQUIRE(err.find("filename") != std::string::npos);
    REQUIRE_FALSE(fetcher.fetch("http://127.0.0.1:1/..", &out, &err));
    REQUIRE(out == "sentinel");
    fs::remove_all(dir);
}

TEST_CASE("ModelFetcher failed download leaves no file", "[agent][fetch]") {
    int port = free_port();
    httplib::Server srv;
    srv.Get("/ok.onnx", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("bytes", "application/octet-stream");
    });
    std::thread t([&]() { srv.listen("127.0.0.1", port); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto dir = fs::temp_directory_path() / "md_fetch_atomic";
    fs::remove_all(dir);
    ModelFetcher fetcher(dir.string());
    std::string out = "sentinel", err;

    REQUIRE_FALSE(fetcher.fetch("http://127.0.0.1:" + std::to_string(port) + "/missing.onnx", &out, &err));
    REQUIRE(out == "sentinel");
    REQUIRE_FALSE(err.empty());
    REQUIRE(fs::is_empty(dir));

    REQUIRE(fetcher.fetch("http://127.0.0.1:" + std::to_string(port) + "/ok.onnx", &out, &err));
    int files = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
        ++files;
        REQUIRE(e.path().filename().string().find(".tmp-") == std::string::npos);
    }
    REQUIRE(files == 1);

    srv.stop(); t.join();
    fs::remove_all(dir);
}
