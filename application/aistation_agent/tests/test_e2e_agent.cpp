#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "agent_runtime.hpp"
#include "httplib.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

static int free_port() {
    static std::mt19937 rng(std::random_device{}());
    return 25000 + static_cast<int>(rng() % 4000);
}

TEST_CASE("Agent E2E: local det emits HTTP event and serves snapshot", "[agent][e2e]") {
    if (!std::ifstream("test_data/test_video60.mp4").good() ||
        !std::ifstream("test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx").good())
        SKIP("test data absent");

    // HTTP 事件接收器
    const int recv_port = free_port();
    httplib::Server recv;
    std::atomic<int> events{0};
    std::string last_body;
    recv.Post("/cb", [&](const httplib::Request& req, httplib::Response& res) {
        ++events; last_body = req.body; res.status = 200;
    });
    std::thread recv_thread([&]() { recv.listen("127.0.0.1", recv_port); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    const int agent_port = free_port();
    auto buf = fs::temp_directory_path() / "md_e2e_buffer";
    fs::remove_all(buf);

    AgentOptions o;
    o.host = "127.0.0.1"; o.port = agent_port;
    o.edge_code = "edge-e2e";
    AgentRuntime rt(o);
    REQUIRE(rt.start());

    json body = {
        {"task_id", 777},
        {"camera", {{"id", 7}, {"name", "cam7"}, {"url", "test_data/test_video60.mp4"}, {"transport", "tcp"}}},
        {"models", json::array({{{"name", "det"}, {"type", "det"}, {"backend", "ort"}, {"device", "cpu"},
            {"path", "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx"},
            {"labels", json::array({"person"})}, {"input_size", json::array({640, 640})},
            {"confidence_threshold", 0.35}}})},
        {"alarm_interval_sec", 0},
        {"algorithm_type", "INTRUSION"},
        {"preview", {{"enabled", false}}},
        {"events", {{"transport", "http"},
                    {"http", {{"url", "http://127.0.0.1:" + std::to_string(recv_port) + "/cb"}, {"token", "tk"}}},
                    {"buffer", {{"dir", buf.string()}, {"max_mb", 8}}}}}
    };

    httplib::Client cli("127.0.0.1", agent_port);
    auto created = cli.Post("/api/v1/tasks", body.dump(), "application/json");
    REQUIRE(created);
    REQUIRE(json::parse(created->body)["task_id"] == "777");

    auto started = cli.Post("/api/v1/tasks/777/start");
    REQUIRE(started);
    REQUIRE(started->status == 200);

    // 等待首个事件（含模型加载 + 推理）
    for (int i = 0; i < 300 && events.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    REQUIRE(events.load() >= 1);
    auto e = json::parse(last_body);
    REQUIRE(e["edge_code"] == "edge-e2e");
    REQUIRE(e["camera_id"] == 7);
    REQUIRE(e["task_id"] == 777);
    REQUIRE(e["detections"].is_array());
    REQUIRE(!e["detections"].empty());
    REQUIRE(e["event_id"].is_string());
    REQUIRE(e["event_id"].get<std::string>().size() == 36);
    REQUIRE(e["ts"].is_string());
    REQUIRE(!e["ts"].get<std::string>().empty());
    REQUIRE(e["ts"].get<std::string>().back() == 'Z');
    REQUIRE(e["schema_version"] == 1);

    // 快照
    bool got_snapshot = false;
    for (int i = 0; i < 100; ++i) {
        auto snap = cli.Get("/api/v1/tasks/777/snapshot.jpg");
        if (snap && snap->status == 200 && snap->body.size() > 100) { got_snapshot = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    REQUIRE(got_snapshot);

    // 并发冒烟：持续拉取 /api/v1/metrics 的同时反复 stop/start 任务，覆盖
    // metrics()（旧实现持 mtx_ 再取 PipelineManager::mtx_）与 stop_task（持
    // PipelineManager::mtx_ 并 join 检测线程，检测线程再经 sink 取 mtx_）
    // 的相反锁序路径。修复后应稳定完成，不出现 ABBA 死锁；客户端设读超时，
    // 避免在缺陷回归时无限阻塞。
    std::atomic<bool> hammer{true};
    std::thread metrics_thread([&]() {
        httplib::Client mc("127.0.0.1", agent_port);
        mc.set_connection_timeout(2, 0);
        mc.set_read_timeout(5, 0);
        while (hammer.load()) {
            auto r = mc.Get("/api/v1/metrics");
            (void)r;
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    for (int round = 0; round < 3; ++round) {
        auto st = cli.Post("/api/v1/tasks/777/stop");
        REQUIRE(st);
        auto s2 = cli.Post("/api/v1/tasks/777/start");
        REQUIRE(s2);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    hammer = false;
    metrics_thread.join();

    auto stopped = cli.Post("/api/v1/tasks/777/stop");
    REQUIRE(stopped);
    auto deleted = cli.Delete("/api/v1/tasks/777");
    REQUIRE(deleted);
    REQUIRE(rt.manager().task_count() == 0);

    rt.stop();
    recv.stop(); recv_thread.join();
    fs::remove_all(buf);
}

#ifdef ENABLE_MQTT
#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using sock_t = SOCKET;
#define CLOSE_SOCK closesocket
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
using sock_t = int;
#define CLOSE_SOCK close
#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif
#endif

namespace {
bool was_timeout() {
#if defined(_WIN32)
    return WSAGetLastError() == WSAETIMEDOUT;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

void set_recv_timeout(sock_t s, int ms) {
#if defined(_WIN32)
    DWORD tv = static_cast<DWORD>(ms);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));
#else
    timeval tv{};
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

// 进程内 mock broker（与 Task 6 test_mqtt_publisher.cpp 相同实现）
struct MockBroker {
    int port = 0;
    std::atomic<bool> running{true};
    std::thread th;
    std::string topic;
    std::string payload;
    std::atomic<int> publishes{0};

    ~MockBroker() { stop(); }

    static uint32_t read_varint(sock_t s, int first) {
        uint32_t mult = 1, value = 0;
        int b = first;
        while (true) {
            value += (b & 127) * mult; mult *= 128;
            if (!(b & 128)) break;
            unsigned char c;
            if (recv(s, reinterpret_cast<char*>(&c), 1, 0) != 1) return 0;
            b = c;
        }
        return value;
    }
    static bool read_n(sock_t s, char* buf, int n) {
        int got = 0;
        while (got < n) { int r = recv(s, buf + got, n - got, 0); if (r <= 0) return false; got += r; }
        return true;
    }
    void respond(sock_t c, const char* data, int n) { send(c, data, n, 0); }
    void serve() {
        sock_t listener = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{}; addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(static_cast<unsigned short>(port));
        bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        listen(listener, 1);
        sock_t c = INVALID_SOCKET;
        while (running.load()) {
            // Windows 下 SO_RCVTIMEO 不约束 accept，改用 select 带超时轮询，
            // 即使无客户端连接也能让 stop() 及时返回。
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(listener, &rfds);
            timeval tv{};
            tv.tv_sec = 0;
            tv.tv_usec = 200000;
            const int sel = select(static_cast<int>(listener) + 1, &rfds, nullptr, nullptr, &tv);
            if (sel == 0) continue;
            if (sel < 0) { CLOSE_SOCK(listener); return; }
            c = accept(listener, nullptr, nullptr);
            if (c != INVALID_SOCKET) break;
            if (!running.load()) break;
        }
        if (c == INVALID_SOCKET) { CLOSE_SOCK(listener); return; }
        set_recv_timeout(c, 300);
        while (running.load()) {
            unsigned char hdr;
            const int rh = recv(c, reinterpret_cast<char*>(&hdr), 1, 0);
            if (rh != 1) {
                if (rh < 0 && was_timeout()) continue;
                break;
            }
            const int type = (hdr >> 4) & 0x0F;
            unsigned char lb;
            if (recv(c, reinterpret_cast<char*>(&lb), 1, 0) != 1) break;
            const uint32_t rem = read_varint(c, lb);
            std::vector<char> body(rem);
            if (rem > 0 && !read_n(c, body.data(), static_cast<int>(rem))) break;
            if (type == 1) {            // CONNECT -> CONNACK
                const char connack[4] = {0x20, 0x02, 0x00, 0x00};
                respond(c, connack, 4);
            } else if (type == 3) {     // PUBLISH
                if (rem < 2) continue;
                const int tlen = (static_cast<unsigned char>(body[0]) << 8) | static_cast<unsigned char>(body[1]);
                topic.assign(body.data() + 2, tlen);
                int off = 2 + tlen;
                const int qos = (hdr >> 1) & 0x03;
                int pid = 0;
                if (qos > 0 && rem >= static_cast<uint32_t>(off + 2)) {
                    pid = (static_cast<unsigned char>(body[off]) << 8) | static_cast<unsigned char>(body[off + 1]);
                    off += 2;
                }
                payload.assign(body.data() + off, rem - off);
                ++publishes;
                if (qos == 1) {
                    const char puback[4] = {0x40, 0x02, static_cast<char>(pid >> 8), static_cast<char>(pid & 0xFF)};
                    respond(c, puback, 4);
                }
            } else if (type == 12) {    // PINGREQ -> PINGRESP
                const char pingresp[2] = {static_cast<char>(0xD0), 0x00};
                respond(c, pingresp, 2);
            } else if (type == 14) {    // DISCONNECT
                break;
            }
        }
        CLOSE_SOCK(c);
        CLOSE_SOCK(listener);
    }
    void start() { th = std::thread([this]() { serve(); }); }
    void stop() {
        running = false;
        // 自连接唤醒可能阻塞在 select/accept 的服务线程，保证快速返回。
        sock_t w = socket(AF_INET, SOCK_STREAM, 0);
        if (w != INVALID_SOCKET) {
            sockaddr_in a{}; a.sin_family = AF_INET;
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            a.sin_port = htons(static_cast<unsigned short>(port));
            connect(w, reinterpret_cast<sockaddr*>(&a), sizeof(a));
            CLOSE_SOCK(w);
        }
        if (th.joinable()) th.join();
    }
};
}  // namespace
#endif  // ENABLE_MQTT

#ifdef ENABLE_MQTT
TEST_CASE("Agent E2E: local det emits MQTT event", "[agent][e2e][mqtt]") {
    if (!std::ifstream("test_data/test_video60.mp4").good() ||
        !std::ifstream("test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx").good())
        SKIP("test data absent");
#if defined(_WIN32)
    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    MockBroker broker; broker.port = free_port();
    broker.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    const int agent_port = free_port();
    auto buf = fs::temp_directory_path() / "md_e2e_buffer_mqtt";
    fs::remove_all(buf);

    AgentOptions o;
    o.host = "127.0.0.1"; o.port = agent_port;
    o.edge_code = "edge-e2e";
    o.heartbeat_interval_sec = 3600;
    AgentRuntime rt(o);
    REQUIRE(rt.start());

    json body = {
        {"task_id", 888},
        {"camera", {{"id", 8}, {"name", "cam8"}, {"url", "test_data/test_video60.mp4"}, {"transport", "tcp"}}},
        {"models", json::array({{{"name", "det"}, {"type", "det"}, {"backend", "ort"}, {"device", "cpu"},
            {"path", "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx"},
            {"labels", json::array({"person"})}, {"input_size", json::array({640, 640})},
            {"confidence_threshold", 0.35}}})},
        {"alarm_interval_sec", 0},
        {"algorithm_type", "INTRUSION"},
        {"preview", {{"enabled", false}}},
        {"events", {{"transport", "mqtt"},
                    {"mqtt", {{"broker", "tcp://127.0.0.1:" + std::to_string(broker.port)},
                              {"topic", ""}, {"qos", 1}}},
                    {"buffer", {{"dir", buf.string()}, {"max_mb", 8}}}}}
    };

    httplib::Client cli("127.0.0.1", agent_port);
    auto created = cli.Post("/api/v1/tasks", body.dump(), "application/json");
    REQUIRE(created);
    REQUIRE(json::parse(created->body)["task_id"] == "888");

    auto started = cli.Post("/api/v1/tasks/888/start");
    REQUIRE(started);
    REQUIRE(started->status == 200);

    for (int i = 0; i < 300 && broker.publishes.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    REQUIRE(broker.publishes.load() >= 1);
    REQUIRE(broker.topic == "aistation/default/edge/edge-e2e/camera/8/detect");
    auto me = json::parse(broker.payload);
    REQUIRE(me["edge_code"] == "edge-e2e");
    REQUIRE(me["camera_id"] == 8);
    REQUIRE(me["task_id"] == 888);
    REQUIRE(me["detections"].is_array());
    REQUIRE(!me["detections"].empty());
    REQUIRE(me["event_id"].is_string());
    REQUIRE(me["event_id"].get<std::string>().size() == 36);
    REQUIRE(me["ts"].is_string());
    REQUIRE(!me["ts"].get<std::string>().empty());
    REQUIRE(me["ts"].get<std::string>().back() == 'Z');
    REQUIRE(me["schema_version"] == 1);

    auto stopped = cli.Post("/api/v1/tasks/888/stop");
    REQUIRE(stopped);
    auto deleted = cli.Delete("/api/v1/tasks/888");
    REQUIRE(deleted);

    rt.stop();
    broker.stop();
    fs::remove_all(buf);
}
#endif  // ENABLE_MQTT
