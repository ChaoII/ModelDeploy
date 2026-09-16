#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "agent_runtime.hpp"
#include "httplib.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

/// 退出保护：无论测试因断言失败抛异常还是正常返回，析构时都先执行 cleanup（如
/// stop() 服务、置位停止标志）再 join 线程。避免可 join 线程在栈展开时析构触发
/// std::terminate —— Windows 上表现为 0xC0000409 进程崩溃，掩盖真实断言失败。
template <typename Cleanup>
struct JoinGuard {
    std::thread& th;
    Cleanup cleanup;
    ~JoinGuard() {
        cleanup();
        if (th.joinable()) th.join();
    }
};

/// 判定事件体是否为空检测心跳（heartbeat=true 或 detections 为空数组）。
bool is_heartbeat_event(const json& e) {
    if (!e.is_object()) return true;
    if (e.value("heartbeat", false)) return true;
    if (!e.contains("detections") || !e["detections"].is_array()) return true;
    return e["detections"].empty();
}

}  // namespace

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
    std::string detection_body;  // 首个真实检测事件体（跳过心跳），接收线程写、主线程读
    std::string snapshot_body;   // 首个携带内联快照的事件体
    std::mutex body_mu;          // 保护上述两个字段
    recv.Post("/cb", [&](const httplib::Request& req, httplib::Response& res) {
        {
            std::lock_guard<std::mutex> lk(body_mu);
            // 就地记录首个匹配事件：心跳会持续覆盖"最新事件"，若只轮询最新值，
            // 已收到的检测事件可能在两次轮询之间被心跳覆盖而丢失。
            if (detection_body.empty()) {
                const json parsed = json::parse(req.body, nullptr, false);
                if (parsed.is_object() && !is_heartbeat_event(parsed)) detection_body = req.body;
            }
            if (snapshot_body.empty() && req.body.find("\"snapshot\"") != std::string::npos)
                snapshot_body = req.body;
        }
        res.status = 200;
    });
    std::thread recv_thread([&]() { recv.listen("127.0.0.1", recv_port); });
    JoinGuard recv_guard{recv_thread, [&]() { recv.stop(); }};
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
                    {"buffer", {{"dir", buf.string()}, {"max_mb", 8}}},
                    {"snapshot", {{"enabled", true}, {"inline", true}, {"quality", 50}, {"max_width", 320}}}}}
    };

    httplib::Client cli("127.0.0.1", agent_port);
    auto created = cli.Post("/api/v1/tasks", body.dump(), "application/json");
    REQUIRE(created);
    REQUIRE(json::parse(created->body)["task_id"] == "777");

    auto started = cli.Post("/api/v1/tasks/777/start");
    REQUIRE(started);
    REQUIRE(started->status == 200);

    // 等待首个"真实检测"事件（含模型加载 + 推理）。SP4-b 起任务运行期会周期性上报
    // 心跳（heartbeat=true、detections 为空），心跳同样经事件 sink 送达本接收器，
    // 因此必须跳过心跳事件，否则会在空 detections 上误判失败。
    std::string got_body;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lk(body_mu);
            got_body = detection_body;
        }
        if (!got_body.empty()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    REQUIRE_FALSE(got_body.empty());  // 30s 内未等到真实检测事件（可能只收到心跳）
    json e = json::parse(got_body);
    REQUIRE(e["edge_code"] == "edge-e2e");
    REQUIRE(e["camera_id"] == 7);
    REQUIRE(e["task_id"] == 777);
    REQUIRE(e["detections"].is_array());
    REQUIRE(!e["detections"].empty());
    REQUIRE(e.contains("objects"));
    REQUIRE(e["event_id"].is_string());
    REQUIRE(e["event_id"].get<std::string>().size() == 36);
    REQUIRE(e["ts"].is_string());
    REQUIRE(!e["ts"].get<std::string>().empty());
    REQUIRE(e["ts"].get<std::string>().back() == 'Z');
    REQUIRE(e["schema_version"] == 2);
    // 内联快照为尽力而为：首个检测事件可能早于 Pipeline 首个 BGR 快照就绪
    // （快照每 2 帧更新一次），故等待后续携带 snapshot 的事件再断言。
    std::string with_snapshot;
    for (int i = 0; i < 100; ++i) {
        {
            std::lock_guard<std::mutex> lk(body_mu);
            with_snapshot = snapshot_body;
        }
        if (!with_snapshot.empty()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    REQUIRE(with_snapshot.find("\"snapshot\"") != std::string::npos);
    REQUIRE(with_snapshot.find("\"data\"") != std::string::npos);

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
    // 断言失败时也必须先停止压测再 join，避免线程在栈展开时析构触发 std::terminate。
    JoinGuard metrics_guard{metrics_thread, [&]() { hammer = false; }};
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

TEST_CASE("Agent E2E: shutdown with a running task joins detection before teardown",
          "[agent][e2e][shutdown]") {
    if (!std::ifstream("test_data/test_video60.mp4").good() ||
        !std::ifstream("test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx").good())
        SKIP("test data absent");

    const int agent_port = free_port();
    auto buf = fs::temp_directory_path() / "md_e2e_buffer_shutdown";
    fs::remove_all(buf);

    AgentOptions o;
    o.host = "127.0.0.1"; o.port = agent_port;
    o.edge_code = "edge-shutdown";
    AgentRuntime rt(o);
    REQUIRE(rt.start());

    // http 回调指向不可达端口：队列会积压并在停机时被安全停止/销毁，正好覆盖
    // “检测线程仍在经 sink 向 DurableQueue enqueue 时直接停机” 的 UAF 路径。
    json body = {
        {"task_id", 999},
        {"camera", {{"id", 9}, {"name", "cam9"}, {"url", "test_data/test_video60.mp4"}, {"transport", "tcp"}}},
        {"models", json::array({{{"name", "det"}, {"type", "det"}, {"backend", "ort"}, {"device", "cpu"},
            {"path", "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx"},
            {"labels", json::array({"person"})}, {"input_size", json::array({640, 640})},
            {"confidence_threshold", 0.35}}})},
        {"alarm_interval_sec", 0},
        {"algorithm_type", "INTRUSION"},
        {"preview", {{"enabled", false}}},
        {"events", {{"transport", "http"},
                    {"http", {{"url", "http://127.0.0.1:1/cb"}, {"token", "tk"}}},
                    {"buffer", {{"dir", buf.string()}, {"max_mb", 8}}}}}
    };

    httplib::Client cli("127.0.0.1", agent_port);
    auto created = cli.Post("/api/v1/tasks", body.dump(), "application/json");
    REQUIRE(created);
    auto started = cli.Post("/api/v1/tasks/999/start");
    REQUIRE(started);
    REQUIRE(started->status == 200);
    REQUIRE(rt.manager().task_count() == 1);

    // 让检测线程进入 sink → enqueue 路径后再直接停机（不先 stop/delete 任务）。
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    rt.stop();
    fs::remove_all(buf);
}

// 单进程多任务回归：在持续发送心跳（heartbeat_sec=1）的同时反复删除/重建其中一个运行中
// 任务，覆盖 on_removed 摘除并停止 DurableQueue 与 heartbeat_loop 经 EventBus sink 调用
// 队列 enqueue 的竞态（原实现在锁外持裸 DurableQueue* → 写已释放队列，堆破坏/0xC0000005）。
// 修复后队列由 shared_ptr 持有，在途 enqueue 期间对象不被析构。断言运行期始终存活、任务数一致。
TEST_CASE("Agent E2E: two tasks with heartbeats survive churn", "[agent][e2e][churn]") {
    if (!std::ifstream("test_data/test_video60.mp4").good() ||
        !std::ifstream("test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx").good())
        SKIP("test data absent");

    const int agent_port = free_port();
    auto buf = fs::temp_directory_path() / "md_e2e_buffer_churn";
    fs::remove_all(buf);

    AgentOptions o;
    o.host = "127.0.0.1"; o.port = agent_port;
    o.edge_code = "edge-churn";
    o.heartbeat_interval_sec = 3600;   // 关闭外部心跳上报，只保留任务级心跳事件
    AgentRuntime rt(o);
    REQUIRE(rt.start());

    auto make_body = [&](int tid, int cam) {
        return json{
            {"task_id", tid},
            {"camera", {{"id", cam}, {"name", "cam" + std::to_string(cam)},
                        {"url", "test_data/test_video60.mp4"}, {"transport", "tcp"}}},
            {"models", json::array({{{"name", "det"}, {"type", "det"}, {"backend", "ort"}, {"device", "cpu"},
                {"path", "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx"},
                {"labels", json::array({"person"})}, {"input_size", json::array({640, 640})},
                {"confidence_threshold", 0.35}}})},
            {"heartbeat_sec", 1},          // 心跳每秒到期（首次 last=epoch 立即到期），制造并发
            {"alarm_interval_sec", 0},
            {"algorithm_type", "INTRUSION"},
            {"preview", {{"enabled", false}}},
            {"events", {{"transport", "http"},
                        {"http", {{"url", "http://127.0.0.1:1/cb"}, {"token", "tk"}}},
                        {"buffer", {{"dir", buf.string()}, {"max_mb", 8}}}}}
        };
    };

    httplib::Client cli("127.0.0.1", agent_port);
    for (int tid : {801, 802}) {
        auto created = cli.Post("/api/v1/tasks", make_body(tid, tid - 800).dump(), "application/json");
        REQUIRE(created);
        REQUIRE(cli.Post(("/api/v1/tasks/" + std::to_string(tid) + "/start").c_str()));
    }
    REQUIRE(rt.manager().task_count() == 2);

    // 并发读：metrics / tasks 列表
    std::atomic<bool> hammer{true};
    std::thread reader([&] {
        httplib::Client mc("127.0.0.1", agent_port);
        mc.set_connection_timeout(2, 0);
        mc.set_read_timeout(5, 0);
        while (hammer.load()) {
            (void)mc.Get("/api/v1/metrics");
            (void)mc.Get("/api/v1/tasks");
        }
    });
    JoinGuard reader_guard{reader, [&]() { hammer = false; }};

    // 反复删除 / 重建 801（运行中），与心跳事件并发
    for (int round = 0; round < 6; ++round) {
        auto del = cli.Delete("/api/v1/tasks/801");
        REQUIRE(del);
        auto created = cli.Post("/api/v1/tasks", make_body(801, 1).dump(), "application/json");
        REQUIRE(created);
        auto started = cli.Post("/api/v1/tasks/801/start");
        REQUIRE(started);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    hammer = false;
    reader.join();

    REQUIRE(rt.manager().task_count() == 2);
    (void)cli.Delete("/api/v1/tasks/801");
    (void)cli.Delete("/api/v1/tasks/802");
    REQUIRE(rt.manager().task_count() == 0);
    rt.stop();
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
    std::mutex mu;                    // 保护下列字段（broker 线程写、主线程读）
    std::string topic;
    std::string payload;
    std::string detection_topic;      // 首个非心跳（真实检测）事件
    std::string detection_payload;

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
                int off = 2 + tlen;
                const int qos = (hdr >> 1) & 0x03;
                int pid = 0;
                if (qos > 0 && rem >= static_cast<uint32_t>(off + 2)) {
                    pid = (static_cast<unsigned char>(body[off]) << 8) | static_cast<unsigned char>(body[off + 1]);
                    off += 2;
                }
                {
                    std::lock_guard<std::mutex> lk(mu);
                    topic.assign(body.data() + 2, tlen);
                    payload.assign(body.data() + off, rem - off);
                    // 就地记录首个真实检测事件：心跳会持续覆盖"最新事件"，
                    // 只轮询最新值可能丢失两次轮询之间的检测事件。
                    if (detection_payload.empty()) {
                        const json parsed = json::parse(payload, nullptr, false);
                        if (parsed.is_object() && !is_heartbeat_event(parsed)) {
                            detection_topic = topic;
                            detection_payload = payload;
                        }
                    }
                }
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

    // 等待首个"真实检测"MQTT 事件：任务心跳（heartbeat=true、detections 为空）同样会
    // 发布到同一 topic，必须跳过心跳，否则会在空 detections 上误判失败。
    std::string topic_seen;
    std::string got_payload;
    const auto mqtt_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < mqtt_deadline) {
        {
            std::lock_guard<std::mutex> lk(broker.mu);
            topic_seen = broker.detection_topic;
            got_payload = broker.detection_payload;
        }
        if (!got_payload.empty()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    REQUIRE_FALSE(got_payload.empty());  // 30s 内未等到真实检测 MQTT 事件（可能只收到心跳）
    REQUIRE(topic_seen == "aistation/default/edge/edge-e2e/camera/8/detect");
    json me = json::parse(got_payload);
    REQUIRE(me["edge_code"] == "edge-e2e");
    REQUIRE(me["camera_id"] == 8);
    REQUIRE(me["task_id"] == 888);
    REQUIRE(me["detections"].is_array());
    REQUIRE(!me["detections"].empty());
    REQUIRE(me.contains("objects"));
    REQUIRE(me["event_id"].is_string());
    REQUIRE(me["event_id"].get<std::string>().size() == 36);
    REQUIRE(me["ts"].is_string());
    REQUIRE(!me["ts"].get<std::string>().empty());
    REQUIRE(me["ts"].get<std::string>().back() == 'Z');
    REQUIRE(me["schema_version"] == 2);

    auto stopped = cli.Post("/api/v1/tasks/888/stop");
    REQUIRE(stopped);
    auto deleted = cli.Delete("/api/v1/tasks/888");
    REQUIRE(deleted);

    rt.stop();
    broker.stop();
    fs::remove_all(buf);
}
#endif  // ENABLE_MQTT
