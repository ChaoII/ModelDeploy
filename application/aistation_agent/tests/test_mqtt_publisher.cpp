#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "event_publisher.hpp"

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
}  // namespace

namespace {
struct MockBroker {
    int port;
    std::atomic<bool> running{true};
    std::thread th;
    std::string topic;
    std::string payload;
    std::atomic<int> publishes{0};
    std::atomic<int> last_qos{-1};

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
        set_recv_timeout(listener, 300);
        sock_t c = INVALID_SOCKET;
        while (running.load()) {
            c = accept(listener, nullptr, nullptr);
            if (c != INVALID_SOCKET) break;
            if (!was_timeout()) { CLOSE_SOCK(listener); return; }
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
                last_qos = qos;
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
    void stop() { running = false; if (th.joinable()) th.join(); }
};
}  // namespace

TEST_CASE("MqttPublisher publishes QoS1 event to broker", "[agent][publisher][mqtt]") {
#ifdef ENABLE_MQTT
#if defined(_WIN32)
    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    static std::mt19937 rng(std::random_device{}());
    MockBroker broker; broker.port = 22000 + static_cast<int>(rng() % 5000);
    broker.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    MqttConfig cfg;
    cfg.broker = "tcp://127.0.0.1:" + std::to_string(broker.port);
    cfg.topic = "aistation/default/edge/edge-01/camera/7/detect";
    cfg.client_id = "aistation-agent-test";
    MqttPublisher pub(cfg);

    DetectionEvent e;
    e.event_id = "22222222-2222-4222-8222-222222222222";
    e.edge_code = "edge-01"; e.camera_id = 7; e.task_id = 123;
    e.algorithm_type = "INTRUSION"; e.ts = "2026-09-12T08:00:00.123Z"; e.latency_ms = 5.0;
    EventDetection d; d.label = "person"; d.label_id = 0; d.confidence = 0.9f;
    e.detections.push_back(d);

    REQUIRE(pub.publish(e));
    for (int i = 0; i < 50 && broker.publishes.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    REQUIRE(broker.publishes.load() == 1);
    REQUIRE(broker.last_qos.load() == 1);
    REQUIRE(broker.topic == cfg.topic);
    REQUIRE(broker.payload.find("\"event_id\":\"22222222-2222-4222-8222-222222222222\"") != std::string::npos);
    broker.stop();
#else
    SUCCEED("ENABLE_MQTT disabled: mqtt publisher stub not exercised");
#endif
}
