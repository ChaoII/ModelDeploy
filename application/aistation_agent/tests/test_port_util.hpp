#pragma once
// 测试用本地端口分配器（仅测试目标使用）。
//
// 背景 / 根因：httplib 的 Server 默认在 socket 上设置 SO_REUSEADDR
// （application/third_party/httplib.h 的 default_socket_options）。在 Windows 上，
// SO_REUSEADDR 允许两个 socket 绑定同一 addr:port，且**先绑定者**接收新连接
// （本仓库已实测：先 bind 的监听器持续拿到全部连接）。
//
// 因此测试里若两处端口"随机"取值相同，"后启动"的服务会被"先启动"的服务遮蔽：
// test_e2e_agent.cpp 先起事件接收器 recv（只注册 POST /cb）再起 Agent 控制面，
// 一旦 recv_port == agent_port，Agent 的 bind 仍会成功，但 /api/v1/tasks 会被
// recv 接管并返回 404/空响应体 → `json::parse("")` 抛异常 → 表现为与负载无关、
// 无法按需复现的"偶发 POST /tasks 失败"。已用"强制同端口"稳定复现该现象。
//
// 本分配器保证：
//   1) 进程内唯一（set + mutex，inline 函数的 static 局部量在整个进程内共享，
//      跨编译单元也唯一）——消除最可能的"自碰撞"；
//   2) 候选端口当前确实可绑定（用临时 socket 探测；探测不设 SO_REUSEADDR，
//      从而能发现已存在的监听器，含外部进程/残留服务）。
//
// 说明：探测后立即释放端口，理论上仍存在与"其它进程同时抢同一端口"的极小竞态；
// 这属于测试基础设施固有的 TOCTOU，无法在不持有 socket 到服务启动的前提下消除。
// 本分配器消除的是**进程内自碰撞**（即已复现的真实根因），并把外部碰撞概率降到极低。
#include <mutex>
#include <random>
#include <set>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace aistation_test {

/// 端口候选范围（[base, base+span)）。避开 Windows 动态端口范围（49152+），
/// 且所有 agent 测试统一使用同一范围，避免不同 TU 的区间互相重叠。
constexpr int kPortBase = 26000;
constexpr int kPortSpan = 3000;

/// 探测 `port` 是否可绑定（不设 SO_REUSEADDR，以便发现已存在的监听器）。
inline bool port_bindable(int port) {
#if defined(_WIN32)
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return false;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons(static_cast<unsigned short>(port));
    const bool ok = bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0;
    closesocket(s);
    return ok;
#else
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return false;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons(static_cast<unsigned short>(port));
    const bool ok = bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0;
    close(s);
    return ok;
#endif
}

/// 分配一个"进程内唯一且当前可绑定"的本地测试端口。
inline int test_free_port() {
    static std::mutex mu;
    static std::set<int> used;
    static std::mt19937 rng(std::random_device{}());
    std::lock_guard<std::mutex> lk(mu);
    for (int attempt = 0; attempt < 400; ++attempt) {
        const int port = kPortBase + static_cast<int>(rng() % kPortSpan);
        if (used.count(port)) continue;          // 进程内已分配过 → 跳过
        if (!port_bindable(port)) continue;      // 正被占用（外部/残留）→ 跳过
        used.insert(port);
        return port;
    }
    // 极端兜底：范围内候选耗尽时退化为随机值（调用方随后 bind 失败会显式报错），
    // 避免测试死循环。
    return kPortBase + static_cast<int>(rng() % kPortSpan);
}

}  // namespace aistation_test
