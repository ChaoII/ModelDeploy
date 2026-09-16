#pragma once
#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include "nlohmann/json.hpp"

class Heartbeat {
public:
    Heartbeat(std::string cloud_url, std::string edge_code, std::string token,
              std::function<nlohmann::json()> metrics_provider,
              int interval_sec = 30, int max_channels = 8);
    ~Heartbeat();
    void start();
    void stop();
    bool send_once();
    nlohmann::json make_payload() const;

    /// 连续心跳失败次数（成功即清零）——用于判定"边缘是否已与云端失联"
    uint64_t consecutive_failures() const { return consecutive_failures_.load(); }
    /// 累计心跳失败次数（单调递增，供 metrics 观测）
    uint64_t total_failures() const { return total_failures_.load(); }
    /// 最近一次失败原因：HTTP 状态码；-1 表示连接失败/无响应；0 表示尚无失败
    int last_failure_status() const { return last_failure_status_.load(); }

    /// 连续失败达到该阈值时在日志中升级为失联告警
    static constexpr uint64_t kDisconnectWarnThreshold = 3;

private:
    std::string cloud_url_, edge_code_, token_;
    std::function<nlohmann::json()> metrics_provider_;
    int interval_sec_, max_channels_;
    nlohmann::json capabilities_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    // 心跳失败可观测性（合同审计 HIGH 3.4：旧实现只打一行 stderr 且无计数，
    // 运维无法发现"边缘已失联/密钥不一致被 403"）。
    std::atomic<uint64_t> consecutive_failures_{0};
    std::atomic<uint64_t> total_failures_{0};
    std::atomic<int> last_failure_status_{0};
};
