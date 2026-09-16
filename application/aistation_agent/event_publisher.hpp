#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include "event_bus.hpp"
#include "httplib.h"

struct MqttConfig {
    std::string broker;      // tcp://host:1883
    std::string topic;       // 已解析主题
    std::string client_id;
    std::string username;
    std::string password;
    int qos = 1;
};

class EventPublisher {
public:
    virtual ~EventPublisher() = default;
    virtual bool publish(const DetectionEvent& e) = 0;
    virtual std::string name() const = 0;
};

class HttpPublisher : public EventPublisher {
public:
    HttpPublisher(std::string url, std::string token);
    bool publish(const DetectionEvent& e) override;
    std::string name() const override { return "http"; }

private:
    std::string url_, base_, path_, token_;
    std::unique_ptr<httplib::Client> client_;
};

class MqttPublisher : public EventPublisher {
public:
    explicit MqttPublisher(MqttConfig cfg);
    ~MqttPublisher() override;
    bool publish(const DetectionEvent& e) override;
    std::string name() const override { return "mqtt"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// 边缘持久化缓存队列：离线落盘、重启补发、超限丢最旧、按 event_id 去重
class DurableQueue {
public:
    DurableQueue(std::string dir, int max_mb, EventPublisher* transport, int retry_ms = 1000);
    DurableQueue(std::string dir, size_t max_bytes, EventPublisher* transport, int retry_ms);
    ~DurableQueue();

    void enqueue(const DetectionEvent& e);    // 非阻塞；失败/离线时落盘
    size_t pending() const;
    /// 队首传输失败（publish=false）的累计次数（可观测，避免静默累积）
    uint64_t send_failed() const { return send_failed_.load(); }
    /// 磁盘写失败（落盘失败 → 事件必然丢失）的累计次数
    uint64_t write_failed() const { return write_failed_.load(); }
    /// 超限丢弃（丢最旧）的累计条数
    uint64_t dropped() const { return dropped_.load(); }
    /// 当前落盘占用字节数（观测队列水位）
    size_t pending_bytes() const;
    void start();
    void stop();

    /// 发布前采集内联快照：返回 false 或空串则不附加
    using SnapshotProvider = std::function<bool(int64_t task_id, std::string* b64, int* w, int* h)>;
    void set_snapshot_provider(SnapshotProvider p);

private:
    void load_existing();
    void worker_loop();
    bool enforce_limit();

    std::string dir_;
    size_t max_bytes_;
    EventPublisher* transport_;
    int retry_ms_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<std::string> files_;
    std::unordered_set<std::string> event_ids_;
    size_t bytes_ = 0;
    uint64_t seq_ = 0;
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> dropped_{0};
    // 可观测性计数（合同审计 HIGH：旧实现磁盘写失败直接丢事件且不计数、传输永久失败
    // 静默累积，业务侧无法区分"未发送过期丢弃"与"已发送重试"）。三者都对外暴露。
    std::atomic<uint64_t> send_failed_{0};    // 传输失败次数（publish 返回 false）
    std::atomic<uint64_t> write_failed_{0};   // 磁盘写失败次数（事件因此必然丢失）
    SnapshotProvider snapshot_provider_;
};
