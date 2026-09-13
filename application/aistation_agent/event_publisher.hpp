#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
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
    uint64_t dropped() const { return dropped_.load(); }
    void start();
    void stop();

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
};
