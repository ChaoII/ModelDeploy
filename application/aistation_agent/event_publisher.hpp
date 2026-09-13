#pragma once
#include <memory>
#include <string>
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
