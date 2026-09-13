#include "event_publisher.hpp"
#include <iostream>

// ── URL 解析：full url -> (scheme://host:port, /path) ──
static bool split_http_url(const std::string& url, std::string* base, std::string* path) {
    const auto scheme = url.find("://");
    if (scheme == std::string::npos) return false;
    const std::string rest = url.substr(scheme + 3);
    const auto slash = rest.find('/');
    if (slash == std::string::npos) {
        *base = url;
        *path = "/";
    } else {
        *base = url.substr(0, scheme + 3 + slash);
        *path = rest.substr(slash);
    }
    return true;
}

HttpPublisher::HttpPublisher(std::string url, std::string token)
    : url_(std::move(url)), token_(std::move(token)) {
    if (split_http_url(url_, &base_, &path_)) {
        client_ = std::make_unique<httplib::Client>(base_);
        client_->set_connection_timeout(3, 0);
        client_->set_read_timeout(10, 0);
        client_->set_write_timeout(10, 0);
    }
}

bool HttpPublisher::publish(const DetectionEvent& e) {
    if (!client_) return false;
    httplib::Headers headers{{"Authorization", "Bearer " + token_}};
    auto res = client_->Post(path_.c_str(), headers, e.to_json().dump(), "application/json");
    return res && res->status >= 200 && res->status < 300;
}

// ── MQTT：Task 6 实现；未启用时为空实现 ──
#ifndef ENABLE_MQTT
struct MqttPublisher::Impl {};
MqttPublisher::MqttPublisher(MqttConfig) : impl_(std::make_unique<Impl>()) {
    std::cerr << "[MqttPublisher] built without ENABLE_MQTT (disabled)" << std::endl;
}
MqttPublisher::~MqttPublisher() = default;
bool MqttPublisher::publish(const DetectionEvent&) { return false; }
#endif

#ifdef ENABLE_MQTT
#include "MQTTClient.h"
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

struct MqttPublisher::Impl {
    MqttConfig cfg;
    MQTTClient client = nullptr;
    std::mutex mtx;
    std::atomic<bool> running{true};
    std::thread keepalive;

    bool ensure_connected() {
        if (client && MQTTClient_isConnected(client)) return true;
        if (!client) {
            if (MQTTClient_create(&client, cfg.broker.c_str(), cfg.client_id.c_str(),
                                  MQTTCLIENT_PERSISTENCE_NONE, nullptr) != MQTTCLIENT_SUCCESS) {
                client = nullptr;
                return false;
            }
        }
        MQTTClient_connectOptions opts = MQTTClient_connectOptions_initializer;
        opts.keepAliveInterval = 30;
        opts.cleansession = 1;
        opts.connectTimeout = 5;
        opts.retryInterval = 0;
        if (!cfg.username.empty()) opts.username = cfg.username.c_str();
        if (!cfg.password.empty()) opts.password = cfg.password.c_str();
        return MQTTClient_connect(client, &opts) == MQTTCLIENT_SUCCESS;
    }

    void keepalive_loop() {
        while (running.load()) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                ensure_connected();
                if (client) MQTTClient_yield();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    }
};

MqttPublisher::MqttPublisher(MqttConfig cfg) : impl_(std::make_unique<Impl>()) {
    impl_->cfg = std::move(cfg);
    if (impl_->cfg.client_id.empty()) impl_->cfg.client_id = "aistation-agent";
    if (!impl_->ensure_connected())
        std::cerr << "[MqttPublisher] initial connect failed: " << impl_->cfg.broker << std::endl;
    impl_->keepalive = std::thread([this]() { impl_->keepalive_loop(); });
}

MqttPublisher::~MqttPublisher() {
    if (!impl_) return;
    impl_->running = false;
    if (impl_->keepalive.joinable()) impl_->keepalive.join();
    if (impl_->client) {
        MQTTClient_disconnect(impl_->client, 1000);
        MQTTClient_destroy(&impl_->client);
    }
}

bool MqttPublisher::publish(const DetectionEvent& e) {
    if (!impl_) return false;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->ensure_connected()) return false;
    const std::string payload = e.to_json().dump();
    MQTTClient_message msg = MQTTClient_message_initializer;
    msg.payload = const_cast<char*>(payload.data());
    msg.payloadlen = static_cast<int>(payload.size());
    msg.qos = impl_->cfg.qos;
    msg.retained = 0;
    MQTTClient_deliveryToken token;
    if (MQTTClient_publishMessage(impl_->client, impl_->cfg.topic.c_str(), &msg, &token) != MQTTCLIENT_SUCCESS)
        return false;
    return MQTTClient_waitForCompletion(impl_->client, token, 5000L) == MQTTCLIENT_SUCCESS;
}
#endif
