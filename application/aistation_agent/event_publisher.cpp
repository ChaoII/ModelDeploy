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
