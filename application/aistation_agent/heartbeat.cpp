#include "heartbeat.hpp"
#include "capability.hpp"
#include "httplib.h"
#include <algorithm>
#include <chrono>
#include <iostream>

#ifndef MD_VERSION
#define MD_VERSION "0.0.0"
#endif

Heartbeat::Heartbeat(std::string cloud_url, std::string edge_code, std::string token,
                     std::function<nlohmann::json()> metrics_provider,
                     int interval_sec, int max_channels)
    : cloud_url_(std::move(cloud_url)), edge_code_(std::move(edge_code)), token_(std::move(token)),
      metrics_provider_(std::move(metrics_provider)), interval_sec_(interval_sec),
      max_channels_(max_channels), capabilities_(detect_capabilities(max_channels)) {}

Heartbeat::~Heartbeat() { stop(); }

nlohmann::json Heartbeat::make_payload() const {
    nlohmann::json metrics = nlohmann::json::object();
    if (metrics_provider_) {
        try {
            metrics = metrics_provider_();
        } catch (...) {
            metrics = nlohmann::json::object();
        }
    }
    return nlohmann::json{
        {"edge_code", edge_code_},
        {"token", token_},
        {"capabilities", capabilities_},
        {"metrics", metrics},
        {"version", MD_VERSION},
    };
}

bool Heartbeat::send_once() {
    if (cloud_url_.empty()) return false;
    const auto scheme = cloud_url_.find("://");
    if (scheme == std::string::npos) return false;
    const std::string rest = cloud_url_.substr(scheme + 3);
    const auto slash = rest.find('/');
    const std::string base = slash == std::string::npos ? cloud_url_ : cloud_url_.substr(0, scheme + 3 + slash);
    const std::string path = "/api/v1/video/edge/heartbeat";
    httplib::Client cli(base);
    cli.set_connection_timeout(3, 0);
    cli.set_read_timeout(5, 0);
    auto res = cli.Post(path.c_str(), make_payload().dump(), "application/json");
    if (!res || res->status < 200 || res->status >= 300) {
        std::cerr << "[Heartbeat] post failed" << std::endl;
        return false;
    }
    return true;
}

void Heartbeat::start() {
    if (cloud_url_.empty() || running_.exchange(true)) return;
    thread_ = std::thread([this]() {
        const int interval_ms = std::max(interval_sec_, 1) * 1000;
        int backoff_ms = 1000;
        while (running_.load()) {
            int wait_ms;
            bool sent = false;
            try {
                sent = send_once();
            } catch (...) {
                sent = false;
            }
            if (sent) {
                backoff_ms = 1000;
                wait_ms = interval_ms;
            } else {
                wait_ms = backoff_ms;
                backoff_ms = std::min(backoff_ms * 2, 60000);
            }
            for (int waited = 0; waited < wait_ms && running_.load(); waited += 100)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
}

void Heartbeat::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
}
