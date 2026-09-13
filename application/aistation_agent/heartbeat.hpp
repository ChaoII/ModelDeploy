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

private:
    std::string cloud_url_, edge_code_, token_;
    std::function<nlohmann::json()> metrics_provider_;
    int interval_sec_, max_channels_;
    nlohmann::json capabilities_;
    std::thread thread_;
    std::atomic<bool> running_{false};
};
