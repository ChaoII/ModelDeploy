#include "event_publisher.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

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

// ── DurableQueue：边缘持久化缓存队列 ──
namespace fs = std::filesystem;

namespace {
constexpr size_t kExtLen = 5;   // ".json"

std::string pad_seq(uint64_t s) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%020llu", static_cast<unsigned long long>(s));
    return buf;
}

/// 文件名 `<20位seq>-<event_id>.json` -> event_id（去掉定宽序号前缀与 .json 后缀）
std::string event_id_of(const std::string& filename) {
    const auto dash = filename.find('-');
    if (dash == std::string::npos) return std::string();
    std::string id = filename.substr(dash + 1);
    if (id.size() > kExtLen && id.compare(id.size() - kExtLen, kExtLen, ".json") == 0)
        id.resize(id.size() - kExtLen);
    return id;
}
}  // namespace

DurableQueue::DurableQueue(std::string dir, int max_mb, EventPublisher* transport, int retry_ms)
    : DurableQueue(std::move(dir), static_cast<size_t>(max_mb) * 1024ull * 1024ull, transport, retry_ms) {}

DurableQueue::DurableQueue(std::string dir, size_t max_bytes, EventPublisher* transport, int retry_ms)
    : dir_(std::move(dir)), max_bytes_(max_bytes), transport_(transport), retry_ms_(retry_ms) {
    fs::create_directories(dir_);
    load_existing();
}

DurableQueue::~DurableQueue() { stop(); }

void DurableQueue::load_existing() {
    std::lock_guard<std::mutex> lk(mtx_);
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir_, ec)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() <= kExtLen) continue;
        if (name.compare(name.size() - kExtLen, kExtLen, ".json") != 0) continue;
        files_.push_back(name);
        event_ids_.insert(event_id_of(name));
        bytes_ += static_cast<size_t>(entry.file_size(ec));
        try {
            const uint64_t s = std::stoull(name.substr(0, name.find('-')));
            if (s >= seq_) seq_ = s + 1;
        } catch (...) {}
    }
    std::sort(files_.begin(), files_.end());
}

bool DurableQueue::enforce_limit() {
    bool changed = false;
    while (bytes_ > max_bytes_ && !files_.empty()) {
        const std::string victim = files_.front();
        files_.pop_front();
        event_ids_.erase(event_id_of(victim));
        std::error_code ec;
        const auto sz = fs::file_size(fs::path(dir_) / victim, ec);
        if (!ec) bytes_ = (sz < bytes_) ? bytes_ - static_cast<size_t>(sz) : 0;
        fs::remove(fs::path(dir_) / victim, ec);
        ++dropped_;
        changed = true;
    }
    return changed;
}

void DurableQueue::enqueue(const DetectionEvent& e) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!e.event_id.empty() && event_ids_.count(e.event_id)) return;   // 去重
        const std::string tmp = (fs::path(dir_) / ("tmp-" + pad_seq(seq_))).string();
        const std::string final_name = pad_seq(seq_) + "-" + e.event_id + ".json";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f.is_open()) return;
            f << e.to_json().dump();
            f.flush();
        }
        std::error_code ec;
        fs::rename(tmp, fs::path(dir_) / final_name, ec);
        if (ec) { fs::remove(tmp, ec); return; }
        files_.push_back(final_name);
        event_ids_.insert(event_id_of(final_name));
        const auto sz = fs::file_size(fs::path(dir_) / final_name, ec);
        if (!ec) bytes_ += static_cast<size_t>(sz);
        ++seq_;
        enforce_limit();
    }
    cv_.notify_one();
}

size_t DurableQueue::pending() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return files_.size();
}

void DurableQueue::start() {
    if (running_.exchange(true)) return;
    worker_ = std::thread([this]() { worker_loop(); });
}

void DurableQueue::stop() {
    if (!running_.exchange(false)) return;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void DurableQueue::worker_loop() {
    while (running_.load()) {
        std::string name;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this]() { return !running_.load() || !files_.empty(); });
            if (!running_.load()) break;
            name = files_.front();
        }
        bool ok = false;
        try {
            std::ifstream f(fs::path(dir_) / name, std::ios::binary);
            std::stringstream ss; ss << f.rdbuf();
            DetectionEvent e = DetectionEvent::from_json(nlohmann::json::parse(ss.str()));
            ok = transport_ && transport_->publish(e);
        } catch (...) { ok = false; }

        if (ok) {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!files_.empty() && files_.front() == name) {
                files_.pop_front();
                event_ids_.erase(event_id_of(name));
                std::error_code ec;
                const auto sz = fs::file_size(fs::path(dir_) / name, ec);
                if (!ec) bytes_ = (sz < bytes_) ? bytes_ - static_cast<size_t>(sz) : 0;
                fs::remove(fs::path(dir_) / name, ec);
                enforce_limit();
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_ms_));   // 保持队首按序重试
        }
    }
}
