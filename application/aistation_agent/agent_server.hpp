#pragma once
#include <string>
#include <functional>
#include <atomic>
#include <cstdint>
#include <thread>
#include "httplib.h"
#include "nlohmann/json.hpp"
#include "pipeline_manager.hpp"
#include "config_adapter.hpp"

struct AgentHooks {
    std::function<void(const std::string& task_id, const AdaptedTask&)> on_created;
    std::function<void(const std::string& task_id, const AdaptedTask&)> on_updated;
    std::function<void(const std::string& task_id)> on_removed;
};

/// AIStation 控制面 REST（httplib；错误体与 ServingServer 一致）
class AgentServer {
public:
    AgentServer(PipelineManager& mgr, const ConfigAdapter& adapter,
                const std::string& host = "0.0.0.0", int port = 19090);
    ~AgentServer();

    void set_api_key(std::string secret) { api_key_ = std::move(secret); }
    void set_hooks(AgentHooks hooks) { hooks_ = std::move(hooks); }
    void set_metrics_provider(std::function<nlohmann::json()> fn) { metrics_provider_ = std::move(fn); }

    bool start();
    void stop();
    bool is_running() const { return running_.load(); }

    /// 模型热更新世代号：每次成功热交换 +1；失败/未就绪不递增（供测试与观测）
    uint64_t generation() const { return generation_.load(); }

private:
    PipelineManager& mgr_;
    const ConfigAdapter& adapter_;
    std::string host_;
    int port_;
    httplib::Server server_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    bool routes_registered_{false};
    std::string api_key_;
    AgentHooks hooks_;
    std::function<nlohmann::json()> metrics_provider_;
    // 热更新世代号：每次成功换入引擎自增（std::atomic，控制面并发调用安全）
    std::atomic<uint64_t> generation_{0};

    void register_routes();
    static std::string err_json(const std::string& msg, const std::string& code = "BAD_REQUEST");
    static std::string ok_json(const nlohmann::json& data = {});
    /// 热更新端点专用失败体：{"ok":false,"error":<中文>,"generation":<当前世代>}
    static std::string fail_json(const std::string& msg, uint64_t generation);
    static nlohmann::json status_to_json(const TaskStatus& ts);
    static std::string get_id(const httplib::Request& req, const char* key = "id");
};
