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

/// 模型名能否作为单段 URL 路径（path segment）表示。
/// 拒绝空名、'/'（路径分隔符，无法在单段中表示 —— 百分号解码后会把段拆开导致路由不匹配）
/// 与控制字符（<0x20 / 0x7F，无法出现在 HTTP 请求行）。
/// 空格与 '?' '#' '%' 等允许：客户端按 RFC 3986 百分号编码即可；服务端 httplib 解析请求行
/// 时已对 req.path 解码一次（path_params 即明文），处理器不应二次解码（会破坏字面 %XX）。
/// 失败时写 err（中文）并返回 false。
bool is_valid_model_name_segment(const std::string& name, std::string* err = nullptr);

/// AIStation 控制面 REST（httplib；错误体与 ServingServer 一致）
class AgentServer {
public:
    AgentServer(PipelineManager& mgr, const ConfigAdapter& adapter,
                const std::string& host = "0.0.0.0", int port = 19090);
    ~AgentServer();

    void set_api_key(std::string secret) { api_key_ = std::move(secret); }
    /// 显式开发模式：未配置 api_key 时仍放行控制面（默认关闭 = fail-closed）。
    /// 只允许本地开发使用；生产必须配置 --api-key，否则 /api/v1/* 一律 503。
    void set_allow_anonymous(bool v) { allow_anonymous_ = v; }
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
    // 未配置密钥时是否放行（默认 false = fail-closed；仅 --allow-anonymous 显式开启）
    bool allow_anonymous_ = false;
    // pre_routing 由 httplib 多个工作线程并发调用：告警只打一次，用原子避免竞态
    std::atomic<bool> anon_warned_{false};
    AgentHooks hooks_;
    std::function<nlohmann::json()> metrics_provider_;
    // 热更新世代号：每次成功换入引擎自增（std::atomic，控制面并发调用安全）。
    // 进程内**单调递增**：只用 fetch_add 单点递增，并发调用也不会回退/重复。
    // 不持久化 —— 进程重启后归 0 重新计数（spec 允许，不要求跨进程单调）。
    std::atomic<uint64_t> generation_{0};

    void register_routes();
    static std::string err_json(const std::string& msg, const std::string& code = "BAD_REQUEST");
    static std::string ok_json(const nlohmann::json& data = {});
    /// 热更新端点专用失败体：{"ok":false,"error":<中文>,"generation":<当前世代>}
    static std::string fail_json(const std::string& msg, uint64_t generation);
    static nlohmann::json status_to_json(const TaskStatus& ts);
    static std::string get_id(const httplib::Request& req, const char* key = "id");
};
