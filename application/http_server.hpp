#pragma once
#include <string>
#include <vector>
#include <map>
#include <atomic>
#include <thread>
#include <future>
#include <mutex>
#include <chrono>

#include "httplib.h"
#include "pipeline_manager.hpp"

class HttpServer {
public:
    HttpServer(PipelineManager& mgr,
               const std::string& host = "0.0.0.0",
               int port = 8080);
    ~HttpServer();

    bool start();
    void stop();
    bool is_running() const { return running_.load(); }
    // 设置媒体服务器 HTTP-FLV 端口（前端 deriveHttpFlv 用它；默认 8080）
    void set_media_server_port(int port) { media_server_port_ = port; }
    // 可选 Bearer 鉴权：非空时保护 /api/v1/*（静态页与 /health 放行）
    void set_api_keys(std::vector<std::string> keys) { api_keys_ = std::move(keys); }
    // 可选全局限流：qps>0 时按恒定速率放行，超限返回 429
    void set_rate_limit(double qps) { rate_limit_qps_ = qps; }
    // 数据目录（/api/v1/save 落盘用；空则不落盘）
    void set_data_dir(std::string d) { data_dir_ = std::move(d); }

private:
    PipelineManager& mgr_;
    std::string host_;
    int port_;
    httplib::Server server_;
    std::thread server_thread_;
    std::atomic<bool> running_{false};

    void register_routes();
    std::string load_web_ui() const;
    // 从 application/third_party 解析静态资源（如 flv.min.js），带缓存
    std::string load_asset(const std::string& name) const;
    // 媒体服务器 HTTP-FLV 端口（前端 deriveHttpFlv 用它；默认 8080）
    int media_server_port_ = 8080;
    // 可选 Bearer 鉴权 key（非空启用）
    std::vector<std::string> api_keys_;
    // 可选全局限流（令牌桶，容量=1）
    double rate_limit_qps_ = 0.0;
    std::mutex rate_mtx_;
    double rate_tokens_ = 1.0;
    std::chrono::steady_clock::time_point rate_last_ = std::chrono::steady_clock::now();
    bool rate_acquire();
    // 数据目录（/api/v1/save 落盘用）
    std::string data_dir_;

    // 统一错误体：与 SDK ServingServer 一致的 { "error": { "code", "message" } }
    static std::string err_json(const std::string& msg, const std::string& code = "BAD_REQUEST");
    static std::string ok_json(const nlohmann::json& data = {});
    static nlohmann::json task_status_to_json(const TaskStatus& ts);
    static nlohmann::json model_config_to_json(const ModelConfig& m);

    // web_ui.html 一次性加载缓存（启动后不再每次读盘）
    mutable std::mutex web_ui_mtx_;
    mutable std::string web_ui_cache_;
    mutable bool web_ui_loaded_ = false;
    // 静态资源缓存
    mutable std::mutex asset_mtx_;
    mutable std::map<std::string, std::string> asset_cache_;
};
