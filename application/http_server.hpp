#pragma once
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <future>
#include <mutex>

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

private:
    PipelineManager& mgr_;
    std::string host_;
    int port_;
    httplib::Server server_;
    std::thread server_thread_;
    std::atomic<bool> running_{false};

    void register_routes();
    std::string load_web_ui() const;
    // 媒体服务器 HTTP-FLV 端口（前端 deriveHttpFlv 用它；默认 8080）
    int media_server_port_ = 8080;
    // 可选 Bearer 鉴权 key（非空启用）
    std::vector<std::string> api_keys_;

    // 统一错误体：与 SDK ServingServer 一致的 { "error": { "code", "message" } }
    static std::string err_json(const std::string& msg, const std::string& code = "BAD_REQUEST");
    static std::string ok_json(const nlohmann::json& data = {});
    static nlohmann::json task_status_to_json(const TaskStatus& ts);
    static nlohmann::json model_config_to_json(const ModelConfig& m);

    // web_ui.html 一次性加载缓存（启动后不再每次读盘）
    mutable std::mutex web_ui_mtx_;
    mutable std::string web_ui_cache_;
    mutable bool web_ui_loaded_ = false;
};
