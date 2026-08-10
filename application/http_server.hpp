#pragma once
#include <string>
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

    static std::string err_json(const std::string& msg);
    static std::string ok_json(const nlohmann::json& data = {});
    static nlohmann::json task_status_to_json(const TaskStatus& ts);
    static nlohmann::json model_config_to_json(const ModelConfig& m);

    // web_ui.html 一次性加载缓存（启动后不再每次读盘）
    mutable std::mutex web_ui_mtx_;
    mutable std::string web_ui_cache_;
    mutable bool web_ui_loaded_ = false;
};
