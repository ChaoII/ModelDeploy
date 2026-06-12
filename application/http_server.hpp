#pragma once
#include <string>
#include <atomic>
#include <thread>
#include <future>

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

    static std::string err_json(const std::string& msg);
    static std::string ok_json(const std::string& data = "{}");
    static std::string task_list_json(const std::vector<TaskStatus>& tasks);
    static std::string task_config_json(const TaskConfig& cfg);
};
