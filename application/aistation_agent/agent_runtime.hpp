#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include "agent_server.hpp"
#include "config_adapter.hpp"
#include "event_bus.hpp"
#include "event_publisher.hpp"
#include "heartbeat.hpp"
#include "model_fetcher.hpp"
#include "pipeline_manager.hpp"

struct AgentOptions {
    std::string host = "0.0.0.0";
    int port = 19090;
    std::string data_dir;
    std::string api_key;
    // 显式开发模式：允许控制面在未配置 --api-key 时启动（默认 false = fail-closed）。
    bool allow_anonymous = false;
    std::string cloud_url;
    std::string edge_code = "edge-01";
    std::string secret;
    std::string model_cache_dir;
    std::string s3_endpoint;
    int max_channels = 8;
    int heartbeat_interval_sec = 30;
};

class AgentRuntime {
public:
    explicit AgentRuntime(AgentOptions opts);
    ~AgentRuntime();
    bool start();
    void stop();
    PipelineManager& manager() { return mgr_; }
    EventBus& event_bus() { return bus_; }
    int port() const { return opts_.port; }

private:
    void on_created(const std::string& task_id, const AdaptedTask& t);
    void on_updated(const std::string& task_id, const AdaptedTask& t);
    void on_removed(const std::string& task_id);
    nlohmann::json metrics();

    bool capture_snapshot_b64(const std::string& task_id, const SnapshotConfig& cfg,
                              std::string* b64, int* w, int* h);
    void schedule_loop();
    void heartbeat_loop();

    struct SchedEntry {
        aistation::Schedule schedule;
        // 该任务是否由本调度控制器停止（仅此类任务会在窗口重新开启后自动恢复）；
        // 用于区分手动停止，避免与用户操作相互打架。
        bool stopped_by_us = false;
    };
    std::unordered_map<std::string, SchedEntry> schedules_;
    std::thread schedule_thread_;
    std::atomic<bool> schedule_running_{false};

    struct HeartbeatEntry {
        int interval_sec = 5;
        std::chrono::steady_clock::time_point last{};   // 默认=未发过，立即到期
    };
    std::unordered_map<std::string, HeartbeatEntry> heartbeats_;
    std::thread heartbeat_thread_;
    std::atomic<bool> heartbeat_running_{false};

    AgentOptions opts_;
    PipelineManager mgr_;
    ModelFetcher fetcher_;
    ConfigAdapter adapter_;
    EventBus bus_;
    std::unique_ptr<AgentServer> server_;
    std::unique_ptr<Heartbeat> heartbeat_;
    std::mutex mtx_;
    std::unordered_map<std::string, std::unique_ptr<EventPublisher>> publishers_;
    // 队列用 shared_ptr 持有：EventBus sink 可能在 mtx_ 之外调用 enqueue，若此时
    // on_removed 把队列摘走并 stop()/析构，裸指针会写已释放对象（UAF）。shared_ptr
    // 让在途 sink 持有一份强引用，保证 enqueue 期间队列存活。
    std::unordered_map<std::string, std::shared_ptr<DurableQueue>> queues_;
    std::unordered_map<int64_t, std::shared_ptr<DurableQueue>> queue_by_task_;
};
