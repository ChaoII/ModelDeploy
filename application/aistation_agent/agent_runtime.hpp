#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
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

    AgentOptions opts_;
    PipelineManager mgr_;
    ModelFetcher fetcher_;
    ConfigAdapter adapter_;
    EventBus bus_;
    std::unique_ptr<AgentServer> server_;
    std::unique_ptr<Heartbeat> heartbeat_;
    std::mutex mtx_;
    std::unordered_map<std::string, std::unique_ptr<EventPublisher>> publishers_;
    std::unordered_map<std::string, std::unique_ptr<DurableQueue>> queues_;
    std::unordered_map<int64_t, DurableQueue*> queue_by_task_;
};
