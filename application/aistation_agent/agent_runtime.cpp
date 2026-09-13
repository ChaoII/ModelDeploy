#include "agent_runtime.hpp"

#include <iostream>
#include <iterator>
#include <utility>

AgentRuntime::AgentRuntime(AgentOptions opts)
    : opts_(std::move(opts)),
      fetcher_(opts_.model_cache_dir.empty() ? "data/model_cache" : opts_.model_cache_dir,
               opts_.s3_endpoint),
      adapter_(&fetcher_) {}

AgentRuntime::~AgentRuntime() { stop(); }

bool AgentRuntime::start() {
    bus_.set_sink([this](const DetectionEvent& e) {
        // 仅在锁内查找并拷贝队列指针；enqueue 含磁盘 I/O，必须在锁外执行，
        // 否则检测线程会长时间阻塞在 mtx_ 上（与 PipelineManager 形成锁序风险）。
        DurableQueue* q = nullptr;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            auto it = queue_by_task_.find(e.task_id);
            if (it != queue_by_task_.end()) q = it->second;
        }
        if (q) q->enqueue(e);
    });

    AgentHooks hooks;
    hooks.on_created = [this](const std::string& id, const AdaptedTask& t) { on_created(id, t); };
    hooks.on_updated = [this](const std::string& id, const AdaptedTask& t) { on_updated(id, t); };
    hooks.on_removed = [this](const std::string& id) { on_removed(id); };

    server_ = std::make_unique<AgentServer>(mgr_, adapter_, opts_.host, opts_.port);
    server_->set_api_key(opts_.api_key);
    server_->set_hooks(std::move(hooks));
    server_->set_metrics_provider([this]() { return metrics(); });
    if (!server_->start()) return false;

    heartbeat_ = std::make_unique<Heartbeat>(
        opts_.cloud_url, opts_.edge_code,
        opts_.secret.empty() ? opts_.api_key : opts_.secret,
        [this]() { return metrics(); }, opts_.heartbeat_interval_sec, opts_.max_channels);
    heartbeat_->start();
    return true;
}

void AgentRuntime::stop() {
    // 停机顺序（关键：确保释放队列/发布器前已无任何读者）：
    // 1) 先停 heartbeat_：其 metrics 回调会读 queues_；
    // 2) 再停 server_：可能触发 hooks（on_removed/on_updated），且其 handler 也会读 queues_；
    // 3) 然后 mgr_.stop_all()：停止所有 pipeline 并 join 检测线程 —— 检测线程经
    //    EventBus sink 在 mtx_ 外调用 q->enqueue(e)，join 后保证不再有 sink 回调在途；
    // 4) 最后销毁 queues_/publishers_：此时已无读者，不会 UAF。
    if (heartbeat_) heartbeat_->stop();
    if (server_) server_->stop();
    mgr_.stop_all();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (auto& [id, q] : queues_)
            if (q) q->stop();
        queues_.clear();
        publishers_.clear();
        queue_by_task_.clear();
    }
}

nlohmann::json AgentRuntime::metrics() {
    // 锁序：先在 mtx_ 之外访问 PipelineManager（其 mtx_ 与检测线程 join 相关），
    // 再单独持 mtx_ 汇总本类队列；避免 mtx_ -> PipelineManager::mtx_ 与检测
    // 线程反向获取 mtx_ 构成 ABBA 死锁。
    size_t running = 0;
    for (const auto& t : mgr_.list_tasks())
        if (t.running) ++running;

    uint64_t qlen = 0, dropped = 0;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (const auto& [id, q] : queues_) {
            qlen += q->pending();
            dropped += q->dropped();
        }
    }
    return nlohmann::json{
        {"running_channels", running},
        {"event_queue_len", qlen},
        {"event_dropped_total", dropped},
    };
}

void AgentRuntime::on_created(const std::string& task_id, const AdaptedTask& t) {
    std::unique_ptr<EventPublisher> pub;
    if (t.events.transport == "mqtt") {
        MqttConfig mc;
        mc.broker = t.events.mqtt_broker;
        mc.topic = t.events.mqtt_topic.empty()
            ? "aistation/" + t.tenant + "/edge/" + opts_.edge_code + "/camera/" +
              std::to_string(t.camera.id) + "/detect"
            : t.events.mqtt_topic;
        mc.client_id = t.events.mqtt_client_id.empty()
            ? "aistation-agent-" + opts_.edge_code : t.events.mqtt_client_id;
        mc.username = t.events.mqtt_username;
        mc.password = t.events.mqtt_password;
        mc.qos = t.events.mqtt_qos;
        pub = std::make_unique<MqttPublisher>(mc);
    } else {
        pub = std::make_unique<HttpPublisher>(t.events.http_url, t.events.http_token);
    }
    auto queue = std::make_unique<DurableQueue>(
        t.events.buffer_dir + "/" + task_id, t.events.buffer_max_mb, pub.get());
    queue->start();

    EventMeta meta;
    meta.edge_code = opts_.edge_code;
    meta.camera_id = t.camera.id;
    int64_t task_id_num = 0;
    if (!t.sdk.id.empty()) {
        try {
            task_id_num = std::stoll(t.sdk.id);
        } catch (const std::exception& e) {
            std::cerr << "[AgentRuntime] non-numeric task_id '" << t.sdk.id
                      << "', using 0 for event routing: " << e.what() << std::endl;
        }
    }
    meta.task_id_num = task_id_num;
    meta.algorithm_type = t.algorithm_type;
    meta.alarm_interval_sec = t.alarm_interval_sec;

    {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_by_task_[meta.task_id_num] = queue.get();
        queues_[task_id] = std::move(queue);
        publishers_[task_id] = std::move(pub);
    }
    bus_.register_task(task_id, meta);
    mgr_.set_detection_sink(task_id, [this, task_id](const std::vector<DetectionBox>& boxes,
                                                     int w, int h, double lat) {
        bus_.on_detections(task_id, boxes, w, h, lat);
    });
}

void AgentRuntime::on_updated(const std::string& task_id, const AdaptedTask& t) {
    on_removed(task_id);
    on_created(task_id, t);
}

void AgentRuntime::on_removed(const std::string& task_id) {
    bus_.unregister_task(task_id);
    std::lock_guard<std::mutex> lk(mtx_);
    auto qit = queues_.find(task_id);
    if (qit != queues_.end()) {
        if (qit->second) qit->second->stop();
        for (auto it = queue_by_task_.begin(); it != queue_by_task_.end();)
            it = (it->second == qit->second.get()) ? queue_by_task_.erase(it) : std::next(it);
        queues_.erase(qit);
    }
    publishers_.erase(task_id);
}
