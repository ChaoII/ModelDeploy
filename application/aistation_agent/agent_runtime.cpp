#include "agent_runtime.hpp"

#include <chrono>
#include <iostream>
#include <iterator>
#include <thread>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "agent_util.hpp"

AgentRuntime::AgentRuntime(AgentOptions opts)
    : opts_(std::move(opts)),
      fetcher_(opts_.model_cache_dir.empty() ? "data/model_cache" : opts_.model_cache_dir,
               opts_.s3_endpoint),
      adapter_(&fetcher_) {
    // 本地模型加载白名单（安全审计 HIGH #6）：设置后限制 file:// / 裸路径的加载范围
    fetcher_.set_local_root(opts_.model_root);
}

AgentRuntime::~AgentRuntime() { stop(); }

bool AgentRuntime::start() {
    // 控制面鉴权 fail-closed（安全审计 BLOCKER）：未配置 --api-key 时**拒绝启动**，
    // 除非显式 --allow-anonymous（仅限本地开发）。否则默认部署即"控制面裸奔"：
    // 任意可达主机可下发/启停/删除任务、热更新模型（file:// 任意本地文件、http(s) SSRF）。
    // 注意心跳 token 的缺省回退（--secret 为空时用 --api-key）不能替代控制面密钥。
    if (opts_.api_key.empty() && !opts_.allow_anonymous) {
        std::cerr << "[AgentRuntime] refusing to start: control plane requires an API key but "
                     "--api-key is empty (fail-closed). Pass --api-key <secret>; for local "
                     "development only you may pass --allow-anonymous to run without authentication."
                  << std::endl;
        return false;
    }
    if (opts_.allow_anonymous) {
        std::cerr << "[AgentRuntime] WARNING: --allow-anonymous is set: the control plane will accept "
                     "unauthenticated requests. Use only on a trusted/isolated network; never in production."
                  << std::endl;
    }
    if (opts_.model_root.empty()) {
        std::cerr << "[AgentRuntime] WARNING: --model-root is not set: local model paths from the cloud "
                     "(file:// or plain paths) are accepted without directory confinement. "
                     "Set --model-root <dir> to restrict them." << std::endl;
    }

    bus_.set_sink([this](const DetectionEvent& e) {
        // 仅在锁内查找并拷贝队列 shared_ptr；enqueue 含磁盘 I/O，必须在锁外执行，
        // 否则检测线程会长时间阻塞在 mtx_ 上（与 PipelineManager 形成锁序风险）。
        // 持 shared_ptr 而非裸指针：heartbeat 线程可能在 on_removed 摘除并停止/析构
        // 队列的同时仍在调用本 sink，shared_ptr 保证 enqueue 期间对象存活（防 UAF）。
        std::shared_ptr<DurableQueue> q;
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
    server_->set_allow_anonymous(opts_.allow_anonymous);
    server_->set_hooks(std::move(hooks));
    server_->set_metrics_provider([this]() { return metrics(); });
    if (!server_->start()) return false;

    heartbeat_ = std::make_unique<Heartbeat>(
        opts_.cloud_url, opts_.edge_code,
        opts_.secret.empty() ? opts_.api_key : opts_.secret,
        [this]() { return metrics(); }, opts_.heartbeat_interval_sec, opts_.max_channels);
    heartbeat_->start();

    schedule_running_ = true;
    schedule_thread_ = std::thread([this]() { schedule_loop(); });

    heartbeat_running_ = true;
    heartbeat_thread_ = std::thread([this]() { heartbeat_loop(); });
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
    schedule_running_ = false;
    if (schedule_thread_.joinable()) schedule_thread_.join();

    heartbeat_running_ = false;
    if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
    mgr_.stop_all();
    // 队列 worker 的 join 必须在 mtx_ 之外：worker 可能经 snapshot provider 走
    // PipelineManager::get_task_jpeg（持其 mtx_），而其它路径持 PipelineManager::mtx_
    // 时经 sink 取 AgentRuntime::mtx_，持锁 join 会形成 ABBA 死锁。
    std::vector<std::shared_ptr<DurableQueue>> queues;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (auto& entry : queues_)
            if (entry.second) queues.push_back(entry.second);
        queues_.clear();
        queue_by_task_.clear();
    }
    for (auto& q : queues) q->stop();  // join 在锁外
    {
        std::lock_guard<std::mutex> lk(mtx_);
        // 队列已停，worker 不再持有 EventPublisher*，此时清理发布器才安全。
        publishers_.clear();
        schedules_.clear();
        heartbeats_.clear();
    }
}

nlohmann::json AgentRuntime::metrics() {
    // 锁序：先在 mtx_ 之外访问 PipelineManager（其 mtx_ 与检测线程 join 相关），
    // 再单独持 mtx_ 汇总本类队列；避免 mtx_ -> PipelineManager::mtx_ 与检测
    // 线程反向获取 mtx_ 构成 ABBA 死锁。
    size_t running = 0;
    for (const auto& t : mgr_.list_tasks())
        if (t.running) ++running;

    uint64_t qlen = 0, dropped = 0, write_fail = 0, send_fail = 0;
    size_t qbytes = 0;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (const auto& [id, q] : queues_) {
            qlen += q->pending();
            dropped += q->dropped();
            write_fail += q->write_failed();
            send_fail += q->send_failed();
            qbytes += q->pending_bytes();
        }
    }
    return nlohmann::json{
        {"running_channels", running},
        {"event_queue_len", qlen},
        {"event_queue_bytes", qbytes},
        {"event_dropped_total", dropped},
        // 合同审计 HIGH：磁盘写失败（事件必丢）与传输失败需可观测，不再静默
        {"event_write_fail_total", write_fail},
        {"event_send_fail_total", send_fail},
        // 合同审计 HIGH 3.4：心跳失败需可观测（判断边缘是否已与云端失联）
        {"heartbeat_fail_total", heartbeat_ ? heartbeat_->total_failures() : 0},
        {"heartbeat_consecutive_failures", heartbeat_ ? heartbeat_->consecutive_failures() : 0},
        {"heartbeat_last_status", heartbeat_ ? heartbeat_->last_failure_status() : 0},
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
        // 每任务唯一 client_id（防御云端对同一边缘所有任务下发同一 id）：本任务独占一条
        // MQTT 连接，若与其它任务共用 id，broker 会按规范断开旧连接 → 任务间相互顶下线。
        mc.client_id = aistation::mqtt_client_id_for(t.events.mqtt_client_id, opts_.edge_code,
                                                     task_id);
        mc.username = t.events.mqtt_username;
        mc.password = t.events.mqtt_password;
        mc.qos = t.events.mqtt_qos;
        pub = std::make_unique<MqttPublisher>(mc);
    } else {
        pub = std::make_unique<HttpPublisher>(t.events.http_url, t.events.http_token);
    }
    auto queue = std::make_shared<DurableQueue>(
        t.events.buffer_dir + "/" + task_id, t.events.buffer_max_mb, pub.get());

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
    meta.schedule = t.schedule;
    meta.roi_polygon = t.roi_polygon;

    // 必须在 start() 之前安装 provider，避免队列中的持久化残留先于 provider 发布。
    // 配置在 task 生命周期内不可变，故按值捕获 SnapshotConfig，无需加锁查表。
    const SnapshotConfig snap_cfg = t.snapshot;
    queue->set_snapshot_provider([this, task_id, snap_cfg](int64_t, std::string* b64, int* w, int* h) {
        if (!snap_cfg.enabled || !snap_cfg.inline_snapshot) return false;
        return capture_snapshot_b64(task_id, snap_cfg, b64, w, h);
    });
    queue->start();

    {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_by_task_[meta.task_id_num] = queue;
        queues_[task_id] = queue;
        publishers_[task_id] = std::move(pub);
        schedules_[task_id] = SchedEntry{t.schedule, false};
        heartbeats_[task_id] = HeartbeatEntry{t.heartbeat_sec, {}};
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
    // 队列与发布器必须在同一次持锁中一起移出：worker 可能经 snapshot provider 调
    // PipelineManager::get_task_jpeg（持其 mtx_），与持 PipelineManager::mtx_ 经 sink
    // 取 mtx_ 的路径构成 ABBA，故绝不能持 mtx_ 做 stop()/join。
    // 若分两次持锁（先摘队列、停后再 erase 发布器），在锁外 stop() 的窗口内并发的
    // on_created（同一 task_id）会安装新的 queue+publisher，第二次持锁的
    // publishers_.erase 将误删新发布器，使新队列 worker 持有的 EventPublisher* 悬空（UAF）。
    // 用 shared_ptr 拷贝（而非 move）：EventBus sink / heartbeat_loop 可能在 mtx_ 外
    // 持有同一队列的强引用并调用 enqueue；shared_ptr 让本函数 stop() 与析构不再发生在
    // 在途 enqueue 期间（避免写已释放队列导致堆破坏）。
    std::shared_ptr<DurableQueue> queue;
    std::unique_ptr<EventPublisher> publisher;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        auto qit = queues_.find(task_id);
        if (qit != queues_.end()) {
            queue = qit->second;
            for (auto it = queue_by_task_.begin(); it != queue_by_task_.end();)
                it = (it->second == queue) ? queue_by_task_.erase(it) : std::next(it);
            queues_.erase(qit);
        }
        // 与队列同锁移出发布器，保证本次删除只作用于当前这一代对象。
        auto pit = publishers_.find(task_id);
        if (pit != publishers_.end()) {
            publisher = std::move(pit->second);
            publishers_.erase(pit);
        }
        schedules_.erase(task_id);
        heartbeats_.erase(task_id);
    }
    if (queue) queue->stop();  // join 必须在 mtx_ 之外，避免与 PipelineManager 形成 ABBA
    // queue 已停（worker 已 join、不再持有 EventPublisher*），publisher 局部变量
    // 才在函数返回时析构，避免 UAF/TOCTOU。
}

void AgentRuntime::schedule_loop() {
    while (schedule_running_.load()) {
        // 调度线程必须自吞异常：从 mgr_（list_tasks/start_task/stop_task）逃逸的任何异常，
        // 若未捕获，会经 std::thread 的未捕获异常路径触发 std::terminate，直接杀死整个 Agent
        // 进程（Windows 上表现为 0xC0000409 静默退出：无 C++ 崩溃输出、无 WER 转储、控制面端口
        // 随即拒绝连接）。这与检测线程、心跳线程、持久化队列线程的异常处理方式保持一致。
        try {
            // 取一次运行态，避免每任务重复查询
            std::unordered_map<std::string, bool> running;
            for (const auto& st : mgr_.list_tasks()) running[st.id] = st.running;

            std::vector<std::pair<std::string, bool>> actions;  // (task_id, should_start)
            {
                std::lock_guard<std::mutex> lk(mtx_);
                for (auto& [id, e] : schedules_) {
                    const bool active = aistation::schedule_active_now(e.schedule);
                    const bool is_running = running.count(id) ? running[id] : false;
                    if (!active) {
                        // 窗口外：确保停止；只有由本控制器停止的任务才会在窗口开启后自动恢复
                        if (is_running) {
                            actions.emplace_back(id, false);
                            e.stopped_by_us = true;
                        }
                    } else if (!is_running && e.stopped_by_us) {
                        actions.emplace_back(id, true);
                        e.stopped_by_us = false;
                    }
                }
            }
            for (const auto& [id, should_start] : actions) {
                if (should_start) mgr_.start_task(id);
                else mgr_.stop_task(id);
            }
        } catch (const std::exception& e) {
            std::cerr << "[Schedule] tick failed: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[Schedule] tick failed: unknown" << std::endl;
        }
        for (int i = 0; i < 300 && schedule_running_.load(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));  // ~30s
    }
}

void AgentRuntime::heartbeat_loop() {
    // 定时线程必须自吞异常：未捕获异常会经 std::thread 触发 std::terminate 杀死进程
    // （与 schedule_loop / 心跳上报 / 持久化队列线程一致）。1s 粒度保证 heartbeat_sec
    // 为秒级时到期足够准时。
    while (heartbeat_running_.load()) {
        try {
            std::unordered_map<std::string, bool> running;
            for (const auto& st : mgr_.list_tasks()) running[st.id] = st.running;

            const auto now = std::chrono::steady_clock::now();
            std::vector<std::string> due;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                for (auto& [id, e] : heartbeats_) {
                    const bool is_running = running.count(id) ? running[id] : false;
                    if (!is_running) continue;
                    if (!aistation::heartbeat_due(e.last, e.interval_sec, now)) continue;
                    e.last = now;
                    due.push_back(id);
                }
            }
            // 在锁外发布，避免订阅者回调路径取 mtx_ 造成 ABBA
            for (const auto& id : due) bus_.on_heartbeat(id);
        } catch (const std::exception& e) {
            std::cerr << "[HeartbeatLoop] tick failed: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[HeartbeatLoop] tick failed: unknown" << std::endl;
        }
        for (int i = 0; i < 10 && heartbeat_running_.load(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));  // ~1s
    }
}

bool AgentRuntime::capture_snapshot_b64(const std::string& task_id, const SnapshotConfig& cfg,
                                        std::string* b64, int* w, int* h) {
    // 检测线程在首帧的 sink 回调之后才写入 latest_bgr_ 快照，而队列 worker 可能已经先处理
    // 首条事件：此刻快照尚未就绪，会导致首条事件（进而首个告警）缺失内联快照。
    // 这里轮询等待首帧快照就绪（宽松上限 ~4s，覆盖慢解码/慢推理的启动阶段）。仅首条事件会
    // 等待：一旦快照就绪，后续事件立即命中；任务停止/删除时 get_task_jpeg 立即失败并快速放弃。
    std::vector<uint8_t> jpg;
    bool ready = false;
    for (int attempt = 0; attempt < 160; ++attempt) {
        if (mgr_.get_task_jpeg(task_id, &jpg, cfg.quality) && !jpg.empty()) {
            ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    if (!ready) return false;
    cv::Mat img = cv::imdecode(jpg, cv::IMREAD_COLOR);
    if (img.empty()) return false;
    if (cfg.max_width > 0 && img.cols > cfg.max_width) {
        const double scale = static_cast<double>(cfg.max_width) / static_cast<double>(img.cols);
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(), scale, scale, cv::INTER_AREA);
        img = resized;
    }
    std::vector<uint8_t> out;
    std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, cfg.quality};
    if (!cv::imencode(".jpg", img, out, params) || out.empty()) return false;
    *b64 = aistation::base64_encode_bytes(out);
    *w = img.cols;
    *h = img.rows;
    return true;
}
