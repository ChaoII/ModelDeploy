#include "agent_server.hpp"
#include <iostream>
#include <thread>

using json = nlohmann::json;

AgentServer::AgentServer(PipelineManager& mgr, const ConfigAdapter& adapter,
                         const std::string& host, int port)
    : mgr_(mgr), adapter_(adapter), host_(host), port_(port) {}

AgentServer::~AgentServer() { stop(); }

std::string AgentServer::err_json(const std::string& msg, const std::string& code) {
    return json{{"error", {{"code", code}, {"message", msg}}}}.dump();
}

std::string AgentServer::ok_json(const json& data) {
    json r{{"ok", true}};
    if (data.is_object()) for (auto& [k, v] : data.items()) r[k] = v;
    return r.dump();
}

std::string AgentServer::fail_json(const std::string& msg, uint64_t generation) {
    return json{{"ok", false}, {"error", msg}, {"generation", generation}}.dump();
}

std::string AgentServer::get_id(const httplib::Request& req, const char* key) {
    auto it = req.path_params.find(key);
    return it != req.path_params.end() ? it->second : "";
}

json AgentServer::status_to_json(const TaskStatus& ts) {
    json j;
    j["task_id"] = ts.id;
    j["name"] = ts.name;
    j["running"] = ts.running;
    j["initialized"] = ts.initialized;
    j["error"] = ts.init_error;
    j["input_url"] = ts.input_url;
    j["preview_url"] = ts.preview_url;
    json models = json::array();
    for (size_t i = 0; i < ts.model_names.size(); ++i)
        models.push_back({{"name", ts.model_names[i]},
                          {"type", i < ts.model_types.size() ? ts.model_types[i] : ""}});
    j["models"] = models;
    return j;
}

bool AgentServer::start() {
    if (running_) return true;
    register_routes();
    // 同步绑定：bind_to_port 内部已完成 bind + listen(backlog)，失败立即返回，
    // 不再靠 sleep 后判活（避免绑定失败被误判为启动成功）。
    if (!server_.bind_to_port(host_, port_)) {
        std::cerr << "[AgentServer] failed to bind " << host_ << ":" << port_ << std::endl;
        return false;
    }
    thread_ = std::thread([this]() {
        if (!server_.listen_after_bind()) {
            std::cerr << "[AgentServer] listen stopped on " << host_ << ":" << port_ << std::endl;
            running_ = false;
        }
    });
    // 等待监听线程真正进入 listen_internal()（is_running_ 置位）再返回。
    // 否则调用方可能在监听就绪前就 stop()：此时 httplib 0.47 的 stop() 因
    // is_running_ 仍为 false 而不关闭 socket，导致 stop 无效、join() 阻塞、端口泄漏。
    server_.wait_until_ready();
    running_ = server_.is_running();
    return running_.load();
}

void AgentServer::stop() {
    running_ = false;
    server_.stop();
    if (thread_.joinable()) thread_.join();
}

void AgentServer::register_routes() {
    if (routes_registered_) return;   // 幂等：stop() 后再 start() 不重复追加路由
    routes_registered_ = true;
    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (req.path == "/health" || req.path == "/readyz") return httplib::Server::HandlerResponse::Unhandled;
        if (req.path.rfind("/api/v1/", 0) != 0) return httplib::Server::HandlerResponse::Unhandled;
        if (api_key_.empty()) return httplib::Server::HandlerResponse::Unhandled;
        // 常量时间比较（与 HttpServer 同等安全强度）
        auto ct_equal = [](const std::string& a, const std::string& b) {
            if (a.size() != b.size()) return false;
            unsigned char d = 0;
            for (size_t i = 0; i < a.size(); ++i)
                d |= static_cast<unsigned char>(a[i] ^ b[i]);
            return d == 0;
        };
        const std::string prefix = "Bearer ";
        auto it = req.headers.find("Authorization");
        bool ok = false;
        if (it != req.headers.end() && it->second.size() > prefix.size() &&
            it->second.compare(0, prefix.size(), prefix) == 0) {
            ok = ct_equal(it->second.substr(prefix.size()), api_key_);
        }
        if (ok) return httplib::Server::HandlerResponse::Unhandled;
        res.status = 401;
        res.set_content(err_json("invalid or missing API key", "UNAUTHORIZED"), "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });

    server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(ok_json({{"status", "ok"}}), "application/json");
    });

    server_.Get("/readyz", [this](const httplib::Request&, httplib::Response& res) {
        bool ready = true;
        for (const auto& t : mgr_.list_tasks()) {
            if (!t.init_error.empty()) { ready = false; break; }
            if (t.running && !t.initialized) { ready = false; break; }
        }
        if (!ready) { res.status = 503; res.set_content(err_json("not ready", "MODEL_NOT_READY"), "application/json"); return; }
        res.set_content(ok_json(), "application/json");
    });

    server_.Get("/api/v1/metrics", [this](const httplib::Request&, httplib::Response& res) {
        int running = 0;
        for (const auto& t : mgr_.list_tasks()) if (t.running) ++running;
        json m;
        m["tasks_total"] = mgr_.task_count();
        m["running_channels"] = running;
        if (metrics_provider_) {
            json extra = metrics_provider_();
            for (auto& [k, v] : extra.items()) m[k] = v;
        }
        res.set_content(ok_json({{"metrics", m}}), "application/json");
    });

    server_.Post("/api/v1/tasks", [this](const httplib::Request& req, httplib::Response& res) {
        AdaptedTask t;
        std::string err;
        try {
            if (!adapter_.from_json(json::parse(req.body), &t, &err)) {
                res.status = 400;
                res.set_content(err_json(err), "application/json");
                return;
            }
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(err_json(e.what()), "application/json");
            return;
        }
        if (!mgr_.create_task(t.sdk, &err)) {
            res.status = 400;
            res.set_content(err_json(err), "application/json");
            return;
        }
        if (hooks_.on_created) hooks_.on_created(t.sdk.id, t);
        res.set_content(ok_json({{"task_id", t.sdk.id}}), "application/json");
    });

    server_.Get("/api/v1/tasks", [this](const httplib::Request&, httplib::Response& res) {
        json items = json::array();
        for (const auto& t : mgr_.list_tasks()) items.push_back(status_to_json(t));
        res.set_content(ok_json({{"items", items}}), "application/json");
    });

    server_.Get("/api/v1/tasks/:id", [this](const httplib::Request& req, httplib::Response& res) {
        const std::string id = get_id(req);
        for (const auto& s : mgr_.list_tasks()) {
            if (s.id == id) {
                res.set_content(ok_json({{"task", status_to_json(s)}}), "application/json");
                return;
            }
        }
        res.status = 404;
        res.set_content(err_json("task not found", "NOT_FOUND"), "application/json");
    });

    server_.Post("/api/v1/tasks/:id/start", [this](const httplib::Request& req, httplib::Response& res) {
        const std::string id = get_id(req);
        TaskConfig cfg;
        if (!mgr_.get_task_config(id, &cfg)) {
            res.status = 404;
            res.set_content(err_json("task not found", "NOT_FOUND"), "application/json");
            return;
        }
        if (!mgr_.start_task(id)) {
            res.status = 503;
            res.set_content(err_json("model not ready", "MODEL_NOT_READY"), "application/json");
            return;
        }
        res.set_content(ok_json({{"running", true}}), "application/json");
    });

    server_.Post("/api/v1/tasks/:id/stop", [this](const httplib::Request& req, httplib::Response& res) {
        const std::string id = get_id(req);
        if (!mgr_.stop_task(id)) {
            res.status = 404;
            res.set_content(err_json("task not found", "NOT_FOUND"), "application/json");
            return;
        }
        res.set_content(ok_json({{"running", false}}), "application/json");
    });

    server_.Put("/api/v1/tasks/:id", [this](const httplib::Request& req, httplib::Response& res) {
        const std::string id = get_id(req);
        TaskConfig existing;
        if (!mgr_.get_task_config(id, &existing)) {
            res.status = 404;
            res.set_content(err_json("task not found", "NOT_FOUND"), "application/json");
            return;
        }
        AdaptedTask t;
        std::string err;
        try {
            if (!adapter_.from_json(json::parse(req.body), &t, &err)) {
                res.status = 400;
                res.set_content(err_json(err), "application/json");
                return;
            }
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(err_json(e.what()), "application/json");
            return;
        }
        t.sdk.id = id;                                   // 保持原 id
        mgr_.stop_task(id);                              // 先停后改
        if (!mgr_.update_task(id, t.sdk, &err)) {
            res.status = 400;
            res.set_content(err_json(err), "application/json");
            return;
        }
        if (hooks_.on_updated) hooks_.on_updated(id, t);
        res.set_content(ok_json(), "application/json");
    });

    server_.Delete("/api/v1/tasks/:id", [this](const httplib::Request& req, httplib::Response& res) {
        const std::string id = get_id(req);
        if (!mgr_.remove_task(id)) {
            res.status = 404;
            res.set_content(err_json("task not found", "NOT_FOUND"), "application/json");
            return;
        }
        if (hooks_.on_removed) hooks_.on_removed(id);
        res.set_content(ok_json(), "application/json");
    });

    // ── POST /api/v1/tasks/:id/models/:name/update — 模型热更新（COW 热交换） ──
    // 语义：任务不存在 → 404；模型名不存在 → 404（本接口只更新既有模型，不做隐式追加）；
    // body 非法 → 400；引擎加载失败/未就绪 → ok:false 且保留旧模型（fail-safe）。
    server_.Post("/api/v1/tasks/:id/models/:name/update",
        [this](const httplib::Request& req, httplib::Response& res) {
        const std::string id = get_id(req, "id");
        const std::string name = get_id(req, "name");
        const uint64_t gen = generation_.load();
        TaskConfig existing;
        if (!mgr_.get_task_config(id, &existing)) {
            res.status = 404;
            res.set_content(fail_json("任务不存在: " + id, gen), "application/json");
            return;
        }
        bool model_exists = false;
        for (const auto& m : existing.models)
            if (m.name == name) { model_exists = true; break; }
        if (!model_exists) {
            res.status = 404;
            res.set_content(fail_json("任务中不存在模型: " + name, gen), "application/json");
            return;
        }
        // 复用 /tasks 的 models[] 字段映射解析单模型 ModelConfig（parse_model 与 from_json 同源）
        ModelConfig mcfg;
        std::string err;
        try {
            json body = json::parse(req.body);
            // 路径参数为权威：body 缺 name 时补入，避免强制 body 自带 name
            if (body.is_object() && !body.contains("name")) body["name"] = name;
            if (!adapter_.parse_model(body, &mcfg, &err)) {
                res.status = 400;
                res.set_content(fail_json(err.empty() ? "模型配置非法" : err, gen), "application/json");
                return;
            }
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(fail_json(std::string("模型配置解析失败: ") + e.what(), gen), "application/json");
            return;
        }
        mcfg.name = name;   // 路径参数为准（与既有 PATCH /models/:name 语义一致）
        // 引擎就绪校验（关键 fail-safe）：create_engine 对 cls/ocr/lpr 等在加载失败时仍返回
        // 非空引擎，仅判空会把未加载引擎换入 InferGroup。换入前必须显式校验 is_loaded()。
        // 说明：此处返回的引擎仅作就绪探针即时使用；其 det_model()/config() 返回的裸指针
        // 指向快照内部，只在本次调用内有效，绝不长期持有或跨热更新使用。
        try {
            auto probe = mgr_.create_engine(mcfg);
            if (!probe || !probe->is_loaded()) {
                res.set_content(fail_json("模型加载失败，已保留旧模型: " + name, gen), "application/json");
                return;
            }
        } catch (const std::exception& e) {
            res.set_content(fail_json(std::string("模型加载异常，已保留旧模型: ") + e.what(), gen),
                            "application/json");
            return;
        }
        if (!mgr_.update_model(id, name, mcfg)) {
            res.set_content(fail_json("模型热更新失败，已保留旧模型: " + name, gen), "application/json");
            return;
        }
        const uint64_t next = generation_.fetch_add(1) + 1;   // 仅成功换入才递增
        res.set_content(ok_json({{"generation", next}}), "application/json");
    });

    server_.Get("/api/v1/tasks/:id/stats", [this](const httplib::Request& req, httplib::Response& res) {
        std::string stats;
        if (!mgr_.get_task_stats(get_id(req), &stats)) {
            res.status = 404;
            res.set_content(err_json("task not found", "NOT_FOUND"), "application/json");
            return;
        }
        try { res.set_content(ok_json({{"stats", json::parse(stats)}}), "application/json"); }
        catch (...) { res.set_content(ok_json({{"stats", json::object()}}), "application/json"); }
    });

    server_.Get("/api/v1/tasks/:id/snapshot.jpg", [this](const httplib::Request& req, httplib::Response& res) {
        std::vector<uint8_t> jpg;
        if (!mgr_.get_task_jpeg(get_id(req), &jpg, 80) || jpg.empty()) {
            res.status = 404;
            res.set_content(err_json("snapshot unavailable", "NOT_FOUND"), "application/json");
            return;
        }
        res.set_header("Cache-Control", "no-store");
        res.set_content(reinterpret_cast<const char*>(jpg.data()), jpg.size(), "image/jpeg");
    });
}
