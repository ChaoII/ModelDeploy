#include "agent_server.hpp"
#include <chrono>
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
    running_ = true;
    thread_ = std::thread([this]() {
        if (!server_.listen(host_.c_str(), port_)) {
            std::cerr << "[AgentServer] failed to bind " << host_ << ":" << port_ << std::endl;
            running_ = false;
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return running_.load();
}

void AgentServer::stop() {
    running_ = false;
    server_.stop();
    if (thread_.joinable()) thread_.join();
}

void AgentServer::register_routes() {
    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (req.path == "/health" || req.path == "/readyz") return httplib::Server::HandlerResponse::Unhandled;
        if (req.path.rfind("/api/v1/", 0) != 0) return httplib::Server::HandlerResponse::Unhandled;
        if (api_key_.empty()) return httplib::Server::HandlerResponse::Unhandled;
        const std::string prefix = "Bearer ";
        auto it = req.headers.find("Authorization");
        if (it != req.headers.end() && it->second.compare(0, prefix.size(), prefix) == 0 &&
            it->second.substr(prefix.size()) == api_key_)
            return httplib::Server::HandlerResponse::Unhandled;
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
        if (!mgr_.start_task(id)) {
            res.status = 404;
            res.set_content(err_json("task not found", "NOT_FOUND"), "application/json");
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
            res.set_content("snapshot unavailable", "text/plain");
            return;
        }
        res.set_header("Cache-Control", "no-store");
        res.set_content(reinterpret_cast<const char*>(jpg.data()), jpg.size(), "image/jpeg");
    });
}
