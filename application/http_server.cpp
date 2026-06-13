#include "http_server.hpp"
#include <iostream>
#include <fstream>

HttpServer::HttpServer(PipelineManager& mgr, const std::string& host, int port)
    : mgr_(mgr), host_(host), port_(port) {
}

HttpServer::~HttpServer() {
    stop();
}

bool HttpServer::start() {
    if (running_) return true;
    register_routes();

    std::promise<void> started;
    auto started_fut = started.get_future();

    running_ = true;
    server_.set_start_handler([&started]() { started.set_value(); });

    server_thread_ = std::thread([this]() {
        std::cout << "[HttpServer] " << host_ << ":" << port_ << std::endl;
        if (!server_.listen(host_.c_str(), port_)) {
            std::cerr << "[HttpServer] listen failed" << std::endl;
            running_ = false;
        }
    });

    // 等待服务器开始监听
    started_fut.wait_for(std::chrono::seconds(5));
    return running_.load();
}

void HttpServer::stop() {
    running_ = false;
    server_.stop();
    if (server_thread_.joinable())
        server_thread_.join();
}

// ── JSON helpers ──────────────────────────────

std::string HttpServer::err_json(const std::string& msg) {
    return "{\"ok\":false,\"msg\":\"" + msg + "\"}";
}

std::string HttpServer::ok_json(const std::string& data) {
    if (data.empty() || data[0] != '{')
        return "{\"ok\":true}";
    return "{\"ok\":true," + data.substr(1);
}

std::string HttpServer::task_list_json(const std::vector<TaskStatus>& tasks) {
    std::string j = "{\"ok\":true,\"tasks\":[";
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (i) j += ",";
        j += "{\"id\":\"" + tasks[i].id + "\",\"name\":\"" + tasks[i].name
          + "\",\"running\":" + (tasks[i].running ? "true" : "false")
          + ",\"input_url\":\"" + tasks[i].input_url
          + "\",\"output_url\":\"" + tasks[i].output_url + "\"";
        if (!tasks[i].model_names.empty()) {
            j += ",\"models\":[";
            for (size_t k = 0; k < tasks[i].model_names.size(); ++k) {
                if (k) j += ",";
                j += "{\"name\":\"" + tasks[i].model_names[k]
                  + "\",\"type\":\"" + tasks[i].model_types[k] + "\"}";
            }
            j += "]";
        }
        j += "}";
    }
    j += "]}";
    return j;
}

std::string HttpServer::task_config_json(const TaskConfig& cfg) {
    return "{\"ok\":true,\"task\":" + task_config_to_json(cfg).dump() + "}";
}

// ── Route helpers ─────────────────────────────

static std::string get_id(const httplib::Request& req) {
    auto it = req.path_params.find("id");
    return it != req.path_params.end() ? it->second : "";
}

static ModelConfig parse_model_config(const std::string& body) {
    ModelConfig mcfg;
    try {
        auto j = json::parse(body);
        if (!j.is_object()) return mcfg;
        auto get_s = [&](const std::string& k, const std::string& d = "") -> std::string {
            return j.contains(k) && j[k].is_string() ? j[k].get<std::string>() : d;
        };
        mcfg.name = get_s("name");
        mcfg.path = get_s("path");
        mcfg.type = get_s("type", "detection");
        mcfg.backend = get_s("backend", "ort");
        mcfg.device = get_s("device", "gpu");
        if (j.contains("confidence_threshold")) mcfg.confidence_threshold = j["confidence_threshold"];
        if (j.contains("input_size") && j["input_size"].is_array() && j["input_size"].size() >= 2)
            mcfg.input_size = {j["input_size"][0], j["input_size"][1]};
        if (j.contains("roi") && j["roi"].is_array() && j["roi"].size() >= 4)
            mcfg.roi = {j["roi"][0], j["roi"][1], j["roi"][2], j["roi"][3]};
        if (j.contains("interval")) mcfg.interval = j["interval"];
    } catch (...) {}
    return mcfg;
}

// ── Routes ────────────────────────────────────

void HttpServer::register_routes() {
    // Web UI
    server_.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream f("E:\\CLionProjects\\ModelDeploy\\application\\web_ui.html");
        if (f.is_open()) {
            std::string html((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
            res.set_content(html, "text/html; charset=utf-8");
        } else {
            res.set_content("<h1>Web UI not found</h1>", "text/html");
        }
    });
    server_.Get("/index.html", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream f("E:\\CLionProjects\\ModelDeploy\\application\\web_ui.html");
        if (f.is_open()) {
            std::string html((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
            res.set_content(html, "text/html; charset=utf-8");
        } else {
            res.set_content("<h1>Web UI not found</h1>", "text/html");
        }
    });

    // Health
    server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"ok\":true}", "application/json");
    });

    // POST /api/v1/tasks — create
    server_.Post("/api/v1/tasks", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto cfg = task_config_from_json(json::parse(req.body));
            if (!mgr_.create_task(cfg)) {
                res.set_content(err_json("create failed"), "application/json");
                return;
            }
            res.set_content(ok_json(), "application/json");
        } catch (const std::exception& e) {
            res.set_content(err_json(e.what()), "application/json");
        }
    });

    // GET /api/v1/tasks — list
    server_.Get("/api/v1/tasks", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(task_list_json(mgr_.list_tasks()), "application/json");
    });

    // GET /api/v1/tasks/:id — config
    server_.Get("/api/v1/tasks/:id", [this](const httplib::Request& req, httplib::Response& res) {
        TaskConfig cfg;
        if (!mgr_.get_task_config(get_id(req), &cfg)) {
            res.set_content(err_json("not found"), "application/json");
            return;
        }
        res.set_content(ok_json("\"task\":" + task_config_to_json(cfg).dump()), "application/json");
    });

    // DELETE /api/v1/tasks/:id — remove
    server_.Delete("/api/v1/tasks/:id", [this](const httplib::Request& req, httplib::Response& res) {
        if (!mgr_.remove_task(get_id(req))) {
            res.set_content(err_json("not found"), "application/json");
            return;
        }
        res.set_content(ok_json(), "application/json");
    });

    // POST /api/v1/tasks/:id/start
    server_.Post("/api/v1/tasks/:id/start", [this](const httplib::Request& req, httplib::Response& res) {
        if (!mgr_.start_task(get_id(req))) {
            res.set_content(err_json("start failed"), "application/json");
            return;
        }
        res.set_content(ok_json(), "application/json");
    });

    // POST /api/v1/tasks/:id/stop
    server_.Post("/api/v1/tasks/:id/stop", [this](const httplib::Request& req, httplib::Response& res) {
        if (!mgr_.stop_task(get_id(req))) {
            res.set_content(err_json("stop failed"), "application/json");
            return;
        }
        res.set_content(ok_json(), "application/json");
    });

    // PATCH /api/v1/tasks/:id/models — add model
    server_.Patch("/api/v1/tasks/:id/models", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto mcfg = parse_model_config(req.body);
            if (mcfg.name.empty()) {
                res.set_content(err_json("name required"), "application/json");
                return;
            }
            if (!mgr_.add_model(get_id(req), mcfg)) {
                res.set_content(err_json("add model failed"), "application/json");
                return;
            }
            res.set_content(ok_json(), "application/json");
        } catch (const std::exception& e) {
            res.set_content(err_json(e.what()), "application/json");
        }
    });

    // DELETE /api/v1/tasks/:id/models/:name — remove model
    server_.Delete("/api/v1/tasks/:id/models/:name",
        [this](const httplib::Request& req, httplib::Response& res) {
        auto id = req.path_params.find("id");
        auto name = req.path_params.find("name");
        if (id == req.path_params.end() || name == req.path_params.end()) {
            res.set_content(err_json("missing params"), "application/json");
            return;
        }
        if (!mgr_.remove_model(id->second, name->second)) {
            res.set_content(err_json("remove failed"), "application/json");
            return;
        }
        res.set_content(ok_json(), "application/json");
    });

    // PATCH /api/v1/tasks/:id/models/:name — update model
    server_.Patch("/api/v1/tasks/:id/models/:name",
        [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto id = req.path_params.find("id");
            auto name = req.path_params.find("name");
            if (id == req.path_params.end() || name == req.path_params.end()) {
                res.set_content(err_json("missing params"), "application/json");
                return;
            }
            auto mcfg = parse_model_config(req.body);
            mcfg.name = name->second;
            if (!mgr_.update_model(id->second, name->second, mcfg)) {
                res.set_content(err_json("update failed"), "application/json");
                return;
            }
            res.set_content(ok_json(), "application/json");
        } catch (const std::exception& e) {
            res.set_content(err_json(e.what()), "application/json");
        }
    });
}
