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
          + "\",\"running\":" + (tasks[i].running ? "true" : "false") + "}";
    }
    j += "]}";
    return j;
}

std::string HttpServer::task_config_json(const TaskConfig& cfg) {
    auto jv = task_config_to_json(cfg);
    return "{\"ok\":true,\"task\":" + jv.as_object().fields.at("id").as_string() + "}";
    // simplified — for now just return basic info
}

// ── Route helpers ─────────────────────────────

static std::string get_id(const httplib::Request& req) {
    auto it = req.path_params.find("id");
    return it != req.path_params.end() ? it->second : "";
}

static ModelConfig parse_model_config(const std::string& body) {
    ModelConfig mcfg;
    auto j = parse_json(body);
    if (!j.is_object()) return mcfg;
    mcfg.name = JsonValue::get_string(j, "name", "");
    mcfg.path = JsonValue::get_string(j, "path", "");
    mcfg.type = JsonValue::get_string(j, "type", "detection");
    mcfg.backend = JsonValue::get_string(j, "backend", "ort");
    mcfg.device = JsonValue::get_string(j, "device", "gpu");
    mcfg.confidence_threshold = static_cast<float>(
        JsonValue::get_number(j, "confidence_threshold", 0.5));
    auto size = JsonValue::get_array(j, "input_size");
    if (size.size() >= 2) {
        mcfg.input_size = {size[0].as_int(), size[1].as_int()};
    }
    auto roi = JsonValue::get_array(j, "roi");
    if (roi.size() >= 4) {
        mcfg.roi = {roi[0].as_int(), roi[1].as_int(), roi[2].as_int(), roi[3].as_int()};
    }
    mcfg.interval = JsonValue::get_int(j, "interval", 1);
    return mcfg;
}

// ── Routes ────────────────────────────────────

void HttpServer::register_routes() {
    // Web UI
    server_.Get("/", [this](const httplib::Request&, httplib::Response& res) {
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
            auto cfg = task_config_from_json(parse_json(req.body));
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
        res.set_content(ok_json("\"task\":" + task_config_to_json(cfg).as_string()), "application/json");
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
